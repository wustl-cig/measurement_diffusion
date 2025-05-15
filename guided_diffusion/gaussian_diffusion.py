"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""


# -----------------
# Importing from Python module
# -----------------
import enum
import math
import numpy as np
import torch as th
import torch

# -----------------
# Importing from files
# -----------------
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from utility.func_utility import plot_multiples_in_one
from datasets.fastMRI import fmult, ftran, ftran_non_mask, apply_mask_on_kspace_wthout_ftranfmult, get_fastmri_mask
from datasets.ffhq import get_ffhq_mask

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, diffusion_model_type, training_data_noiselevel):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        assert scale == 1
        
        beta_start = scale * 0.0001
        beta_end = scale * 0.02

        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )

    elif schedule_name == "cosine":
        raise ValueError("Not fullly implemented for this case.")
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )

    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL



class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)
        
        self.diffusion_sigma_t = self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, training_data_noiselevel = None, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        
        coef1 = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        coef2 = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        original_noise_std = (coef2/coef1)
        
        # index value of coef1 which is the array
        if training_data_noiselevel is None:
            added_noise_std = original_noise_std
        else:
            training_data_noiselevel = torch.tensor(training_data_noiselevel, device=coef1.device)
            training_data_noiselevel = training_data_noiselevel.expand_as(coef1)
            if original_noise_std.flatten()[0] < training_data_noiselevel.flatten()[0]:
                raise ValueError(f"x_start should be denoised image by denoisers")
                added_noise_std = original_noise_std
            else:
                original_noise_var = torch.square(original_noise_std)
                training_data_noiselevel_var = torch.square(training_data_noiselevel)
                added_noise_std = torch.sqrt(torch.abs(original_noise_var - training_data_noiselevel_var))
                # print(f"added_noise_std: {added_noise_std}")
        
        return (coef1 * (x_start + (added_noise_std) * noise))
    
    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, param_dicts, measurement = None, measurement_cond_fn = None, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        
        param_dicts['num_time_steps'] = self.num_timesteps
        param_dicts['alphas_cumprod_t'] = _extract_into_tensor(self.alphas_cumprod, self._scale_timesteps(t), x.shape)
        param_dicts['alphas_cumprod_t_plus_1'] = _extract_into_tensor(self.alphas_cumprod_next, self._scale_timesteps(t), x.shape)

        param_dicts['sqrt_alphas_cumprod_t'] = _extract_into_tensor(self.sqrt_alphas_cumprod, self._scale_timesteps(t), x.shape)
        param_dicts['sqrt_one_minus_alphas_cumprod_t'] = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, self._scale_timesteps(t), x.shape)
        param_dicts['sqrt_recip_alphas_cumprod'] = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, self._scale_timesteps(t), x.shape)
        param_dicts['sqrt_recipm1_alphas_cumprod'] = sqrt_recipm1_alphas_cumprod = _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, self._scale_timesteps(t), x.shape)
        
        model_output = model(x, self._scale_timesteps(t), param_dicts = param_dicts, **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            raise ValueError("Our code is not intended for this condition")
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)
            #  PREVIOUS IMPLEMENTATION
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            raise ValueError("Our code is not intended for this condition")
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
                
                et = self._predict_eps_from_xstart(x_t=x, t=t, pred_xstart=model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
                et = model_output
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "et": et,
        }

    def scaling_timestep_to_1000scale(self, unscaled_t):
        scaling_constant = int(1000/self.num_timesteps)
        scaled_t = unscaled_t*scaling_constant
        return scaled_t
        

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        # print(f"self.rescale_timesteps: {self.rescale_timesteps}")
        # raise ValueError(f"self.rescale_timesteps: {self.rescale_timesteps}")
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        param_dicts,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        
        out = self.p_mean_variance(
            model,
            x,
            t,
            param_dicts = param_dicts,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            raise ValueError("Maybe not intended")
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}


    def p_sample_loop(
        self,
        model,
        shape,
        param_dicts,
        x_gt = None,
        measurement = None,
        measurement_cond_fn = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        progress = True

        final = None

        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            param_dicts = param_dicts,
            x_gt = x_gt,
            measurement = measurement,
            measurement_cond_fn = measurement_cond_fn,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        param_dicts,
        x_gt = None,
        measurement = None,
        measurement_cond_fn = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            raise ValueError("Maybe not intended for this")
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)
            
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    param_dicts = param_dicts,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]
                
    def ddim_sample(
        self,
        model,
        x,
        t,
        param_dicts,
        measurement = None,
        measurement_cond_fn = None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):  
        # print(f"eta: {eta}")
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        param_dicts['ddim_eta'] = eta

        if param_dicts['diffusion_model_type'] in ['gsure_diffusion', 'unconditional_diffusion', 'measurement_diffusion', 'ambient_diffusion']:
            out = self.p_mean_variance(
                model,
                x,
                t,
                measurement = measurement,
                measurement_cond_fn = measurement,
                param_dicts = param_dicts,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])
        
        else:
            raise ValueError(f"Unknown diffusion model type: {param_dicts['diffusion_model_type']}")
            
        if cond_fn is not None:
            raise ValueError("Maybe not intended for this")
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
            
        
        if param_dicts['solve_inverse_problem'] == False:
            alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
            sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
            )
            # Equation 12.
            noise = th.randn_like(x)
            mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
            )
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
            sample = mean_pred + nonzero_mask * sigma * noise
            return {"sample": sample, "pred_xstart": out["pred_xstart"]}

        elif param_dicts['solve_inverse_problem'] == True and param_dicts['diffusion_model_type'] in ['measurement_diffusion']:

            at = _extract_into_tensor(self.alphas_cumprod, self._scale_timesteps(t), x.shape)
            at_next = _extract_into_tensor(self.alphas_cumprod_prev, self._scale_timesteps(t), x.shape)
            at = at.flatten()[0]
            at_next = at_next.flatten()[0]
            sigma_t = (1-at_next**2).sqrt()
            
            et = out['et']

            gamma_t = 1.

            if et.size(1) == x.shape[1]*2:
                et = et[:, :et.size(1) // 2]
            else:
                pass
            x0_t_hat = out['pred_xstart']
            
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0
                
            c1 = (1 - at_next).sqrt() * eta
            c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)

            xt_next = at_next.sqrt() * x0_t_hat + gamma_t * nonzero_mask * (c1 * torch.randn_like(x0_t_hat) + c2 * et)
            
            return {"sample": xt_next, "pred_xstart": out["pred_xstart"]}

        else:
            raise ValueError("Maybe not intended for this")
            

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        param_dicts,
        x_gt = None,
        measurement = None,
        measurement_cond_fn = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        # eta = 1.0
        
        progress = True
        
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model = model,
            shape = shape,
            x_gt = x_gt,
            measurement = measurement,
            measurement_cond_fn = measurement_cond_fn,
            param_dicts = param_dicts,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        param_dicts,
        x_gt = None,
        measurement = None,
        measurement_cond_fn = None,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        # print(f"gd.py eta: {eta}")
        # eta = 1.
        # print(f"gd.py eta 2: {eta}")
        
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            # print(f"gd.py t: {t} \ eta: {eta}")
            with th.no_grad():
                out = self.ddim_sample(
                    model = model,
                    x = img,
                    t = t,
                    measurement = measurement,
                    measurement_cond_fn = measurement_cond_fn,
                    param_dicts = param_dicts,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
                
                yield out
                img = out["sample"]
                
    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}


    def training_losses(self, model, clean_micro, t, param_dicts, model_kwargs=None, noise=None):
        """
        * NOTE: This function is intended for the default training scheme of guided_diffusion, not for the MCM training scheme.
        
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(clean_micro)
        x_t = self.q_sample(clean_micro, t, noise=noise)
        
        dataset_name = param_dicts['dataset_name']

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            raise ValueError("May be not here")
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=clean_micro,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)
            
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:  
                raise ValueError("May be not here")
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=clean_micro,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0
            
            # -----------------
            # Define the appropreiate weighting constant depending on the model_mean_type (Reference: arXiv preprint arXiv:2411.18702)
            # -----------------
            if self.model_mean_type in [ModelMeanType.START_X]:
                pred_xstart = model_output
                model_mean_type_title = "predX"
                weight_constant = 1.

            elif self.model_mean_type in [ModelMeanType.EPSILON]:
                pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
                model_mean_type_title = "predEps"
                weight_constant = (_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, clean_micro.shape) / _extract_into_tensor(self.sqrt_alphas_cumprod, t, clean_micro.shape))
                weight_constant = 1/torch.square(weight_constant)
            else:
                raise ValueError(f"Check the diffusion_predtype: {diffusion_predtype}")        
            

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=clean_micro, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: clean_micro,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == clean_micro.shape
            # terms["mse"] = mean_flat((target - model_output) ** 2)
            terms["mse"] = mean_flat(weight_constant*((clean_micro-pred_xstart) ** 2))
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
                
        else:
            raise NotImplementedError(self.loss_type)
        
        if t.shape[0] != 1:
            indexed_t = t[0]
        else:
            indexed_t = t

        if ((indexed_t < 100) and (indexed_t % 6 == 0) and (indexed_t != 0)) or ((indexed_t >= 100) and (indexed_t % 300 == 0)):
            figures_name_list = ["Ground Truth", f"Noisy {t}", "Prediction x start"]
            if self.model_mean_type in [ModelMeanType.START_X]:
                pred_xstart_for_plot = model_output.clone()
                model_mean_type_title = "predX"
            elif self.model_mean_type in [ModelMeanType.EPSILON]:
                pred_xstart_for_plot = self._predict_xstart_from_eps(x_t, t, model_output).clone()
                model_mean_type_title = "predEps"
            figures_list = [clean_micro[0].unsqueeze(0), x_t[0].unsqueeze(0), pred_xstart_for_plot[0].unsqueeze(0)]
            
            plot_multiples_in_one(figures_list, figures_name_list, dataset_name = dataset_name, png_file_name = f"{dataset_name}_uncond_{model_mean_type_title}")

        return terms

    def training_losses_noiseless_msm(self, model, clean_micro, t, param_dicts, model_kwargs=None, noise=None):
        """
        * NOTE: This function is for MCM training with noiseless, subsampled data.

        Compute training losses for a single timestep.

        :param model: the model to train.
        :param clean_micro: the ground truth images (or measurements) which are not used for training.
        :param t: a batch of timestep indices.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # ------------------
        # Define the parameters for the training
        # ------------------
        dataset_name = param_dicts['dataset_name']
        training_data_noiselevel = param_dicts['training_data_noiselevel']
        acceleration_rate = param_dicts['acceleration_rate']
        mask_pattern = param_dicts['mask_pattern']
        diffusion_model_type = param_dicts['diffusion_model_type']
        diffusion_predtype = param_dicts['diffusion_predtype']
        cond_channels = param_dicts['cond_channels']
        t_and_alphas_at_training_data_noiselevel = param_dicts['t_and_alphas_at_training_data_noiselevel']
        latent_type =  param_dicts['latent_type'] # TODO: It can be image, measurement space
        assert training_data_noiselevel == 0
        assert (latent_type in ['image_space', 'measurement_space']) and (diffusion_model_type == 'measurement_diffusion' and cond_channels == 0)


        # ------------------
        # * Create measurements based on the training scenario by applying dataset-specific subsampling masks and optional noise.
        # ------------------
        if dataset_name == "fastmri":
            b, c, h, w = clean_micro.shape
            mask, mask_np = get_fastmri_mask(batch_size = b, image_w_h = h, mask_pattern = mask_pattern, acceleration_rate = acceleration_rate, dataset_name = "fastmri")
            mask = mask.to(clean_micro.device)
            model_kwargs['low_res'] = mask

            measurement_micro = fmult(clean_micro, model_kwargs['smps'], model_kwargs['low_res'])

            measurement_micro = measurement_micro.to(clean_micro.device)

        elif dataset_name in ['ffhq']:
            b, c, h, w = clean_micro.shape
            assert acceleration_rate <= 1
            mask = get_ffhq_mask(batch_size = b, image_w_h = h, mask_pattern = mask_pattern, acceleration_rate = acceleration_rate, dataset_name = "ffhq")
            mask = mask.to(clean_micro.device)
            model_kwargs['low_res'] = mask
            model_kwargs['smps'] = mask

            measurement_micro = clean_micro * model_kwargs['low_res']
            measurement_micro = measurement_micro.to(clean_micro.device)
        
        else:
            raise ValueError(f"Check the dataset_name: {dataset_name}")
        
        # ------------------
        # Scale the timestep to a 1000-step scale. (t = 10 -> scaled_model_input_t = 1000 if `timestep_respacing' is specified as 'ddim10')
        # This is useful for training with only the necessary sparse steps, as required by DDIM sampling, without having to train across all 1000 steps.
        # ------------------
        scaled_model_input_t = self.scaling_timestep_to_1000scale(t)

        # ------------------
        # Get the diffusion noise level corresponding to the timestep
        # ------------------
        diffusion_noise_level_at_t = (_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, measurement_micro.shape) / _extract_into_tensor(self.sqrt_alphas_cumprod, t, measurement_micro.shape)).flatten()[0]

        # ------------------
        # Mask out the existing noise on the zero-filled part of the measurement
        # ------------------
        measurement_micro = apply_mask_on_kspace_wthout_ftranfmult(kspace = measurement_micro, smps = model_kwargs["smps"], mask = model_kwargs["low_res"])
        measurement_micro_image = ftran(measurement_micro, smps = model_kwargs['smps'], mask = model_kwargs['low_res'])
        
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            if latent_type == 'image_space':
                raise ValueError("Maybe not intended for us.")
                noise = th.randn_like(clean_micro)
                measurement_micro_image = ftran_non_mask(measurement_micro, smps = model_kwargs['smps'])
                x_t = self.q_sample(measurement_micro_image, t, noise=noise, training_data_noiselevel = training_data_noiselevel)

            elif latent_type == 'measurement_space':
                noise = th.randn_like(measurement_micro)

                # ------------------
                # Directly add noise on the subsampled measurement
                # ------------------
                x_t = self.q_sample(measurement_micro, t, noise=noise)
                        
            else:
                raise ValueError(f"Not intendned latent_type: {latent_type}")
        else:
            raise ValueError("Not intendned")
        
        terms = {}


        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # -----------------
            # Performs domain transformation from measurement space to image space (e.g., kspace → image space) and masks noise in zero-filled measurement regions.
            # -----------------
            if dataset_name == "fastmri":
                x_t = ftran(x_t, smps = model_kwargs['smps'], mask = model_kwargs['low_res'])

            elif dataset_name == "ffhq":
                x_t = ftran(x_t, smps = model_kwargs['smps'], mask = model_kwargs['low_res'])

            else:
                raise ValueError(f"Not implemented dataset_name: {dataset_name}")
            # -----------------
            # Get model prediction
            # -----------------
            model_output = model(x_t, scaled_model_input_t, **model_kwargs)

            # -----------------
            # Define the appropreiate weighting constant depending on the model_mean_type (Reference: arXiv preprint arXiv:2411.18702)
            # -----------------
            if self.model_mean_type in [ModelMeanType.START_X]:
                pred_xstart = model_output
                model_mean_type_title = "predX"
                weight_constant = 1.
            elif self.model_mean_type in [ModelMeanType.EPSILON]:
                pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
                model_mean_type_title = "predEps"
                weight_constant = (_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, measurement_micro.shape) / _extract_into_tensor(self.sqrt_alphas_cumprod, t, measurement_micro.shape))
                weight_constant = 1/torch.square(weight_constant)
            else:
                raise ValueError(f"Check the diffusion_predtype: {diffusion_predtype}")
            
            # -----------------
            # Performs domain transformation from image space to measurement space (e.g., image space → kspace space) and masks noise in zero-filled measurement regions.
            # -----------------
            degraded_pred_xstart = fmult(pred_xstart, smps = model_kwargs["smps"], mask = model_kwargs["low_res"])
            
            # -----------------
            # Compute the loss between subsampled measurement.
            # -----------------
            terms["mse"] = mean_flat(weight_constant*((measurement_micro-degraded_pred_xstart) ** 2))

            terms["loss"] = terms["mse"]
                
        else:
            raise NotImplementedError(self.loss_type)
        
        # -----------------
        # Plot the intermediate results
        # -----------------
        if t.shape[0] != 1:
            indexed_t = t[0]
        else:
            indexed_t = t
        if (indexed_t % 50 == 0):
            pred_xstart_for_plot = pred_xstart.clone()
            degraded_pred_xstart_for_plot = ftran_non_mask(degraded_pred_xstart.clone(), smps = model_kwargs['smps'])
            y_hat_for_plot = measurement_micro_image.clone()
            
            if dataset_name == "ffhq":
                mask_for_plot = model_kwargs['low_res'].clone()
                mask_for_plot = mask_for_plot.expand(-1, 3, -1, -1)
            elif dataset_name == "fastmri":
                mask_for_plot = model_kwargs['low_res']
            else:
                raise ValueError(f"Not implemented dataset_name: {dataset_name}")
            
            figures_name_list = ["Ground Truth", f"Noisy {t}", "Degraded Pred x start", "Prediction x start", "Measurement", "mask"]
            figures_list = [clean_micro[0].unsqueeze(0), x_t[0].unsqueeze(0), degraded_pred_xstart_for_plot[0].unsqueeze(0), pred_xstart_for_plot[0].unsqueeze(0), y_hat_for_plot[0].unsqueeze(0), mask_for_plot[0].unsqueeze(0)]
            
            png_file_name = f"{diffusion_model_type}_{dataset_name}_traindatanoise_{training_data_noiselevel}_{acceleration_rate}_vbF_{model_mean_type_title}"
            plot_multiples_in_one(figures_list, figures_name_list, dataset_name = dataset_name, png_file_name = png_file_name)
            
        return terms
    
    def training_losses_noisy_msm(self, model, clean_micro, t, param_dicts, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to train.
        :param clean_micro: the ground truth images (or measurements) which are not used for training.
        :param t: a batch of timestep indices.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        # ------------------
        # Define the parameters for the training
        # ------------------
        dataset_name = param_dicts['dataset_name']
        training_data_noiselevel = param_dicts['training_data_noiselevel']
        acceleration_rate = param_dicts['acceleration_rate']
        mask_pattern = param_dicts['mask_pattern']
        diffusion_model_type = param_dicts['diffusion_model_type']
        diffusion_predtype = param_dicts['diffusion_predtype']
        cond_channels = param_dicts['cond_channels']
        t_and_alphas_at_training_data_noiselevel = param_dicts['t_and_alphas_at_training_data_noiselevel']
        tau_SURE = param_dicts['tau_SURE']
        lr_SURE = param_dicts['lr_SURE']
        latent_type =  param_dicts['latent_type'] # TODO: It can be image, measurement space
        assert training_data_noiselevel != 0
        assert (latent_type in ['image_space', 'measurement_space']) and (diffusion_model_type == 'measurement_diffusion' and cond_channels == 0)

        # ------------------
        # Create measurements based on the training scenario by applying dataset-specific subsampling masks and noise.
        # ------------------
        if dataset_name == "fastmri":
            b, c, h, w = clean_micro.shape
            mask, mask_np = get_fastmri_mask(batch_size = b, image_w_h = h, mask_pattern = mask_pattern, acceleration_rate = acceleration_rate, dataset_name = "fastmri")
            mask = mask.to(clean_micro.device)
            model_kwargs['low_res'] = mask

            # -----------------
            # Performs domain transformation from image space to measurement space (e.g., image space → kspace space) and masks noise in zero-filled measurement regions.
            # -----------------
            measurement_micro = fmult(clean_micro, model_kwargs['smps'], model_kwargs['low_res'])
            measurement_noise = torch.randn_like(measurement_micro)
            # -----------------
            # Adding measurement noise
            # -----------------
            measurement_micro = measurement_micro + training_data_noiselevel*measurement_noise
            measurement_micro = measurement_micro.to(clean_micro.device)

        elif dataset_name in ['ffhq']:
            b, c, h, w = clean_micro.shape
            assert acceleration_rate <= 1
            mask = get_ffhq_mask(batch_size = b, image_w_h = h, mask_pattern = mask_pattern, acceleration_rate = acceleration_rate, dataset_name = "ffhq")
            mask = mask.to(clean_micro.device)
            model_kwargs['low_res'] = mask
            model_kwargs['smps'] = mask

            measurement_noise = torch.randn_like(clean_micro)
            measurement_micro = clean_micro * model_kwargs['low_res']
            # -----------------
            # Adding measurement noise
            # -----------------
            measurement_micro = measurement_micro + training_data_noiselevel*measurement_noise

            measurement_micro = measurement_micro.to(clean_micro.device)
        
        else:
            raise ValueError(f"Check the dataset_name: {dataset_name}")

        # ------------------
        # Scale the timestep to 1000 scale (this is useful for the training with sparse steps intended for the ddim sampling once trained)
        # ------------------
        scaled_model_input_t = self.scaling_timestep_to_1000scale(t)

        # ------------------
        # Get the diffusion noise level corresponding to the timestep
        # ------------------
        diffusion_noise_level_at_t = (_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, measurement_micro.shape) / _extract_into_tensor(self.sqrt_alphas_cumprod, t, measurement_micro.shape)).flatten()[0]

        # ------------------
        # Mask out the existing noise on the zero-filled part of the measurement
        # ------------------
        measurement_micro = apply_mask_on_kspace_wthout_ftranfmult(kspace = measurement_micro, smps = model_kwargs["smps"], mask = model_kwargs["low_res"])
        measurement_micro_image = ftran(measurement_micro, smps = model_kwargs['smps'], mask = model_kwargs['low_res'])
        
        if model_kwargs is None:
            model_kwargs = {}
        
        # ---------------
        # Define necessary parameter for SURE loss computation (We adopt the implementation from: https://github.com/edongdongchen/REI)
        # ---------------
        n = measurement_micro.shape[1] * measurement_micro.shape[2] * measurement_micro.shape[3]# * y_hat.shape[4]
        batch_size = measurement_micro.shape[0]
        if acceleration_rate == 0:
            m = n / 1
        else:
            m = n / acceleration_rate

        terms = {}


        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            assert tau_SURE != 0
            # ------------------
            # Denoise the noisy measurement using the same MCM, conditioning on the timestep corresponding to the measurement noise level.
            # ------------------
            t_at_training_data_noiselevel, alphas_at_training_data_noiselevel = t_and_alphas_at_training_data_noiselevel
            t_at_training_data_noiselevel = torch.tensor([t_at_training_data_noiselevel] * measurement_micro.shape[0]).to(measurement_micro.device)
            y1_model_output = model(measurement_micro_image, self._scale_timesteps(t_at_training_data_noiselevel), diffusion_model_type = diffusion_model_type, **model_kwargs)
            if self.model_mean_type in [ModelMeanType.START_X]:
                pred_clean_y1 = y1_model_output
            elif self.model_mean_type in [ModelMeanType.EPSILON]:
                pred_clean_y1 = (math.sqrt(1/alphas_at_training_data_noiselevel))*measurement_micro_image - (math.sqrt(1./alphas_at_training_data_noiselevel - 1))*y1_model_output
            else:
                raise ValueError(f"Check the diffusion_predtype: {diffusion_predtype}")
                
            # ------------------
            # SURE regularization (See details from https://github.com/edongdongchen/REI)
            # ------------------
            sure_noise = th.randn_like(measurement_micro)
            y_hat_plus_noise = math.sqrt(alphas_at_training_data_noiselevel)*(measurement_micro + (tau_SURE/math.sqrt(alphas_at_training_data_noiselevel)) * sure_noise)
            y_hat_plus_noise_image = ftran(y_hat_plus_noise, smps = model_kwargs["smps"], mask = model_kwargs["low_res"])
            y2_model_output = model(y_hat_plus_noise_image, self._scale_timesteps(t_at_training_data_noiselevel), degrade_network_input = True, diffusion_model_type = diffusion_model_type, **model_kwargs)
            if diffusion_predtype == "pred_xstart":
                pred_clean_y2 = y2_model_output
            elif diffusion_predtype == "epsilon":
                pred_clean_y2 = (math.sqrt(1/alphas_at_training_data_noiselevel))*y_hat_plus_noise_image - (math.sqrt(1./alphas_at_training_data_noiselevel - 1))*y2_model_output
            else:
                raise ValueError(f"Check the diffusion_predtype: {diffusion_predtype}")

            y1 = fmult(pred_clean_y1, smps = model_kwargs["smps"], mask = model_kwargs["low_res"])
            y2 = fmult(pred_clean_y2, smps = model_kwargs["smps"], mask = model_kwargs["low_res"])

            eta_square = training_data_noiselevel ** 2

            # ------------------
            # Compute SURE loss
            # ------------------
            loss_sure = torch.sum((y1 - measurement_micro).pow(2)) / (batch_size * m) - eta_square + (2 * eta_square / (tau_SURE * m * batch_size)) * (sure_noise * (y2 - y1)).sum()

            # ------------------
            # Adding diffusion noise depending on the measurement noise level and diffusion noise level.
            # ------------------
            noise = th.randn_like(measurement_micro)
            if diffusion_noise_level_at_t.flatten()[0] > training_data_noiselevel:
                # * Case 1 (diffusion noise > measurement noise) in the paper: Add residual noise to the noisy measurement.
                x_t = self.q_sample(measurement_micro, t, noise=noise, training_data_noiselevel = training_data_noiselevel)
            else:
                # * Case 2 (diffusion noise <= measurement noise) in the paper: Add noise to the denoised measurement.
                x_t = self.q_sample(y1, t, noise=noise)

            # -----------------
            # Performs domain transformation from measurement space to image space (e.g., kspace → image space) and masks noise in zero-filled measurement regions.
            # -----------------
            if dataset_name == "fastmri":
                x_t = ftran(x_t, smps = model_kwargs['smps'], mask = model_kwargs['low_res'])
            elif dataset_name == "ffhq":
                x_t = ftran(x_t, smps = model_kwargs['smps'], mask = model_kwargs['low_res'])

            else:
                raise ValueError(f"Not implemented dataset_name: {dataset_name}")

            # -----------------
            # Get model prediction
            # -----------------
            model_output = model(x_t, scaled_model_input_t, **model_kwargs)

            if self.model_mean_type in [ModelMeanType.START_X]:
                pred_xstart = model_output
                model_mean_type_title = "predX"
                weight_constant = 1.
            elif self.model_mean_type in [ModelMeanType.EPSILON]:
                pred_xstart = self._predict_xstart_from_eps(x_t, t, model_output)
                model_mean_type_title = "predEps"
                weight_constant = (_extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, measurement_micro.shape) / _extract_into_tensor(self.sqrt_alphas_cumprod, t, measurement_micro.shape))
                weight_constant = 1/torch.square(weight_constant)
            else:
                raise ValueError(f"Check the diffusion_predtype: {diffusion_predtype}")

            # ---------------
            # For the noisy measurement scenario, the batch should have the same timesteps. To ensure that, the following conditions are added.
            # ---------------
            if len(measurement_micro.shape) == 5:
                assert weight_constant[0, 0, 0, 0, 0] == weight_constant[0, 1, 1, 1, 1]
                if weight_constant.shape[0] > 1:
                    assert weight_constant[0, 0, 0, 0, 0] == weight_constant[1, 0, 0, 0, 0]
            elif len(measurement_micro.shape) == 4:
                assert weight_constant[0, 0, 0, 0] == weight_constant[0, 1, 1, 1]
                if weight_constant.shape[0] > 1:
                    assert weight_constant[0, 0, 0, 0] == weight_constant[1, 0, 0, 0]
            else:
                raise ValueError(f"Check the shape of weight_constant: {weight_constant.shape}")
                    
            weight_constant = weight_constant.flatten()[0]


            if diffusion_noise_level_at_t.flatten()[0] > training_data_noiselevel:
                # * Case 1 in the paper: (diffusion noise > measurement noise)
                # -----------------
                # Using Tweedie’s formula, we estimate the clean measurement from the prediction of the noisy measurement.
                # -----------------
                pred_xt_via_pred_xstart_image = ((torch.square(diffusion_noise_level_at_t) - (training_data_noiselevel*training_data_noiselevel))/(torch.square(diffusion_noise_level_at_t)))*(pred_xstart - x_t) + x_t
                degraded_pred_xt_via_pred_xstart_kspace = fmult(pred_xt_via_pred_xstart_image, smps = model_kwargs["smps"], mask = model_kwargs["low_res"])

                terms["mse"] = mean_flat(weight_constant*((measurement_micro-degraded_pred_xt_via_pred_xstart_kspace) ** 2)) + loss_sure * weight_constant * lr_SURE
            else:
                # * Case 2 in the paper: (diffusion noise <= measurement noise)
                # -----------------
                # Performs domain transformation from image space to measurement space (e.g., image space → kspace space) and masks noise in zero-filled measurement regions.
                # -----------------
                degraded_pred_xstart = fmult(pred_xstart, smps = model_kwargs["smps"], mask = model_kwargs["low_res"])

                terms["mse"] = mean_flat(weight_constant*((y1-degraded_pred_xstart) ** 2)) + loss_sure * weight_constant * lr_SURE

            terms["loss"] = terms["mse"]
                
        else:
            raise NotImplementedError(self.loss_type)
        
        # -----------------
        # Plot the intermediate results
        # -----------------
        if t.shape[0] != 1:
            indexed_t = t[0]
        else:
            indexed_t = t
        if (indexed_t % 50 == 0):
            pred_xstart_for_plot = pred_xstart.clone()
            degraded_pred_xstart_for_plot = ftran_non_mask(degraded_pred_xstart.clone(), smps = model_kwargs['smps'])
            y_hat_for_plot = measurement_micro_image.clone()
            
            if dataset_name == "ffhq":
                mask_for_plot = model_kwargs['low_res'].clone()
                mask_for_plot = mask_for_plot.expand(-1, 3, -1, -1)
            elif dataset_name == "fastmri":
                mask_for_plot = model_kwargs['low_res']
            else:
                raise ValueError(f"Not implemented dataset_name: {dataset_name}")
            
            pred_clean_y1_for_plot = pred_clean_y1.clone()
            figures_name_list = ["Ground Truth", f"Noisy {t}", "Degraded Pred x start", "Prediction x start", "Measurement", "Denoised yhat", "mask"]
            figures_list = [clean_micro[0].unsqueeze(0), x_t[0].unsqueeze(0), degraded_pred_xstart_for_plot[0].unsqueeze(0), pred_xstart_for_plot[0].unsqueeze(0), y_hat_for_plot[0].unsqueeze(0), pred_clean_y1_for_plot[0].unsqueeze(0), mask_for_plot[0].unsqueeze(0)]
            
            png_file_name = f"{diffusion_model_type}_{dataset_name}_traindatanoise_{training_data_noiselevel}_tauSURE_{tau_SURE}_lrSURE_{lr_SURE}_{acceleration_rate}_vbF_{model_mean_type_title}"
            plot_multiples_in_one(figures_list, figures_name_list, dataset_name = dataset_name, png_file_name = png_file_name)
            
        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }
        
    def calc_weight_mask(self, mask):
        # -----------------
        # Natural images case
        # -----------------
        if mask.shape[1] == 3:
            # print(f"returned")
            return torch.ones_like(mask)
        # -----------------
        # Single coil medical image case which this paper not focus on
        # -----------------
        else:
            __EPS = 0.0001
            num_center_lines = 20
            num_total_lines = mask.shape[-1]
            num_masked_off_center_lines = mask[0, 0, 0].sum().cpu().item() - num_center_lines
            # print(f"num_total_lines: {num_total_lines}")
            num_off_center_lines = num_total_lines - num_center_lines

            weights = (mask > __EPS).int() * (num_off_center_lines / num_masked_off_center_lines)

            center_line_idx = torch.arange((num_total_lines - num_center_lines) // 2,
                                        (num_total_lines + num_center_lines) // 2)
            
            weights[:, :, :, center_line_idx] = 1

            return weights
    

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def renorm_from_minusonetoone_to_zeroone(x):
    return (x + 1.) / 2.
