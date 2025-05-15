# -----------------
# Importing from Python module
# -----------------
from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
# -----------------
# Importing from files
# -----------------
from datasets.fastMRI import fmult, ftran, fmult_non_mask, apply_mask_on_kspace_wthout_ftranfmult, get_fastmri_mask
from datasets.ffhq import get_ffhq_mask
from utility.dds_utility import MulticoilMRI, NaturalImages


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps, y=None, **kwargs):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # print(f"unet.py timesteps: {timesteps}")
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class SuperResModel(UNetModel):
    """
    A UNetModel that performs super-resolution.

    Expects an extra kwarg `low_res` to condition on a low-resolution image.
    """

    def __init__(self, image_size, in_channels, cond_channels, diffusion_model_type, for_training, *args, **kwargs):
        super().__init__(image_size, in_channels + cond_channels, *args, **kwargs)
        
        self.cond_channels = cond_channels
        self.in_channels = in_channels
        self.diffusion_model_type = diffusion_model_type
        self.for_training = for_training

    def forward(self, x, timesteps, param_dicts = None, low_res=None, smps = None, mask_further_degradation = None, **kwargs):
        if self.for_training == True:
            return super().forward(x, timesteps, **kwargs)

        elif self.for_training == False:
            assert (param_dicts['solve_inverse_problem'] == True and param_dicts['measurement_for_inverse_problem'] != None and param_dicts['dataset_name'] in ['ffhq', 'imagenet']) or (param_dicts['solve_inverse_problem'] == True and param_dicts['measurement_for_inverse_problem'] != None and param_dicts['mask_for_inverse_problem'] != None and param_dicts['dataset_name'] == 'fastmri') or param_dicts['solve_inverse_problem'] == False, "Check the solve_inverse_problem and measurement"
            diffusion_predtype = param_dicts['diffusion_predtype']; dataset_name = param_dicts['dataset_name']

            if self.diffusion_model_type in ["unconditional_diffusion"]:
                if diffusion_predtype == "epsilon":
                    eps = super().forward(x, timesteps, **kwargs)
                    return eps
                elif diffusion_predtype == "pred_xstart":
                    pred_xstart = super().forward(x, timesteps, **kwargs)
                    return pred_xstart
                else:
                    raise ValueError(f"Check the diffusion_predtype {diffusion_predtype}")


            elif self.diffusion_model_type == "measurement_diffusion":

                assert param_dicts['cumulative_mask'] != None
                
                # -----------------
                # Get the necessary parameters from param_dicts
                # -----------------
                mask_pattern = param_dicts['mask_pattern']; acceleration_rate = param_dicts['acceleration_rate']; alphas_cumprod_t = param_dicts['alphas_cumprod_t']; alphas_cumprod_t_plus_1 = param_dicts['alphas_cumprod_t_plus_1']; stochastic_loop = param_dicts['stochastic_loop']; sqrt_recip_alphas_cumprod = param_dicts['sqrt_recip_alphas_cumprod']; sqrt_recipm1_alphas_cumprod = param_dicts['sqrt_recipm1_alphas_cumprod']
                
                if dataset_name == "fastmri":
                    b, c, h, w, s = x.shape
                elif dataset_name == "ffhq":
                    b, c, h, w = x.shape
                else:
                    raise ValueError(f"Check the dataset_name {dataset_name}")

                # -----------------
                # Define base variables needed to aggreate denoised results and mask
                # * Note: cumulative mask is used to generate stochastic mask which is sufficiently distinct from the previous one.
                # -----------------
                if param_dicts['previous_pred_ystart'] == None:
                    param_dicts['previous_pred_ystart'] = torch.zeros_like(x)
                cumulative_pred_ystart = param_dicts['previous_pred_ystart']
                if (param_dicts['cumulative_mask'] == 0).sum() == 0:
                    param_dicts['cumulative_mask'] = torch.zeros_like(param_dicts['cumulative_mask'], device=x.device)
                else:
                    pass
                inLoop_cumulative_pred_ystart = torch.zeros_like(param_dicts['previous_pred_ystart'], device=x.device)
                inLoop_cumulative_mask = torch.zeros_like(param_dicts['cumulative_mask'], device=x.device)
                    
                for i in range(stochastic_loop):

                    assert param_dicts['cumulative_mask'] != None, "Check the cumulative_mask"
                    # -----------------
                    # Generate subsampling mask
                    # -----------------
                    if dataset_name == "fastmri":
                        stochastic_mask, _ = get_fastmri_mask(batch_size = b, image_w_h = h, mask_pattern = "randomly_cartesian", acceleration_rate = acceleration_rate, dataset_name = "fastmri", cumulative_mask = param_dicts['cumulative_mask'])
                        stochastic_mask = stochastic_mask.to(x.device)

                    elif dataset_name == "ffhq":
                        cumulative_small_mask_information = [param_dicts['cumulative_small_mask'], param_dicts['cumulative_small_mask_shift_x'], param_dicts['cumulative_small_mask_shift_y']]
                        stochastic_mask, previous_cumulative_small_mask_information = get_ffhq_mask(batch_size = b, image_w_h = h, mask_pattern = mask_pattern, dataset_name = 'ffhq', acceleration_rate = acceleration_rate, cumulative_small_mask_information = cumulative_small_mask_information, timesteps = int(timesteps.item()))
                        smps = stochastic_mask # since natural images do not have sensitivity map, we just put the mask as the sensitivity map

                        # * Note: Mask for the natural image contains shifting coordinates to move the mask. Below, we save the shifting coordinates to define the next stochastic mask.
                        param_dicts['cumulative_small_mask'] = previous_cumulative_small_mask_information[0]
                        param_dicts['cumulative_small_mask_shift_x'] = previous_cumulative_small_mask_information[1]
                        param_dicts['cumulative_small_mask_shift_y'] = previous_cumulative_small_mask_information[2]

                        stochastic_mask = stochastic_mask.to(x.device)

                    else:
                        raise ValueError(f"Check the dataset_name {dataset_name}")
                    assert param_dicts['cumulative_mask'].shape == stochastic_mask.shape
                    
                    param_dicts['cumulative_mask'] += stochastic_mask
                    
                    # * Note: When all pixels are denoised at once, the cumulative mask must be reset to zero to sample a completely new mask.
                    num_zero_in_cumulative_mask = (param_dicts['cumulative_mask'] == 0).sum(dim=3).flatten()[0]
                    if num_zero_in_cumulative_mask == 0:
                        param_dicts['cumulative_mask'] = torch.zeros_like(param_dicts['cumulative_mask'], device=x.device)

                    # -----------------
                    # Performs domain transformation from measurement space to image space (e.g., kspace → image space) and masks noise in zero-filled measurement regions.
                    # -----------------
                    degraded_x = ftran(x, smps = smps, mask = stochastic_mask)

                    # -----------------
                    # Get model prediction
                    # -----------------
                    eps = super().forward(degraded_x, timesteps, **kwargs)
                    pred_xstart = sqrt_recip_alphas_cumprod.flatten()[0] * degraded_x - sqrt_recipm1_alphas_cumprod.flatten()[0] * eps
                    # -----------------
                    # Performs domain transformation from image space to measurement space (e.g., image space → kspace space) and masks noise in zero-filled measurement regions.
                    # -----------------
                    degraded_pred_xstart = fmult(pred_xstart, smps = smps, mask = stochastic_mask)
                    
                    if param_dicts['solve_inverse_problem'] == True and param_dicts['inverse_problem_type'] in ['fastmri_reconstruction']:
                        # -----------------
                        # Define forward operator of input measurement
                        # -----------------
                        if dataset_name in ['ffhq', 'imagenet']:
                            A_funcs = NaturalImages(operator = param_dicts['operator'])
                        elif dataset_name in ['fastmri']:
                            mps = (smps.squeeze(1))#.contiguous()
                            mps = mps.permute(0, 3, 1, 2).contiguous()
                            if dataset_name in ['ffhq', 'imagenet']:
                                A_funcs = NaturalImages(operator = param_dicts['operator'])
                            elif dataset_name == 'fastmri':
                                A_funcs = MulticoilMRI(h, param_dicts['mask_for_inverse_problem'], mps)
                        else:
                            raise ValueError(f"Check the dataset_name: {dataset_name}")
                        A = lambda z: A_funcs.A(z)
                        AT = lambda z: A_funcs.AT(z)
                        Ap = lambda z: A_funcs.A_dagger(z)
                        measurement_for_inverse_problem = param_dicts['measurement_for_inverse_problem']

                        # -----------------
                        # Compute the gradient of the likelihood function
                        # -----------------
                        lambda_t = param_dicts['stepsize_likelihood']
                        if dataset_name in ['ffhq', 'imagenet']:
                            P_yhat = A(degraded_pred_xstart)
                            y_minus_yhat = (P_yhat - measurement_for_inverse_problem)
                            P_y_minus_P_yhat = Ap(y_minus_yhat)
                        elif dataset_name in ['fastmri']:
                            P_yhat = (apply_mask_on_kspace_wthout_ftranfmult(degraded_pred_xstart, smps = smps, mask = param_dicts['mask_for_inverse_problem']))
                            measurement_for_inverse_problem = (apply_mask_on_kspace_wthout_ftranfmult(measurement_for_inverse_problem, smps = smps, mask = param_dicts['mask_for_inverse_problem']))
                            y_minus_yhat = (P_yhat - measurement_for_inverse_problem)
                            P_y_minus_P_yhat = (apply_mask_on_kspace_wthout_ftranfmult(y_minus_yhat, smps = smps, mask = param_dicts['mask_for_inverse_problem']))
                        degraded_pred_xstart = degraded_pred_xstart - lambda_t * P_y_minus_P_yhat
                        degraded_pred_xstart = apply_mask_on_kspace_wthout_ftranfmult(degraded_pred_xstart, smps = smps, mask = stochastic_mask)

                    # -----------------
                    # Aggregate the denoised results and mask
                    # -----------------
                    inLoop_cumulative_pred_ystart += degraded_pred_xstart
                    inLoop_cumulative_mask += stochastic_mask
                    
                    # -----------------
                    # Inject noise to the denoised measurement and go to next iteration
                    # -----------------
                    alpha_bar = alphas_cumprod_t_plus_1
                    alpha_bar_prev = alphas_cumprod_t
                    assert (torch.sqrt((1-alpha_bar_prev)/alpha_bar_prev)).flatten()[0] < torch.sqrt(((1-alpha_bar)/alpha_bar)).flatten()[0]
                    eta = param_dicts['ddim_eta']
                    noise_in_stochastic_loop = torch.randn_like(degraded_pred_xstart)
                    sigma = (
                        eta
                        * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                        * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
                    )
                    noise = noise_in_stochastic_loop
                    if dataset_name == "fastmri":
                        mean_pred = (
                            degraded_pred_xstart * th.sqrt(alpha_bar_prev)
                            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * fmult_non_mask(eps, smps = smps)
                        )
                    elif dataset_name in ["ffhq", "imagenet"]:
                        mean_pred = (
                            degraded_pred_xstart * th.sqrt(alpha_bar_prev)
                            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
                        )
                    nonzero_mask = (
                        (timesteps != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                    )
                    sample = mean_pred + nonzero_mask * sigma * noise
                    x[degraded_pred_xstart != 0] = sample[degraded_pred_xstart != 0]

                # -----------------
                # Compute the weight vector W
                # -----------------
                inLoop_cumulative_mask[inLoop_cumulative_mask == 0] = 1.
                weight_vector = 1./inLoop_cumulative_mask
                
                # -----------------
                # Apply the weight vector W to the combined partial denoised result
                # -----------------
                inLoop_cumulative_pred_ystart = (apply_mask_on_kspace_wthout_ftranfmult(inLoop_cumulative_pred_ystart, smps = smps, mask = weight_vector))#.detach()
                cumulative_pred_ystart[inLoop_cumulative_pred_ystart != 0] = inLoop_cumulative_pred_ystart[inLoop_cumulative_pred_ystart != 0]
                
                if param_dicts['solve_inverse_problem'] == True and param_dicts['inverse_problem_type'] not in ['fastmri_reconstruction']:
                    # -----------------
                    # Define forward operator of input measurement
                    # -----------------
                    if dataset_name in ['ffhq', 'imagenet']:
                        A_funcs = NaturalImages(operator = param_dicts['operator'])
                    elif dataset_name in ['fastmri']:
                        mps = (smps.squeeze(1))#.contiguous()
                        mps = mps.permute(0, 3, 1, 2).contiguous()
                        if dataset_name in ['ffhq', 'imagenet']:
                            A_funcs = NaturalImages(operator = param_dicts['operator'])
                        elif dataset_name == 'fastmri':
                            A_funcs = MulticoilMRI(h, param_dicts['mask_for_inverse_problem'], mps)
                    else:
                        raise ValueError(f"Check the dataset_name: {dataset_name}")
                    A = lambda z: A_funcs.A(z)
                    AT = lambda z: A_funcs.AT(z)
                    Ap = lambda z: A_funcs.A_dagger(z)
                    measurement_for_inverse_problem = param_dicts['measurement_for_inverse_problem']

                    # -----------------
                    # Compute the gradient of the likelihood function
                    # -----------------
                    lambda_t = param_dicts['stepsize_likelihood']
                    if dataset_name in ['ffhq', 'imagenet']:
                        P_yhat = A(cumulative_pred_ystart)
                        y_minus_yhat = (P_yhat - measurement_for_inverse_problem)
                        P_y_minus_P_yhat = Ap(y_minus_yhat)
                    elif dataset_name in ['fastmri']:
                        P_yhat = (apply_mask_on_kspace_wthout_ftranfmult(cumulative_pred_ystart, smps = smps, mask = param_dicts['mask_for_inverse_problem']))
                        measurement_for_inverse_problem = (apply_mask_on_kspace_wthout_ftranfmult(measurement_for_inverse_problem, smps = smps, mask = param_dicts['mask_for_inverse_problem']))
                        y_minus_yhat = (P_yhat - measurement_for_inverse_problem)
                        P_y_minus_P_yhat = (apply_mask_on_kspace_wthout_ftranfmult(y_minus_yhat, smps = smps, mask = param_dicts['mask_for_inverse_problem']))
                    cumulative_pred_ystart = cumulative_pred_ystart - lambda_t * P_y_minus_P_yhat
                
            
                return cumulative_pred_ystart

            else:
                raise ValueError(f"Check the diffusion_model_type {self.diffusion_model_type}")

        


class EncoderUNetModel(nn.Module):
    """
    The half UNet model with attention and timestep embedding.

    For usage, see UNet.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pool="adaptive",
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch
        self.pool = pool
        if pool == "adaptive":
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                zero_module(conv_nd(dims, ch, out_channels, 1)),
                nn.Flatten(),
            )
        elif pool == "attention":
            assert num_head_channels != -1
            self.out = nn.Sequential(
                normalization(ch),
                nn.SiLU(),
                AttentionPool2d(
                    (image_size // ds), ch, num_head_channels, out_channels
                ),
            )
        elif pool == "spatial":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                nn.ReLU(),
                nn.Linear(2048, self.out_channels),
            )
        elif pool == "spatial_v2":
            self.out = nn.Sequential(
                nn.Linear(self._feature_size, 2048),
                normalization(2048),
                nn.SiLU(),
                nn.Linear(2048, self.out_channels),
            )
        else:
            raise NotImplementedError(f"Unexpected {pool} pooling")

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)

    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        results = []
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            if self.pool.startswith("spatial"):
                results.append(h.type(x.dtype).mean(dim=(2, 3)))
        h = self.middle_block(h, emb)
        if self.pool.startswith("spatial"):
            results.append(h.type(x.dtype).mean(dim=(2, 3)))
            h = th.cat(results, axis=-1)
            return self.out(h)
        else:
            h = h.type(x.dtype)
            return self.out(h)
