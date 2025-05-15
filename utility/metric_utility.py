# -----------------
# Importing from Python module
# -----------------
import numpy as np
import torch
import os
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error, structural_similarity
from math import exp
from torch.autograd import Variable
import torch.nn.functional as f
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from skimage.metrics import structural_similarity

# -----------------
# Importing from files
# -----------------
from utility.data_utility import abs_helper, renorm_from_minusonetoone_to_zeroone



def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img

# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# ---------------------------------------------------------------



def compute_psnr_ssim_nmse(img_test, img_true):
    assert img_test.shape == img_true.shape

    psnr = compare_psnr(img_test, img_true)
    ssim_val = ssim(img_test, img_true)
    nmse = compare_nmse(img_test, img_true)
    return psnr, ssim_val, nmse


def compute_psnr_ssim_nmse_lpips(img_test, img_true, device):
    """
    for fastmri
    img_test.shape, img_true.shape: [height, width]
    
    for ffhq and imagenet
    img_test.shape, img_true.shape: [3, height, width]
    """
    assert img_test.shape == img_true.shape

    if len(img_test.shape) > 2:  # [H, W]
        if img_test.shape[2] == 3 and img_true.shape[2] == 3:
            img_test = img_test.permute(2, 0, 1).contiguous()
            img_true = img_true.permute(2, 0, 1).contiguous()
    
    psnr = compare_psnr(img_test, img_true)
    ssim_val = ssim(img_test, img_true)
    nmse = compare_nmse(img_test, img_true)

    
    # model_dir = "/SPECIFY THE LPIPS MODEL/" # model_dir = "/EXAMPLE_PATH/lpips/"
    # os.environ['TORCH_HOME'] = model_dir
    # lpips_loss_loss_fn_vgg = LearnedPerceptualImagePatchSimilarity(net_type='vgg').to(device)

    if len(img_test.shape) == 2:  # [H, W]
        img_test = img_test.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
        img_true = img_true.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
    elif img_test.shape[0] == 1:  # [1, H, W] -> assume grayscale
        img_test = img_test.repeat(3, 1, 1)  # [3, H, W]
        img_true = img_true.repeat(3, 1, 1)  # [3, H, W]

    img_test = img_test.unsqueeze(0).to(device)  # [1, 3, H, W]
    img_true = img_true.unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # lpips = lpips_loss_loss_fn_vgg(img_test, img_true).item()
    lpips = 0

    return psnr, ssim_val, nmse, lpips



def compare_nmse(img_test, img_true):
    return torch.linalg.norm(img_true - img_test) ** 2 / torch.linalg.norm(img_true) ** 2


def compare_mse(img_test, img_true, size_average=True):

    img_diff = img_test - img_true
    img_diff = img_diff ** 2

    if size_average:
        img_diff = img_diff.mean()
    else:
        img_diff = img_diff.mean(-1).mean(-1).mean(-1)

    return img_diff

def compare_psnr(img_test, img_true):

    return 10 * torch.log10((img_true.max() ** 2) / compare_mse(img_test, img_true))

def compare_psnr_mask(img_test, img_true):
    if img_test.shape != img_true.shape:
        assert img_test.shape == img_true.shape

    assert len(img_test.shape) == 3
    img_test = torch.view_as_complex(img_test.permute(1, 2, 0).contiguous()).abs()
    img_true = torch.view_as_complex(img_true.permute(1, 2, 0).contiguous()).abs()

    img_test = (img_test - img_test.min()) / (img_test.max() - img_test.min())
    img_true = (img_true - img_true.min()) / (img_true.max() - img_true.min())

    img_test_masked = img_test * (img_true != 0).int()

    mse = torch.mean((img_test_masked.squeeze() - img_true.squeeze()) ** 2)
    max_pixel_value = 1.0
    psnr = 10 * torch.log10(max_pixel_value / mse)

    return psnr


def ssim(pred, gt):
    """ Compute Structural Similarity Index Metric (SSIM). """

    if gt.shape[0] > 1:
        ssim_list = []
        for i in range(gt.shape[0]):
            ssim_list.append(ssim_im(gt[i], pred[i]))
        return np.array(ssim_list).mean()
    else:
        return ssim_im(pred, gt)


def ssim_im(gt, pred):
    """ Compute Structural Similarity Index Metric (SSIM). """
    # if isinstance(gt.max(), torch.Tensor):
    #     range = (gt.max() - gt.min()).item()
    # else:
    #     range = gt.max() - gt.min()
    if len(gt.shape) == 3:
        gt = gt[0, :]
    if len(pred.shape) == 3:
        pred = pred[0, :]
    

    return structural_similarity(gt.cpu().detach().numpy(), pred.cpu().detach().numpy(), data_range=1, win_size=7)


def compute_mse(img_test, img_true, size_average=True):
    img_diff = img_test - img_true
    img_diff = img_diff ** 2

    if size_average:
        img_diff = img_diff.mean()
    else:
        img_diff = img_diff.mean(-1).mean(-1).mean(-1)
    return img_diff

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def compute_psnr(img_test, img_true, size_average=True, max_value=1):
    
    if (img_test.shape[1] == 2) and (img_true.shape[1] == 2):
        # -----------------
        # For the case of fastmri
        # -----------------
        img_test = abs_helper(img_test)
        img_true = abs_helper(img_true)
    else:
        # -----------------
        # For the case of natural images
        # -----------------
        img_true = (img_true + 1)/2
        img_test = (img_test + 1)/2

    h, w = img_test.shape[-2], img_test.shape[-1]
    img_test = img_test[:, :, 2:h-2, 2:w-2]
    img_true = img_true[:, :, 2:h-2, 2:w-2]

    return 10 * torch.log10((max_value ** 2) / compute_mse(img_test, img_true, size_average))


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = f.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = f.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = f.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = f.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = f.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-1).mean(-1).mean(-1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data_input.type() == img1.data_input.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def compute_ssim(img_test, img_true, size_average=True, window_size=11):
    if (img_test.shape[1] == 2) and (img_true.shape[1] == 2):
        img_test = abs_helper(img_test)
        img_true = abs_helper(img_true)
    else:
        pass
        
    (_, channel, _, _) = img_test.size()
    window = create_window(window_size, channel)


    img_true = (img_true + 1)/2
    img_test = (img_test + 1)/2

    h, w = img_test.shape[-2], img_test.shape[-1]
    img_test = img_test[:, :, 4:h-4, 4:w-4]
    img_true = img_true[:, :, 4:h-4, 4:w-4]

    if img_test.is_cuda:
        window = window.cuda(img_test.get_device())
    window = window.type_as(img_test)

    return _ssim(img_test, img_true, window, window_size, channel, size_average)

def compute_metrics(reconstructed, reference, is_input_already_abs_helpered = False):
    """Compute PSNR, LPIPS, and DC distance between the reconstructed and reference images."""
    if is_input_already_abs_helpered == False:
        reconstructed = abs_helper(reconstructed)
        reference = abs_helper(reference)
    if isinstance(reconstructed, np.ndarray):
        reconstructed = torch.from_numpy(reconstructed)
    if isinstance(reference, np.ndarray):
        reference = torch.from_numpy(reference)

    reconstructed_np = normalize_np(reconstructed.detach().cpu().numpy())
    reference_np = normalize_np(reference.detach().cpu().numpy())
    
    if len(reference_np.shape) in [2]:
        pass
    elif len(reference_np.shape) in [3]:
        pass
    else:
        reference_np = reference_np.squeeze(0).squeeze(0)
        reconstructed_np = reconstructed_np.squeeze(0).squeeze(0)
    
    psnr_value = peak_signal_noise_ratio(reference_np, reconstructed_np, data_range=1)

    if len(reference_np.shape) in [2]:
        ssim_value = structural_similarity(reference_np, reconstructed_np, data_range=1)
    elif len(reference_np.shape) in [3]:
        ssim_value = structural_similarity(reference_np, reconstructed_np, data_range=1, channel_axis=0)
    else:
        raise ValueError("might be wrong")
    
    return psnr_value, ssim_value


def plot_multiples_in_one(figures_list, figures_name_list, dataset_name, png_file_name, save_dir = '.'):
    fig, axs = plt.subplots(1, len(figures_list), figsize=(5*len(figures_list), 5))
    for i, figure in enumerate(figures_list):
        index_image_for_plot = figures_list[i].clone()
        index_image_for_plot = index_image_for_plot.detach()
        if dataset_name == "ffhq":
            index_image_for_plot = renorm_from_minusonetoone_to_zeroone(index_image_for_plot.squeeze().detach().cpu().numpy())
            index_image_for_plot = np.clip(index_image_for_plot, 0, 1)
            if figures_name_list[i] in ["mask", "mask_further_degradation"]:
                axs[i].imshow(np.transpose(index_image_for_plot, (1, 2, 0)), cmap='gray')
            else:
                axs[i].imshow(np.transpose(index_image_for_plot, (1, 2, 0)))
            axs[i].set_title(figures_name_list[i])
        elif dataset_name == "fastmri":
            index_image_for_plot = abs_helper(index_image_for_plot).squeeze().detach().cpu().numpy()
            axs[i].imshow(index_image_for_plot, cmap='gray')
            axs[i].set_title(figures_name_list[i])
        axs[i].axis('off')

    if save_dir == ".":
        plt.savefig(os.path.join(f"{save_dir}", f"{png_file_name}.png"))
    else:
        os.makedirs(Path(save_dir), exist_ok=True)
        plt.savefig(os.path.join(f"{save_dir}", f"{png_file_name}.png"))
    plt.close(fig)


def save_individual_image(image, title, dataset_name, save_path, png_file_name):
    image_for_plot = image.clone()
    image_for_plot = image_for_plot.detach()
    fig, ax = plt.subplots(figsize=(5, 5))
    
    if dataset_name == 'ffhq':
        image_for_plot = renorm_from_minusonetoone_to_zeroone(image_for_plot.squeeze().detach().cpu().numpy())
        image_for_plot = np.clip(image_for_plot, 0, 1)
        image_for_plot = np.transpose(image_for_plot, (1, 2, 0))
        ax.imshow(image_for_plot)

    elif dataset_name == 'fastmri':
        image_for_plot = abs_helper(image_for_plot).squeeze().detach().cpu().numpy()
        ax.imshow(image_for_plot, cmap='gray')

    else:
        raise ValueError(f"Not yet to be implemented dataset_name :{dataset_name}")

    ax.axis('off')
    
    if title != None:
        ax.set_title(title)
        plt.savefig(os.path.join(save_path, f"{png_file_name}.png"))
    else:
        plt.savefig(os.path.join(save_path, f"{png_file_name}.png"), bbox_inches='tight', pad_inches=0)

    plt.close(fig)

def get_time_alpha_to_specific_noiselevel(noise_level_to_get_time, beta_at_clean, denoiser_network_type, num_diffusion_timesteps = 1000, last_time_step = 0):
    if denoiser_network_type == "vp_score":
        scale = 1000 / num_diffusion_timesteps
        assert scale == 1

        beta_start = scale * beta_at_clean
        beta_end = scale * 0.02
        beta_array = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64) # 999 to 0
        alpha_array = 1 - beta_array
        alphas_array = np.cumprod(alpha_array, axis=0)

        discrete_steps = 1000000
        extended_length = discrete_steps
        assert discrete_steps >= extended_length
        
        new_indices = np.linspace(0, len(alphas_array) - 1, discrete_steps)
        denoiser_noise_sigma_array = np.sqrt((1-alphas_array)/(alphas_array))
        extended_denoiser_noise_sigma_array = np.interp(new_indices, np.arange(len(alphas_array)), denoiser_noise_sigma_array)
        extended_alphas_array = 1/(1+np.square(extended_denoiser_noise_sigma_array))
        extended_denoiser_time_array = np.linspace(0, num_diffusion_timesteps - 1, discrete_steps)
        extended_time_array = np.linspace(0, num_diffusion_timesteps - 1, extended_length)

        min_distance = 10000
        matching_indicator = 1
        matching_time = 0
        tolerance = 100

        for i, looped_noise in enumerate(extended_denoiser_noise_sigma_array):
            if np.isclose(looped_noise, noise_level_to_get_time, atol=tolerance):
                if abs(looped_noise - noise_level_to_get_time) < min_distance:
                    min_distance = abs(looped_noise - noise_level_to_get_time)
                    min_distance_time_idx = i
                    min_distance_time = extended_denoiser_time_array[i]
                    min_alphas = extended_alphas_array[i]
                    matching_indicator = 1
                else:
                    break

        assert int(extended_time_array[-1]) == num_diffusion_timesteps-1

        time_idx_array =  np.linspace(0, extended_length - 1, extended_length).astype(int)
        time_array = extended_denoiser_time_array
        time_array = np.where(extended_denoiser_time_array <= last_time_step, last_time_step, extended_denoiser_time_array)
        
        return min_distance_time, min_alphas
    else:
        raise ValueError("Not yet to be implemented")