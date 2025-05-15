# -----------------
# Importing from Python module
# -----------------
import numpy as np
import torch
import os
import time
from datetime import datetime
from pathlib import Path
import json
import math
import matplotlib.pyplot as plt
from torchvision import transforms
# -----------------
# Importing from files
# -----------------
from torch.utils.data import DataLoader, Dataset
from utility.data_utility import abs_helper, renorm_from_minusonetoone_to_zeroone


def plot_multiples_in_one(figures_list, figures_name_list, dataset_name, png_file_name, save_dir = '.'):
    fig, axs = plt.subplots(1, len(figures_list), figsize=(5*len(figures_list), 5))
    for i, figure in enumerate(figures_list):
        index_image_for_plot = figures_list[i].clone()
        index_image_for_plot = index_image_for_plot.detach()
        if dataset_name == "ffhq":
            index_image_for_plot = renorm_from_minusonetoone_to_zeroone(index_image_for_plot.squeeze(0).detach().cpu().numpy())

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
        if len(image_for_plot.shape) == 2: # In the case of grayscale image
            image_for_plot = np.expand_dims(image_for_plot, axis=0)
            ax.imshow(np.transpose(image_for_plot, (1, 2, 0)), cmap='gray')
        else:
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