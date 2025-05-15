import torch
import os
import time
from datetime import datetime
from pathlib import Path
import json
import math
import matplotlib.pyplot as plt
import yaml
import csv

    
def mkdir_exp_recording_folder(save_dir, acceleration_rate, dataset_name, mask_pattern, image_size, batch_size, training_data_noiselevel, tau_SURE, stochastic_loop = None, lr_SURE = None):
    current_time = time.time()
    current_date = datetime.now().strftime("%m%d%Y")
    current_hour_minute = datetime.now().strftime("%H%M")
    if mask_pattern == "uniformly_cartesian":
        mask_pattern = "unimask"
    elif mask_pattern == "randomly_cartesian":
        mask_pattern = "randmask"
    elif mask_pattern == "random_box":
        mask_pattern = "randbox"
    elif mask_pattern == "random_dust":
        mask_pattern = "randdust"
    elif mask_pattern == "ambient_box":
        mask_pattern = "ambientbox"
    elif mask_pattern == "mix_cartesian":
        mask_pattern = "mixmask"
    elif mask_pattern == "None":
        mask_pattern = "None"
    else:
        raise ValueError("mask_pattern should be one of 'uniformly_cartesian', 'randomly_cartesian', 'mix_cartesian'")
        
    if lr_SURE == None:
        unique_name = f"{current_date}_{current_hour_minute}_{dataset_name}_{mask_pattern}_acc{acceleration_rate}_img{image_size}_batch{batch_size}_noiseindata_{training_data_noiselevel}_tauSure_{tau_SURE}"
    else:
        unique_name = f"{current_date}_{current_hour_minute}_{dataset_name}_{mask_pattern}_acc{acceleration_rate}_img{image_size}_batch{batch_size}_noiseindata_{training_data_noiselevel}_tauSure_{tau_SURE}_lrSure_{lr_SURE}"
        
    if stochastic_loop != None:
        unique_name = f"{unique_name}_stochasticloop{stochastic_loop}"
        
    result_file = Path(save_dir) / unique_name / "results.csv"
    os.makedirs(Path(save_dir) / unique_name, exist_ok=True)
    result_dir = Path(save_dir) / unique_name
    return result_dir, result_file


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_metrics_to_csv(file_path, sweep_idx, inverse_problem_type, inverse_problem_solver, acceleration_rate, 
                        measurement_noise_level, ddim_eta, 
                        avg_input_psnr_value, avg_input_ssim_value, avg_input_nmse_value, avg_input_lpips_value,
                        std_input_psnr_value, std_input_ssim_value, std_input_nmse_value, std_input_lpips_value,
                        avg_recon_psnr_value, avg_recon_ssim_value, avg_recon_nmse_value, avg_recon_lpips_value,
                        std_recon_psnr_value, std_recon_ssim_value, std_recon_nmse_value, std_recon_lpips_value):
    
    header = [
        "sweep_idx", "inverse_problem_type", "inverse_problem_solver", "acceleration_rate", "measurement_noise_level", "ddim_eta", 
        "avg_input_psnr", "avg_input_ssim", "avg_input_nmse", "avg_input_lpips", 
        "std_input_psnr", "std_input_ssim", "std_input_nmse", "std_input_lpips", 
        "avg_recon_psnr", "avg_recon_ssim", "avg_recon_nmse", "avg_recon_lpips", 
        "std_recon_psnr", "std_recon_ssim", "std_recon_nmse", "std_recon_lpips"
    ]
    
    data = [
        sweep_idx, inverse_problem_type, inverse_problem_solver, acceleration_rate, measurement_noise_level, ddim_eta, 
        avg_input_psnr_value, avg_input_ssim_value, avg_input_nmse_value, avg_input_lpips_value,
        std_input_psnr_value, std_input_ssim_value, std_input_nmse_value, std_input_lpips_value,
        avg_recon_psnr_value, avg_recon_ssim_value, avg_recon_nmse_value, avg_recon_lpips_value,
        std_recon_psnr_value, std_recon_ssim_value, std_recon_nmse_value, std_recon_lpips_value
    ]
    
    file_exists = os.path.exists(file_path)
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)  # Write the header if file does not exist
        writer.writerow(data)  # Append new row

def save_metrics_to_csv_per_line(file_path, data_index, inverse_problem_type, inverse_problem_solver, acceleration_rate, 
                        measurement_noise_level, ddim_eta,
                        input_psnr_value, input_ssim_value, input_nmse_value, input_lpips_value,
                        recon_psnr_value, recon_ssim_value, recon_nmse_value, recon_lpips_value):
    
    header = [
        "data_index", "inverse_problem_type", "inverse_problem_solver", "acceleration_rate", "measurement_noise_level", "ddim_eta", 
        "input_psnr", "input_ssim", "input_nmse", "input_lpips", 
        "recon_psnr", "recon_ssim", "recon_nmse", "recon_lpips", 
    ]
    
    data = [
        data_index, inverse_problem_type, inverse_problem_solver, acceleration_rate, measurement_noise_level, ddim_eta, 
        input_psnr_value, input_ssim_value, input_nmse_value, input_lpips_value,
        recon_psnr_value, recon_ssim_value, recon_nmse_value, recon_lpips_value,
    ]
    
    file_exists = os.path.exists(file_path)
    
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data)
    
