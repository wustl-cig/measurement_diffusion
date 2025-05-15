# -----------------
# Importing from Python module
# -----------------
import torch
import numpy as np
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial
import tifffile as tiff

# -----------------
# Importing from files
# -----------------
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    dicts_to_dict,
    sr_create_model_and_diffusion,
    create_argparser,
)
from utility.file_utility import mkdir_exp_recording_folder, load_yaml, check_and_mkdir, save_metrics_to_csv, save_metrics_to_csv_per_line
from utility.data_utility import training_dataloader_wrapper
from utility.func_utility import save_individual_image
from datasets.fastMRI import fastMRI, ftran_non_mask, get_fastmri_mask, ftran
from datasets.ffhq import get_ffhqdataset, get_ffhq_mask
from inverse.condition_methods import get_conditioning_method
from inverse.measurements import get_noise, get_operator
from utility.dps_util.img_utils import mask_generator
from utility.metric_utility import compute_psnr_ssim_nmse_lpips, normalize_np
import time

def main():
    parser = create_argparser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--save_tiff', type=bool, default=True)
    args = parser.parse_args()
    
    # -----------------
    # Load configurations and extract information from those
    # -----------------
    task_config = load_yaml(args.task_config); gpu = args.gpu; save_tiff = args.save_tiff; device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'; device = torch.device(device_str); save_dir = task_config['setting']['save_dir']; noise_schedule = task_config['vp_diffusion']['noise_schedule']; diffusion_model_type  = task_config['vp_diffusion']['diffusion_model_type']; image_size = task_config['vp_diffusion']['image_size']; learn_sigma = task_config['vp_diffusion']['learn_sigma']; in_channels = task_config['vp_diffusion']['in_channels']; cond_channels = task_config['vp_diffusion']['cond_channels']; dataset_name = task_config['dataset']['dataset_name'].lower(); mask_pattern = task_config['dataset']['mask_pattern'].lower(); acceleration_rate = task_config['dataset']['acceleration_rate']; batch_size = task_config['setting']['batch_size']; num_workers = task_config['setting']['num_workers']; solve_inverse_problem = task_config['setting']['solve_inverse_problem']; num_channels = task_config['vp_diffusion']['num_channels']; num_res_blocks = task_config['vp_diffusion']['num_res_blocks']; channel_mult = task_config['vp_diffusion']['channel_mult']; class_cond = task_config['vp_diffusion']['class_cond']; use_checkpoint = task_config['vp_diffusion']['use_checkpoint']; attention_resolutions = task_config['vp_diffusion']['attention_resolutions']; num_heads = task_config['vp_diffusion']['num_heads']; num_head_channels = task_config['vp_diffusion']['num_head_channels']; num_heads_upsample = task_config['vp_diffusion']['num_heads_upsample']; use_scale_shift_norm = task_config['vp_diffusion']['use_scale_shift_norm']; dropout = task_config['vp_diffusion']['dropout']; resblock_updown = task_config['vp_diffusion']['resblock_updown']; use_fp16 = task_config['vp_diffusion']['use_fp16']; use_new_attention_order = task_config['vp_diffusion']['use_new_attention_order']; latent_type = task_config['vp_diffusion']['latent_type']; pretrained_model_dir = task_config['setting']['pretrained_model_dir']; num_samples = task_config['setting']['num_samples'] ; use_ddim = task_config['vp_diffusion']['use_ddim'] ; ddim_eta = task_config['vp_diffusion']['ddim_eta'] ; clip_denoised = task_config['vp_diffusion']['clip_denoised'] ; timestep_respacing = task_config['vp_diffusion']['timestep_respacing'] ; training_data_noiselevel = task_config['train']['training_data_noiselevel'] ; diffusion_predtype = task_config['vp_diffusion']['diffusion_predtype']; task_config['train']['lr']; predict_xstart = True if diffusion_predtype == "pred_xstart" else False; dataset_dir = task_config['setting']['dataset_dir']; is_demo = task_config['setting']['is_demo']
    
    assert diffusion_predtype in ["pred_xstart", "epsilon"]
    assert latent_type in ['image_space', 'measurement_space']
    # -----------------
    # Define stochastic loop parameter w
    # -----------------
    if diffusion_model_type == "measurement_diffusion":
        stochastic_loop = task_config['setting']['stochastic_loop']
    else:
        stochastic_loop = 0

    if solve_inverse_problem == True:
        inverse_problem_type = task_config['setting']['inverse_problem_type']
        inverse_problem_solver  = task_config['setting']['inverse_problem_solver']
        inverse_problem_config = task_config['inverse_problems'][inverse_problem_type]
        inverse_problem_solver_config = task_config['inverse_problem_solvers'][inverse_problem_solver]
        measurement_noise_level  = task_config['setting']['measurement_noise_level']
        measurement_noise_setup = {
            'name': 'gaussian',
            'sigma': measurement_noise_level
        }
        noiser = get_noise(**measurement_noise_setup)
        operator = get_operator(device=device, **inverse_problem_config)
        cond_method = get_conditioning_method(inverse_problem_solver_config['method'], operator, noiser, **inverse_problem_solver_config['params'])
        measurement_cond_fn = cond_method.conditioning

        if diffusion_model_type == "measurement_diffusion":
            stepsize_likelihood  = task_config['setting']['stepsize_likelihood']
        else:
            stepsize_likelihood = None
    else:
        inverse_problem_type = None
        inverse_problem_solver = None
        measurement_noise_level = None

    if (use_ddim == False) and timestep_respacing != "":
        raise ValueError("timestep_respacing should be specified when use_ddim is False")
    
    # -----------------
    # For the information to log the training configurations in CSV file.
    # -----------------
    if training_data_noiselevel == 0:
        tau_SURE = 0
        lr_SURE = 0
    elif training_data_noiselevel != 0 and diffusion_model_type == "measurement_diffusion":
        tau_SURE = task_config['train']['tau_SURE']
        lr_SURE = task_config['train']['lr_SURE']
    else:
        raise ValueError("Check the training_data_noiselevel and diffusion_model_type")
    
    # -----------------
    # Directory for saving the results
    # -----------------   
    if diffusion_model_type == "measurement_diffusion":
        save_dir, _ = mkdir_exp_recording_folder(save_dir=save_dir, acceleration_rate=acceleration_rate, dataset_name=dataset_name, mask_pattern = mask_pattern, image_size=image_size, batch_size = batch_size, training_data_noiselevel = training_data_noiselevel, tau_SURE = tau_SURE, lr_SURE = lr_SURE, stochastic_loop = stochastic_loop)
    else:
        save_dir, _ = mkdir_exp_recording_folder(save_dir=save_dir, acceleration_rate=acceleration_rate, dataset_name=dataset_name, mask_pattern = mask_pattern, image_size=image_size, batch_size = batch_size, training_data_noiselevel = training_data_noiselevel, tau_SURE = tau_SURE, lr_SURE = lr_SURE)
    os.environ['OPENAI_LOGDIR'] = str(save_dir) # set the logdir
    csv_file_path = os.path.join(save_dir, "results.csv")

    # -----------------
    # param_dicts is a centralized dictionary containing parameters shared across all later-used functions.
    # -----------------
    param_dicts = {
        "dataset_name": dataset_name,
        "cond_channels": cond_channels,
        "acceleration_rate": acceleration_rate,
        "diffusion_model_type": diffusion_model_type,
        "training_data_noiselevel": training_data_noiselevel,
        "measurement_noise_level": measurement_noise_level,
        "tau_SURE": tau_SURE,
        "mask_pattern": mask_pattern,
        "diffusion_predtype": diffusion_predtype,
        "lr_SURE": lr_SURE,
        "latent_type": latent_type,
        "solve_inverse_problem": solve_inverse_problem,
        "stochastic_loop": stochastic_loop,
        "inverse_problem_type": inverse_problem_type
    }
    
    if dataset_name == "ffhq" and diffusion_model_type == "measurement_diffusion":
        param_dicts['cumulative_small_mask'] = -1
        param_dicts['cumulative_small_mask_shift_x'] = -1
        param_dicts['cumulative_small_mask_shift_y'] = -1

    if solve_inverse_problem == True:
        param_dicts['stepsize_likelihood'] = stepsize_likelihood
    
    assert args.schedule_sampler in ["uniform"]

    # -----------------
    # Diffusion model configuarations
    # -----------------
    model_diffusion_dict = {
        'image_size': image_size,
        'large_size': image_size,
        'small_size': image_size,
        'num_channels': num_channels,
        'num_res_blocks': num_res_blocks,
        'num_heads': num_heads,
        'num_heads_upsample': num_heads_upsample,
        'num_head_channels': num_head_channels,
        'attention_resolutions': attention_resolutions,
        'channel_mult': channel_mult,
        'dropout': dropout,
        'class_cond': class_cond,
        'use_checkpoint': use_checkpoint,
        'use_scale_shift_norm': use_scale_shift_norm,
        'resblock_updown': resblock_updown,
        'use_fp16': use_fp16,
        'use_new_attention_order':use_new_attention_order,
        'learn_sigma': learn_sigma,
        'diffusion_steps': 1000,
        'noise_schedule': 'linear',
        'timestep_respacing': timestep_respacing,
        'use_kl': False,
        'predict_xstart': predict_xstart,
        'rescale_timesteps': True if use_ddim else False,
        'rescale_learned_sigmas': False,
        'in_channels': in_channels,
        'cond_channels': cond_channels,
        'diffusion_model_type': diffusion_model_type,
        'training_data_noiselevel': training_data_noiselevel,
        'for_training': False,
    }

    dist_util.setup_dist(gpu = gpu)
    logger.configure()

    
    model, diffusion = sr_create_model_and_diffusion(
        **dicts_to_dict(model_diffusion_dict, sr_model_and_diffusion_defaults().keys())
    )
    if pretrained_model_dir == '':
        raise ValueError("Model path is not specified. Then, model weight will be randomly assigned.")
    else:
        model.load_state_dict(
            dist_util.load_state_dict(pretrained_model_dir, map_location="cpu")
        )
    model.to(dist_util.dev(gpu = gpu))
    
    

    if dataset_name == "fastmri":
        if solve_inverse_problem == True:
            inv_problem_acceleration_rate = task_config['dataset']['inv_problem_acceleration_rate']
            dataset = fastMRI(mode='test', root = dataset_dir, acceleration_rate=inv_problem_acceleration_rate, mask_pattern=mask_pattern, image_size = image_size, diffusion_model_type=diffusion_model_type, noiselevel_on_measurement = training_data_noiselevel, is_demo = is_demo)
            dataloader = DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False)
        else:
            # * Note: for unconditional image generation, we load the training data to use the coil sensitivity maps used during training.
            dataset = fastMRI(mode='train', root = dataset_dir, acceleration_rate=acceleration_rate, mask_pattern=mask_pattern, image_size = image_size, diffusion_model_type=diffusion_model_type, noiselevel_on_measurement = training_data_noiselevel, is_demo = is_demo)
            dataloader = training_dataloader_wrapper(dataset, batch_size=batch_size, num_workers=num_workers)

    elif dataset_name == "ffhq":
        if solve_inverse_problem == True:
            transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
            
            dataset = get_ffhqdataset(mode = "test", root = dataset_dir, name = dataset_name, transforms=transform, degradation_ratio = acceleration_rate, noiselevel_on_measurement = training_data_noiselevel, diffusion_model_type = diffusion_model_type, mask_pattern = mask_pattern)
            dataloader = DataLoader(dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    shuffle=False,
                    drop_last=False)

        else:
            transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
            dataset = get_ffhqdataset(mode = "train", root = dataset_dir, name = dataset_name, transforms=transform, degradation_ratio = acceleration_rate, noiselevel_on_measurement = training_data_noiselevel, diffusion_model_type = diffusion_model_type, mask_pattern = mask_pattern)
            dataloader = training_dataloader_wrapper(dataset, batch_size=1, num_workers=num_workers)

    else:
        raise ValueError(f"Check the dataset_name: {dataset_name}")

    if use_fp16:
        model.convert_to_fp16()

    model.eval()

    sample_fn = (
            diffusion.p_sample_loop if not use_ddim else diffusion.ddim_sample_loop
    )
    
    if param_dicts['solve_inverse_problem'] == True:
        if inverse_problem_type in ['box_inpainting', 'random_inpainting']:
            mask_gen = mask_generator(
                **inverse_problem_config['mask_opt']
                )
            
        all_images = []
        all_labels = []

        inputs = []
        recons = []
        gts = []

        input_psnr_list = []
        recon_psnr_list = []
        input_ssim_list = []
        recon_ssim_list = []
        input_nmse_list = []
        recon_nmse_list = []
        input_lpips_list = []
        recon_lpips_list = []

        assert diffusion_model_type in ["measurement_diffusion", "ambient_diffusion", "unconditional_diffusion"]
        if diffusion_model_type == "measurement_diffusion":
            shortcut_diffusion_model_type = "msm"
        elif diffusion_model_type == "unconditional_diffusion":
            shortcut_diffusion_model_type = "uncond"
        else:
            raise ValueError("Check the diffusion_model_type")
        csv_file_path_image_per_line = os.path.join(save_dir, f"{shortcut_diffusion_model_type}_{dataset_name}_{inverse_problem_type}_{inverse_problem_solver}_{mask_pattern}_{acceleration_rate}_eta_{measurement_noise_level}_quickval_{num_samples}.csv")

    else:
        measurement_cond_fn = None
    
    
    logger.log(f"# ------------\nSolve inverse problem with {diffusion_model_type}...\n# ------------\n") if param_dicts['solve_inverse_problem'] == True else logger.log(f"# ------------\nGenerating images with {diffusion_model_type}\n# ------------\n")
    all_images = []
    all_labels = []
    count = 0
    total_sampling_time = 0.
    for data_index, data_dicts in enumerate(dataloader):
        x_gt, smps = data_dicts['x'], data_dicts['smps']
        x_gt = (x_gt.squeeze(1)).to(device) 
        smps = (smps.squeeze(1)).to(device) 
        
        if (diffusion_model_type == "measurement_diffusion") and (dataset_name == "fastmri"):
            assert latent_type == "measurement_space"
            num_sensitivity_maps = smps.shape[-1]
            latent_shape = (batch_size, in_channels, image_size, image_size, num_sensitivity_maps)
        else:
            assert latent_type == "image_space"
            latent_shape = (batch_size, in_channels, image_size, image_size)

        model_kwargs = {}

        if param_dicts['solve_inverse_problem'] == True:
            if dataset_name in ["ffhq"]:
                if inverse_problem_type in ['box_inpainting', 'random_inpainting']:
                    mask = mask_gen(x_gt)
                    mask = mask[:, 0, :, :].unsqueeze(dim=0)
                    operator.set_mask(mask)
                    measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
                    mask = mask.to(device)
                    model_kwargs['low_res'] = mask.to(device)
                    model_kwargs['smps'] = mask.to(device)
                    y_hat = operator.forward(x_gt, mask=mask)
                    y_hat = noiser(y_hat)
                    y_hat = operator.forward(y_hat, mask=mask)
                else:
                    y_hat = operator.forward(x_gt)
                    y_hat = noiser(y_hat)
                    model_kwargs['low_res'] = y_hat
                    model_kwargs['smps'] = y_hat

            elif dataset_name == "fastmri":
                b,c,h,w = x_gt.shape
                
                assert mask_pattern == "randomly_cartesian", f"mask_pattern in the paper is randomly_cartesian. But it is {mask_pattern}"

                mask, mask_np = get_fastmri_mask(batch_size = b, image_w_h = h, mask_pattern = mask_pattern, acceleration_rate = inv_problem_acceleration_rate, dataset_name = "fastmri")
                measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
                mask = mask.to(device)
                model_kwargs['low_res'] = mask.to(device)
                model_kwargs['smps'] = smps.to(device)
                y_hat = operator.forward(x_gt, mask=model_kwargs['low_res'], smps = model_kwargs['smps'])
                y_hat = noiser(y_hat)
            else:
                raise ValueError(f"Check the dataset_name: {dataset_name}")
            
            param_dicts['operator'] = operator

        else:
            y_hat = None
           
            if dataset_name == "fastmri" and diffusion_model_type in ["measurement_diffusion", "unconditional_diffusion"]:
                b,c,h,w = x_gt.shape
                
                mask, mask_np = get_fastmri_mask(batch_size = b, image_w_h = h, mask_pattern = mask_pattern, acceleration_rate = acceleration_rate, dataset_name = "fastmri")
                mask = mask.to(device)
                model_kwargs['low_res'] = mask
                model_kwargs['smps'] = smps.to(device)

            else:
                b,c,h,w = x_gt.shape
                mask = get_ffhq_mask(batch_size = b, image_w_h = h, mask_pattern = 'random_box', acceleration_rate = acceleration_rate, dataset_name = "ffhq")
                mask = mask.to(device)
                model_kwargs['low_res'] = mask
                model_kwargs['smps'] = mask

        param_dicts['count'] = data_index

        if diffusion_model_type in ['measurement_diffusion']:
            param_dicts['previous_pred_ystart'] = None
            param_dicts['cumulative_mask'] = torch.zeros((1, 1, x_gt.shape[2], x_gt.shape[3]), device=x_gt.device)
        
        else:
            pass
        
        if (param_dicts['solve_inverse_problem'] == True) and dataset_name == "fastmri":
            param_dicts['measurement_for_inverse_problem'] = y_hat
            param_dicts['mask_for_inverse_problem'] = mask
            param_dicts['smps_for_inverse_problem'] = model_kwargs['smps']
        elif (param_dicts['solve_inverse_problem'] == True) and dataset_name == "ffhq":
            param_dicts['measurement_for_inverse_problem'] = y_hat
            param_dicts['mask_for_inverse_problem'] = None
            param_dicts['smps_for_inverse_problem'] = None
        elif param_dicts['solve_inverse_problem'] == False:
            pass
            
        else:
            raise ValueError("solve_inverse_problem should be either True or False")
        
        start_time = time.time()  # Record the start time
        # -----------------
        # Run sampling (DDIM for acceleration or DDPM for normal sampling)
        # -----------------
        if use_ddim == True:
            sample = sample_fn(
                model,
                shape = latent_shape,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                param_dicts = param_dicts,
                eta = ddim_eta,
                x_gt = x_gt,
                measurement = y_hat,
                measurement_cond_fn = measurement_cond_fn
            )
        else:
            sample = sample_fn(
                model,
                shape = latent_shape,
                clip_denoised=clip_denoised,
                model_kwargs=model_kwargs,
                param_dicts = param_dicts,
                x_gt = x_gt,
                measurement = y_hat,
                measurement_cond_fn = measurement_cond_fn
            )

        end_time = time.time()  # Record the start time
        sampling_time = end_time - start_time
        total_sampling_time += sampling_time
        avg_sampling_time = total_sampling_time / (data_index + 1)
        # print(f'-------\n(AVG) sampling time of {data_index+1}: {avg_sampling_time:.4f} seconds\n-------') # * activate this to print the average sampling time

        # -----------------
        # Sampling is all done, the code below is for saving the results and visualization.
        # -----------------
        if solve_inverse_problem == False:
            if in_channels == 3: # RGB natural images
                sample = sample.clamp(-1, 1)
            elif in_channels == 2: # MRI images
                pass
            if len(sample.shape) == 5 and dataset_name == "fastmri":
                # * Note: If the `sample' is in the measurement domain, we need to apply the inverse Fourier transform to get the image domain.
                sample = ftran_non_mask(sample, smps = model_kwargs['smps'])
            else:
                pass
            
            if (param_dicts['solve_inverse_problem'] == False) and dataset_name == "fastmri":
                check_and_mkdir(save_dir / 'images')
                save_individual_image(sample[0].unsqueeze(0), title = None, dataset_name = dataset_name, save_path = save_dir / 'images', png_file_name = f"{data_index}_sample")

            elif (param_dicts['solve_inverse_problem'] == False) and dataset_name == "fastmri" and acceleration_rate == 1:
                check_and_mkdir(save_dir / 'images')
                save_individual_image(x_gt[0].unsqueeze(0), title = None, dataset_name = dataset_name, save_path = save_dir / 'images', png_file_name = f"{data_index}_sample")
                
            elif (param_dicts['solve_inverse_problem'] == False) and dataset_name == "ffhq":
                check_and_mkdir(save_dir / 'images')
                save_individual_image(sample[0].unsqueeze(0), title = None, dataset_name = dataset_name, save_path = save_dir / 'images', png_file_name = f"{data_index}_sample")
            else:
                raise ValueError("Check the condition")

        elif solve_inverse_problem == True:

            if dataset_name in ["ffhq", "imagenet"]:
                if inverse_problem_type in ['super_resolution']:
                    y_hat = operator.transpose(y_hat)

                y_hat_np = normalize_np(y_hat.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
                sample_np = normalize_np(sample.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
                x_gt_np = normalize_np(x_gt.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
                
                input_img = torch.tensor(y_hat_np).to(device)
                recon_img = torch.tensor(sample_np).to(device)
                recon_gt = torch.tensor(x_gt_np).to(device)

            elif dataset_name == "fastmri":
                input_img = torch.abs(torch.view_as_complex(ftran(y_hat, mask = model_kwargs['low_res'], smps = model_kwargs['smps'])[0].permute(1,2,0).contiguous().cpu().detach()))
                input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min())

                if diffusion_model_type == "measurement_diffusion":
                    sample = ftran_non_mask(sample, smps = model_kwargs['smps'])
                elif diffusion_model_type == "ambient_diffusion":
                    pass

                recon_img = torch.abs(torch.view_as_complex(sample[0].permute(1,2,0).contiguous().cpu().detach()))
                recon_img = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min())

                recon_gt = torch.abs(torch.view_as_complex(x_gt[0].permute(1, 2, 0).contiguous().cpu().detach()))
                recon_gt = (recon_gt - recon_gt.min()) / (recon_gt.max() - recon_gt.min())

                mask_roi = recon_gt.clone()
                mask_roi[mask_roi != 0] = 1

            else:
                raise ValueError(f"Check the dataset_name: {dataset_name}")

            input_psnr_value, input_ssim_value, input_nmse_value, input_lpips_value = compute_psnr_ssim_nmse_lpips(input_img, recon_gt, device = device)
            recon_psnr_value, recon_ssim_value, recon_nmse_value, recon_lpips_value = compute_psnr_ssim_nmse_lpips(recon_img, recon_gt, device = device)
            
            if diffusion_model_type in ['measurement_diffusion']:
                print(f"[{data_index+1}/{len(dataloader)}] {inverse_problem_type} / {inverse_problem_solver} / R = {inv_problem_acceleration_rate} / {measurement_noise_level} / {ddim_eta}") if dataset_name == "fastmri" else print(f"[{data_index+1}/{len(dataloader)}] {inverse_problem_type} / {inverse_problem_solver} / MSM p = {acceleration_rate} / {measurement_noise_level} / {ddim_eta}")
                print(f"input_psnr_value: {input_psnr_value} / input_ssim_value: {input_ssim_value}\nrecon_psnr_value: {recon_psnr_value} / recon_ssim_value: {recon_ssim_value}")
            else:
                raise ValueError(f"Check the inverse_problem_solver {inverse_problem_solver}")
            
            save_metrics_to_csv_per_line(file_path = csv_file_path_image_per_line, data_index = data_index, inverse_problem_type = inverse_problem_type, inverse_problem_solver = inverse_problem_solver, acceleration_rate = inv_problem_acceleration_rate, measurement_noise_level = measurement_noise_level, ddim_eta = ddim_eta, input_psnr_value = input_psnr_value.item(), input_ssim_value = input_ssim_value.item(), input_nmse_value = input_nmse_value.item(), input_lpips_value = input_lpips_value, recon_psnr_value = recon_psnr_value.item(), recon_ssim_value = recon_ssim_value.item(), recon_nmse_value = recon_nmse_value.item(), recon_lpips_value = recon_lpips_value) if dataset_name == "fastmri" else save_metrics_to_csv_per_line(file_path = csv_file_path_image_per_line, data_index = data_index, inverse_problem_type = inverse_problem_type, inverse_problem_solver = inverse_problem_solver, acceleration_rate = acceleration_rate, measurement_noise_level = measurement_noise_level, ddim_eta = ddim_eta, input_psnr_value = input_psnr_value.item(), input_ssim_value = input_ssim_value.item(), input_nmse_value = input_nmse_value.item(), input_lpips_value = input_lpips_value, recon_psnr_value = recon_psnr_value.item(), recon_ssim_value = recon_ssim_value.item(), recon_nmse_value = recon_nmse_value.item(), recon_lpips_value = recon_lpips_value)
            
            inputs.append(input_img.unsqueeze(0))
            recons.append(recon_img.unsqueeze(0))
            gts.append(recon_gt.unsqueeze(0))
            input_psnr_list.append(input_psnr_value.item())
            recon_psnr_list.append(recon_psnr_value)
            input_ssim_list.append(input_ssim_value)
            recon_ssim_list.append(recon_ssim_value)
            input_nmse_list.append(input_nmse_value)
            recon_nmse_list.append(recon_nmse_value)
            input_lpips_list.append(input_lpips_value)
            recon_lpips_list.append(recon_lpips_value)
            
        if (data_index == num_samples):
            print(f'----------------- Quick Validation -----------------')
            break


    if solve_inverse_problem == True:
        avg_input_psnr_value = np.mean([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in input_psnr_list]) ; avg_input_ssim_value = np.mean([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in input_ssim_list]) ; avg_input_nmse_value = np.mean([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in input_nmse_list]) ; avg_input_lpips_value = np.mean([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in input_lpips_list])   
        avg_recon_psnr_value = np.mean([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in recon_psnr_list]) ; avg_recon_ssim_value = np.mean([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in recon_ssim_list]) ; avg_recon_nmse_value = np.mean([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in recon_nmse_list]) ; avg_recon_lpips_value = np.mean([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in recon_lpips_list])   
        std_input_psnr_value = np.std([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in input_psnr_list]) ; std_input_ssim_value = np.std([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in input_ssim_list]) ; std_input_nmse_value = np.std([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in input_nmse_list]) ; std_input_lpips_value = np.std([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in input_lpips_list])   
        std_recon_psnr_value = np.std([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in recon_psnr_list]) ; std_recon_ssim_value = np.std([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in recon_ssim_list]) ; std_recon_nmse_value = np.std([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in recon_nmse_list]) ; std_recon_lpips_value = np.std([t.cpu().numpy() if isinstance(t, torch.Tensor) else t for t in recon_lpips_list])   

        print('----------------- Summary -----------------')
        print(f"{inverse_problem_type} / {inverse_problem_solver} / R = {inv_problem_acceleration_rate} / {measurement_noise_level} / {ddim_eta}") if dataset_name == "fastmri" else print(f"{inverse_problem_type} / {inverse_problem_solver} / {acceleration_rate} / {measurement_noise_level} / {ddim_eta}")
        print('----------------- Total Average -----------------')
        print(f"avg input psnr: {avg_input_psnr_value} / avg input ssim: {avg_input_ssim_value} \navg recon psnr: {avg_recon_psnr_value} / avg recon ssim: {avg_recon_ssim_value}")
        print('----------------- Total STD -----------------')
        print(f"std input psnr: {std_input_psnr_value} / std input ssim: {std_input_ssim_value} \nstd recon psnr: {std_recon_psnr_value} / std recon ssim: {std_recon_ssim_value}")
        

        if save_tiff == True:
            recon_arr = torch.cat(recons, axis=0)
            recon_arr *= 255
            recon_arr = recon_arr.to("cpu", torch.uint8).numpy()

            input_arr = torch.cat(inputs, axis=0)
            input_arr *= 255
            input_arr = input_arr.to("cpu", torch.uint8).numpy()

            gt_arr = torch.cat(gts, axis=0)
            gt_arr *= 255
            gt_arr = gt_arr.to("cpu", torch.uint8).numpy()

            tiff.imwrite(os.path.join(save_dir, f"input_acc{inv_problem_acceleration_rate}_sigma{measurement_noise_level}.tiff"), input_arr, imagej=True) if dataset_name == "fastmri" else tiff.imwrite(os.path.join(save_dir, f"input_acc{acceleration_rate}_sigma{measurement_noise_level}.tiff"), input_arr, imagej=True)
            tiff.imwrite(os.path.join(save_dir, f"recon_{shortcut_diffusion_model_type}_{inverse_problem_type}_acc{inv_problem_acceleration_rate}_sigma{measurement_noise_level}_{timestep_respacing}_ddimeta{ddim_eta}.tiff"), recon_arr, imagej=True) if dataset_name == "fastmri" else tiff.imwrite(os.path.join(save_dir, f"recon_{shortcut_diffusion_model_type}_{inverse_problem_type}_acc{acceleration_rate}_sigma{measurement_noise_level}_{timestep_respacing}_ddimeta{ddim_eta}.tiff"), recon_arr, imagej=True)
            tiff.imwrite(os.path.join(save_dir, f"gt.tiff"), gt_arr, imagej=True)

        sweep_idx_str = str(0)
        
        save_metrics_to_csv(file_path = csv_file_path, sweep_idx = sweep_idx_str, inverse_problem_type = inverse_problem_type, inverse_problem_solver = inverse_problem_solver, acceleration_rate = inv_problem_acceleration_rate, measurement_noise_level = measurement_noise_level, ddim_eta = ddim_eta, avg_input_psnr_value = avg_input_psnr_value, avg_input_ssim_value = avg_input_ssim_value, avg_input_nmse_value = avg_input_nmse_value, avg_input_lpips_value = avg_input_lpips_value, std_input_psnr_value = std_input_psnr_value, std_input_ssim_value = std_input_ssim_value, std_input_nmse_value = std_input_nmse_value, std_input_lpips_value = std_input_lpips_value, avg_recon_psnr_value = avg_recon_psnr_value, avg_recon_ssim_value = avg_recon_ssim_value, avg_recon_nmse_value = avg_recon_nmse_value, avg_recon_lpips_value = avg_recon_lpips_value, std_recon_psnr_value = std_recon_psnr_value, std_recon_ssim_value = std_recon_ssim_value, std_recon_nmse_value = std_recon_nmse_value, std_recon_lpips_value = std_recon_lpips_value) if dataset_name == "fastmri" else save_metrics_to_csv(file_path = csv_file_path, sweep_idx = sweep_idx_str, inverse_problem_type = inverse_problem_type, inverse_problem_solver = inverse_problem_solver, acceleration_rate = acceleration_rate, measurement_noise_level = measurement_noise_level, ddim_eta = ddim_eta, avg_input_psnr_value = avg_input_psnr_value, avg_input_ssim_value = avg_input_ssim_value, avg_input_nmse_value = avg_input_nmse_value, avg_input_lpips_value = avg_input_lpips_value, std_input_psnr_value = std_input_psnr_value, std_input_ssim_value = std_input_ssim_value, std_input_nmse_value = std_input_nmse_value, std_input_lpips_value = std_input_lpips_value, avg_recon_psnr_value = avg_recon_psnr_value, avg_recon_ssim_value = avg_recon_ssim_value, avg_recon_nmse_value = avg_recon_nmse_value, avg_recon_lpips_value = avg_recon_lpips_value, std_recon_psnr_value = std_recon_psnr_value, std_recon_ssim_value = std_recon_ssim_value, std_recon_nmse_value = std_recon_nmse_value, std_recon_lpips_value = std_recon_lpips_value)

    logger.log(f"# ------------\nSampling complete\n# ------------\n")


if __name__ == "__main__":
    main()
