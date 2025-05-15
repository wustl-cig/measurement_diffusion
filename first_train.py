# -----------------
# Importing from Python module
# -----------------
import torch
import os
import torchvision.transforms as transforms

# -----------------
# Importing from files
# -----------------
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    dicts_to_dict,
    sr_create_model_and_diffusion,
    create_argparser
)
from guided_diffusion.train_util import TrainLoop
from utility.file_utility import mkdir_exp_recording_folder, load_yaml
from utility.data_utility import training_dataloader_wrapper
from utility.func_utility import get_time_alpha_to_specific_noiselevel
from datasets.fastMRI import fastMRI
from datasets.ffhq import get_ffhqdataset

def main():
    parser = create_argparser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_config', type=str)
    args = parser.parse_args()
    
    # -----------------
    # Load configurations and extract information from those
    # -----------------
    task_config = load_yaml(args.task_config); gpu = args.gpu; device_str = f"cuda:{gpu}" if torch.cuda.is_available() else 'cpu'; device = torch.device(device_str) ; save_dir = task_config['setting']['save_dir']; noise_schedule = task_config['vp_diffusion']['noise_schedule']; diffusion_model_type  = task_config['vp_diffusion']['diffusion_model_type']; image_size = task_config['vp_diffusion']['image_size']; microbatch = task_config['setting']['microbatch'] ; learn_sigma = task_config['vp_diffusion']['learn_sigma'] ; in_channels = task_config['vp_diffusion']['in_channels']; cond_channels = task_config['vp_diffusion']['cond_channels']; dataset_name = task_config['dataset']['dataset_name'].lower(); mask_pattern = task_config['dataset']['mask_pattern'].lower(); acceleration_rate = task_config['dataset']['acceleration_rate']; batch_size = task_config['setting']['batch_size']; num_workers = task_config['setting']['num_workers']; pretrained_model_dir = task_config['setting']['pretrained_model_dir']; save_interval = task_config['setting']['save_interval']; log_interval = task_config['setting']['log_interval']; pretrained_check_point = task_config['vp_diffusion']['model_path']; num_channels = task_config['vp_diffusion']['num_channels']; num_res_blocks = task_config['vp_diffusion']['num_res_blocks']; channel_mult = task_config['vp_diffusion']['channel_mult']; class_cond = task_config['vp_diffusion']['class_cond']; use_checkpoint = task_config['vp_diffusion']['use_checkpoint']; attention_resolutions = task_config['vp_diffusion']['attention_resolutions']; num_heads = task_config['vp_diffusion']['num_heads']; num_head_channels = task_config['vp_diffusion']['num_head_channels']; num_heads_upsample = task_config['vp_diffusion']['num_heads_upsample']; use_scale_shift_norm = task_config['vp_diffusion']['use_scale_shift_norm']; dropout = task_config['vp_diffusion']['dropout']; resblock_updown = task_config['vp_diffusion']['resblock_updown']; use_fp16 = task_config['vp_diffusion']['use_fp16']; use_new_attention_order = task_config['vp_diffusion']['use_new_attention_order']; latent_type = task_config['vp_diffusion']['latent_type']; training_data_noiselevel = task_config['train']['training_data_noiselevel']; diffusion_predtype = task_config['vp_diffusion']['diffusion_predtype']; lr = task_config['train']['lr']; use_ddim = task_config['vp_diffusion']['use_ddim']; timestep_respacing = task_config['vp_diffusion']['timestep_respacing']; dataset_dir = task_config['setting']['dataset_dir']; is_demo = task_config['setting']['is_demo']
    assert diffusion_predtype in ["pred_xstart", "epsilon"] # Note: Our code supports both epsilon and pred_xstart, but we observed epsilon is better.
    predict_xstart = True if diffusion_predtype == "pred_xstart" else False

    assert (use_ddim == True and timestep_respacing != "") or (use_ddim == False and timestep_respacing == "") # Note: use_ddim can also be specified during training to refer to skipped timesteps for more efficient training.
    if use_ddim == True and timestep_respacing != "":
        rescale_timesteps = True
    else:
        rescale_timesteps = False

    assert (learn_sigma == False) and (class_cond == False)     # Note: Our code is not intended to learn sigma
    assert latent_type in ['image_space', 'measurement_space']
    assert args.schedule_sampler in ["uniform"]

    # -----------------
    # In the case of noisy training data, get the corrsponding time and alpha to the training data noise level.
    # -----------------
    if (training_data_noiselevel != 0) and (diffusion_model_type == "measurement_diffusion"):
        t_and_alphas_at_training_data_noiselevel =  get_time_alpha_to_specific_noiselevel(noise_level_to_get_time = training_data_noiselevel, beta_at_clean = 0.0001, denoiser_network_type ="vp_score", num_diffusion_timesteps = 1000)
    else:
        t_and_alphas_at_training_data_noiselevel =  0, 0

    # -----------------
    # Here is the SURE setting for noisy training scenario
    # -----------------
    if training_data_noiselevel == 0:
        tau_SURE = 0
        lr_SURE = 0
    else:
        tau_SURE = task_config['train']['tau_SURE']
        lr_SURE = task_config['train']['lr_SURE']

    # -----------------
    # param_dicts is a centralized dictionary containing parameters shared across all later-used functions.
    # -----------------
    param_dicts = {
        "dataset_name": dataset_name,
        "cond_channels": cond_channels,
        "acceleration_rate": acceleration_rate,
        "diffusion_model_type": diffusion_model_type,
        "training_data_noiselevel": training_data_noiselevel,
        "t_and_alphas_at_training_data_noiselevel": t_and_alphas_at_training_data_noiselevel,
        "tau_SURE": tau_SURE,
        "mask_pattern": mask_pattern,
        "diffusion_predtype": diffusion_predtype,
        "lr_SURE": lr_SURE,
        "latent_type": latent_type
    }
    
    save_dir, _ = mkdir_exp_recording_folder(save_dir=save_dir, acceleration_rate=acceleration_rate, dataset_name=dataset_name, mask_pattern = mask_pattern, image_size=image_size, batch_size = batch_size, training_data_noiselevel = training_data_noiselevel, tau_SURE = tau_SURE, lr_SURE = lr_SURE)
    os.environ['OPENAI_LOGDIR'] = str(save_dir) # Note: This code save the necessary cache files in the directory specified by the user.
    
    # -----------------
    # Model and Diffusion Settings
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
        'rescale_timesteps': rescale_timesteps,
        'rescale_learned_sigmas': False,
        'in_channels': in_channels,
        'cond_channels': cond_channels,
        'diffusion_model_type': diffusion_model_type,
        'training_data_noiselevel': training_data_noiselevel,
        'for_training': True,
    }
    dist_util.setup_dist(gpu = gpu)
    logger.configure()
    model, diffusion = sr_create_model_and_diffusion(
        **dicts_to_dict(model_diffusion_dict, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev(gpu = gpu))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # -----------------
    # Load the dataloader
    # -----------------
    if dataset_name == "fastmri":
        dataset = fastMRI(mode='train', root = dataset_dir, acceleration_rate=acceleration_rate, mask_pattern=mask_pattern, image_size = image_size, diffusion_model_type=diffusion_model_type, noiselevel_on_measurement = training_data_noiselevel, is_demo = is_demo)
        dataloader = training_dataloader_wrapper(dataset, batch_size=batch_size, num_workers=num_workers)

    elif dataset_name == "ffhq":
        transform = transforms.Compose([transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])
        dataset = get_ffhqdataset(mode = "train", root = dataset_dir, name = dataset_name, transforms=transform, degradation_ratio = acceleration_rate, noiselevel_on_measurement = training_data_noiselevel, diffusion_model_type = diffusion_model_type, mask_pattern = mask_pattern)
        dataloader = training_dataloader_wrapper(dataset, batch_size=batch_size, num_workers=num_workers)

    else:
        raise ValueError("Check the dataset_name")

    # -----------------
    # Call the main training function
    # -----------------
    logger.log(f"# ------------\nTraining {diffusion_model_type} / Dataset: {dataset_name} / R: {acceleration_rate} / measurement noise: {training_data_noiselevel}...\n# ------------\n") if dataset_name == "fastmri" else logger.log(f"# ------------\nTraining {diffusion_model_type} / Dataset: {dataset_name} / p: {acceleration_rate} / measurement noise: {training_data_noiselevel}...\n# ------------\n")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=dataloader,
        batch_size=batch_size,
        microbatch=microbatch,
        lr=lr,
        ema_rate=args.ema_rate,
        log_interval=log_interval,
        save_interval=save_interval,
        resume_checkpoint=pretrained_model_dir,
        use_fp16=use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        gpu = gpu,
        param_dicts = param_dicts,
    ).run_loop()

if __name__ == "__main__":
    main()