from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torch
import matplotlib.pyplot as plt
import numpy as np
import random


__DATASET__ = {}

def save_individual_image(image, title, save_path):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    # Make directory first
    # check_and_mkdir(save_path)
    plt.savefig(save_path)
    plt.close(fig)

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper

def random_box(b, h, w, degradation_ratio):
    if h == 256:
        patch_size = 32
    elif h == 128:
        patch_size = 16
    elif h == 64:
        patch_size = 8
    else:
        raise ValueError(f"Unknown image size {h}.")
    coarse_mask = (torch.rand((b, 1, h // patch_size, w // patch_size)) > degradation_ratio).float()

    mask = F.interpolate(coarse_mask, size=(h, w), mode='bicubic', align_corners=False)

    mask = (mask > 0.5).int()

    return mask

def generate_valid_masks(b, patch_size, degradation_ratio, base_mask_and_mask_out_of_this = None):
    num_zeros = int(patch_size * patch_size * (degradation_ratio))

    
    if base_mask_and_mask_out_of_this == None:
        masking_out_of_this = torch.ones((b, 1, patch_size, patch_size), dtype=torch.int)
    else:
        masking_out_of_this = base_mask_and_mask_out_of_this.clone()

    flat_mask = masking_out_of_this.view(b, -1)

    for i in range(b):
        # random seed 
        idx = torch.randperm(flat_mask.size(1))[:num_zeros]  # Ensure enough non-zero values
        flat_mask[i, idx] = 0

    return masking_out_of_this.view(b, 1, patch_size, patch_size)

def generate_valid_masks_for_sampling(b, patch_size, degradation_ratio, base_mask_and_mask_out_of_this = None, timesteps = None):
    num_zeros = int(patch_size * patch_size * (degradation_ratio))

    
    if base_mask_and_mask_out_of_this == None:
        masking_out_of_this = torch.ones((b, 1, patch_size, patch_size), dtype=torch.int)
    else:
        masking_out_of_this = base_mask_and_mask_out_of_this.clone()

    flat_mask = masking_out_of_this.view(b, -1)

    for i in range(b):
        idx = torch.randperm(flat_mask.size(1))[:num_zeros]  # Ensure enough non-zero values
        flat_mask[i, idx] = 0

    return masking_out_of_this.view(b, 1, patch_size, patch_size)




def get_random_dust_mask(b, h, w, degradation_ratio, mask_full_rgb = True):
    survival_probability = 1-degradation_ratio
    corruption_mask = np.random.binomial(1, survival_probability, size=(1, 3, h, w)).astype(np.float32)
    corruption_mask = torch.tensor(corruption_mask, dtype=torch.float32)
    
    if mask_full_rgb:
        corruption_mask = corruption_mask[:, 0]
        corruption_mask = corruption_mask.repeat([3, 1, 1, 1]).transpose(1, 0)

    return corruption_mask

import torch.nn.functional as F

@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, diffusion_model_type, noiselevel_on_measurement, degradation_ratio, mask_pattern, root: str, mode: str = "train", transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        all_fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(all_fpaths) > 0, "File list is empty. Check the root."
        
        if len(all_fpaths) < 100:
            # print(f"# !!!!!!!!!!!!\nWarning: Only {len(all_fpaths)} images found in the dataset. This may only be good for demo, but not be sufficient for training.\n# !!!!!!!!!!!!")
            self.fpaths = all_fpaths[:]  # Use 1000–70000 for training
        else:
            if mode == "train":
                self.fpaths = all_fpaths[1000:]  # Use 1000–70000 for training
            elif mode == "test":
                self.fpaths = all_fpaths[:100]  # Use 0–1000 for testing
            else:
                raise ValueError(f"Unknown mode {mode}. Use 'train' or 'test'.")


        self.noiselevel_on_measurement = noiselevel_on_measurement
        self.degradation_ratio = degradation_ratio
        self.mode = mode  # Store mode for debugging or future extensions
        self.diffusion_model_type = diffusion_model_type
        self.mask_pattern = mask_pattern
        
    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        
        fpath = self.fpaths[index]
        x = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            x = self.transforms(x)
        else:
            raise ValueError("Not implemented yet.")

        x = x.unsqueeze(0)

        return {'x': x, 'smps': x}
        
def gsure_random_box(b, h, w, degradation_ratio, cumulative_small_mask_information = None, timesteps = None):
    patch_size_options = [4]
    patch_size = random.choice(patch_size_options)

    if cumulative_small_mask_information == None:
        small_mask = generate_valid_masks(b = b, patch_size = patch_size, degradation_ratio = degradation_ratio)
        mask = small_mask.repeat_interleave(h // patch_size, 2).repeat_interleave(w // patch_size, 3)
        shift_range = int((h // patch_size))
        shift_x = random.randint(0, shift_range)
        shift_y = random.randint(0, shift_range)
        mask = torch.roll(mask, shifts=(shift_x, shift_y), dims=(2, 3))
        
        return mask

    else:
        
        cumulative_small_mask, previous_shift_x, previous_shift_y = cumulative_small_mask_information
        
        # -----------------
        # if zero_ratio_in_previous_mask is smaller than degradation ratio, pick everything from zero_ratio and remaining from the mask
        # -----------------
        if previous_shift_x != -1 and previous_shift_y != -1:

            opposite_of_previous_small_mask = (cumulative_small_mask == 0).int() # I should include everyhing of this.
            
            gap_probability = ((degradation_ratio-(1-degradation_ratio))/(degradation_ratio))

            gap_probability = gap_probability - 0.1

            masking_out_of_this = generate_valid_masks_for_sampling(b = b, patch_size = patch_size, degradation_ratio = gap_probability, base_mask_and_mask_out_of_this = opposite_of_previous_small_mask, timesteps = timesteps)

            masking_out_of_previous_mask = generate_valid_masks_for_sampling(b = b, patch_size = patch_size, degradation_ratio = 0.95, base_mask_and_mask_out_of_this = cumulative_small_mask, timesteps = timesteps)
            
            small_mask = (masking_out_of_this*opposite_of_previous_small_mask)# + (opposite_of_previous_small_mask)
            small_mask = small_mask + masking_out_of_previous_mask
            small_mask = torch.clamp(small_mask, 0, 1)
            
            
            mask = small_mask.repeat_interleave(h // patch_size, 2).repeat_interleave(w // patch_size, 3)
            mask = torch.roll(mask, shifts=(previous_shift_x, previous_shift_y), dims=(2, 3))

            
            cumulative_small_mask = torch.ones_like(cumulative_small_mask)
            previous_shift_x = -1
            previous_shift_y = -1
            
        else:
            small_mask = generate_valid_masks_for_sampling(b = b, patch_size = patch_size, degradation_ratio = degradation_ratio, timesteps = timesteps)
            
            mask = small_mask.repeat_interleave(h // patch_size, 2).repeat_interleave(w // patch_size, 3)
            shift_range_min = int(0.0*(h // patch_size)) # * best
            shift_range_max = int(0.5*(h // patch_size)) # * best

            shift_x = random.randint(shift_range_min, shift_range_max)
            shift_y = random.randint(shift_range_min, shift_range_max)
            mask = torch.roll(mask, shifts=(shift_x, shift_y), dims=(2, 3))
            
            cumulative_small_mask = small_mask
            previous_shift_x = shift_x
            previous_shift_y = shift_y

        return mask, [cumulative_small_mask, previous_shift_x, previous_shift_y]


def get_ffhqdataset(name: str, root: str, noiselevel_on_measurement: float, degradation_ratio: float, mode: str, diffusion_model_type: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    # Pass mode to the dataset initialization
    return __DATASET__[name](root=root, noiselevel_on_measurement=noiselevel_on_measurement, 
                             degradation_ratio=degradation_ratio, mode=mode, diffusion_model_type = diffusion_model_type, **kwargs)

def get_ffhqdataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader

def get_ffhq_mask(batch_size, image_w_h, mask_pattern, dataset_name, acceleration_rate, cumulative_small_mask_information = None, timesteps = None):
    assert (mask_pattern in ['randomly_cartesian', 'uniformly_cartesian', 'mix_cartesian', 'random_box', 'random_dust', 'ambient_box']) and (dataset_name in ['ffhq', 'fastmri']) 

    if mask_pattern == "random_box" and dataset_name == "ffhq":
        mask = gsure_random_box(b = batch_size, h = image_w_h, w = image_w_h, degradation_ratio = acceleration_rate, cumulative_small_mask_information = cumulative_small_mask_information, timesteps = timesteps)
    else:
        raise ValueError(f"Not yet to be implemented mask_pattern: {mask_pattern}")
    
    return mask