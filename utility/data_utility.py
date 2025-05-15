import torch
import os
import time
from datetime import datetime
from pathlib import Path
import json
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def renorm_from_minusonetoone_to_zeroone(x):
    return (x + 1.) / 2.

class DatasetWrapper(Dataset):
    def __init__(self, parent_dataset, diffusion_model_type, dataset_name):
        self.dataset = parent_dataset
        self.dataset_name = dataset_name
        self.diffusion_model_type = diffusion_model_type

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        # print(f"item: {item}")
        if self.diffusion_model_type in ['measurement_conditional']:
            x = self.dataset[item]['x']
            x_hat = self.dataset[item]['x_hat']
            mask = (self.dataset[item]['mask']).unsqueeze(1)
            return x, y_hat, {'low_res': x_hat}

        elif self.diffusion_model_type in ['mask_conditional', 'measurement_diffusion']:
            if self.dataset_name == "fastmri":
                indexed_dataset = self.dataset[item]
                x = indexed_dataset['x'] # shape: 1, 2, 256, 256
                y_hat = indexed_dataset['y_hat']
                mask = indexed_dataset['mask']
                smps = indexed_dataset['smps']

                return x, y_hat, {'low_res': mask, 'smps': smps}
            elif self.dataset_name == "ffhq":
                indexed_dataset = self.dataset[item]
                x = indexed_dataset['x']#.unsqueeze(0)
                y_hat = indexed_dataset['y_hat']#.unsqueeze(0)
                mask = indexed_dataset['mask']#.unsqueeze(0)

                return x, y_hat, {'low_res': mask, 'smps': mask}

        elif self.diffusion_model_type in ['gsure_diffusion']:
            # raise ValueError(f"gsure_diffusion need to be implemented")
            if self.dataset_name == "fastmri":
                indexed_dataset = self.dataset[item]
                x = indexed_dataset['x']
                y_hat = indexed_dataset['y_hat']
                mask = indexed_dataset['mask']
                smps = indexed_dataset['smps']
                return  x, y_hat, {'low_res': mask, 'smps': smps}
        
            elif self.dataset_name == "ffhq":
                indexed_dataset = self.dataset[item]
                x = indexed_dataset['x']
                y_hat = indexed_dataset['y_hat']
                mask = indexed_dataset['mask']
                # raise ValueError(f"mask.shape: {mask.shape}")
                return x, y_hat, {'low_res': mask, 'smps': mask}
            else:
                raise ValueError(f"Check the dataset_name {self.dataset_name}")
        
        elif self.diffusion_model_type in ["unconditional_diffusion"]:
            indexed_dataset = self.dataset[item]
            x = indexed_dataset['x']
            y_hat = indexed_dataset['y_hat']
            mask = indexed_dataset['mask']
            smps = indexed_dataset['smps']

            return x, y_hat, {'low_res': mask, 'smps': smps}#, {'low_res': x_hat}
            """
            x = self.dataset[item]['x']
            y_hat = self.dataset[item]['y_hat']
            mask = self.dataset[item]['mask']
            return x, y_hat, {'low_res': mask, 'smps': mask}#, {'low_res': x_hat}
            """
    
        else:
            raise ValueError(f"Check the diffusion_model_type: {self.diffusion_model_type}")
            x = self.dataset[item]['x']
            # x_hat = self.dataset[item]['x_hat']
            return x, {}, {}#, {'low_res': x_hat}

def training_dataloader_wrapper(dataset, batch_size, num_workers):
    # raise ValueError(f"dataset: {dataset} \n batch_size: {batch_size} \n num_workers: {num_workers}")
    data_loader = DataLoader(
        dataset=dataset,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    while True:
        yield from data_loader


def crop_images(images, crop_width):
    # raise ValueError(f"images.shape: {images.shape}")
    if len(images.shape) != 4:
        raise ValueError("Check the size of the images")
    # crop_width = min(images.shape[2], images.shape[3])

    cropped_images = []

    for i in range(images.shape[0]):  # Iterate over the number of images
        img = images[i]  # Select one image
        for j in range(images.shape[1]):  # Iterate over the channels
            channel = img[j]  # Select one channel
            cropping_transform = transforms.CenterCrop((crop_width, crop_width))
            cropped_channel = cropping_transform(channel)
            cropped_images.append(cropped_channel)

    # Reshape the cropped images
    cropped_images = torch.stack(cropped_images, dim=0)
    cropped_images = cropped_images.view(images.shape[0], images.shape[1], crop_width, crop_width)

    return cropped_images

def abs_helper(x, axis=1, is_normalization=True):
    x = torch.sqrt(torch.sum(x ** 2, dim=axis, keepdim=True))

    if is_normalization:
        for i in range(x.shape[0]):
            x[i] = (x[i] - torch.min(x[i])) / (torch.max(x[i]) - torch.min(x[i]) + 1e-16)

    x = x.to(torch.float32)

    return x