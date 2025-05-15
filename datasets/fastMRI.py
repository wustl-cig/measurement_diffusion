import os
import numpy as np
import torch
import h5py
import tqdm
import tifffile
from torch.utils.data import Dataset

import h5py
import os
import numpy as np
import tqdm
import torch
import tifffile
import pandas
from torch.utils.data import Dataset
import random

from utility.data_utility import crop_images

DEMO_DATASHEET = pandas.read_csv(os.path.join('datasets/fmri_samples', 'DEMO_fastmri_brain_multicoil_20230102.csv'))
DATASHEET = pandas.read_csv(os.path.join('datasets/fmri_samples', 'fastmri_brain_multicoil_20230102.csv'))

def INDEX2_helper(idx, key_, is_demo = False):
    if is_demo == False:
        file_id_df = DATASHEET[key_][DATASHEET['INDEX'] == idx]
    else:
        file_id_df = DEMO_DATASHEET[key_][DEMO_DATASHEET['INDEX'] == idx]

    assert len(file_id_df.index) == 1

    return file_id_df[idx]

INDEX2FILE = lambda idx: INDEX2_helper(idx, 'FILE')
DEMO_INDEX2FILE = lambda idx: INDEX2_helper(idx, key_='FILE', is_demo = True)

def INDEX2DROP(idx, is_demo):
    ret = INDEX2_helper(idx, 'DROP', is_demo = is_demo)

    if ret in ['0', 'false', 'False', 0.0]:
        return False
    else:
        return True

def INDEX2SLICE_START(idx, is_demo):
    ret = INDEX2_helper(idx, 'SLICE_START', is_demo = is_demo)

    if isinstance(ret, np.float64) and ret >= 0:
        return int(ret)
    else:
        return None

def INDEX2SLICE_END(idx, is_demo):
    ret = INDEX2_helper(idx, 'SLICE_END', is_demo = is_demo)

    if isinstance(ret, np.float64) and ret >= 0:
        return int(ret)
    else:
        return None

def ftran_non_mask(y, smps):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """
    if len(smps.shape) == 5:
        # mask^H
        batch_size = y.shape[0]
        y = y.permute([0, 4, 2, 3, 1]).contiguous()
        y = torch.view_as_complex(y)
        smps = smps.permute([0, 1, 4, 2, 3]).contiguous().squeeze(1)
        # F^H
        y = torch.fft.ifftshift(y, [-2, -1])
        x = torch.fft.ifft2(y, norm='ortho')
        x = torch.fft.fftshift(x, [-2, -1])
        
        x = x * torch.conj(smps)

        x = x.sum(1)

        x = torch.view_as_real(x).permute([0, 3, 1, 2]).contiguous()
    else:
        x = y
        
    return x

def fmult_non_mask(x, smps):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """
    if len(smps.shape) == 5:
        y = x.permute([0, 2, 3, 1]).contiguous()
        y = (torch.view_as_complex(y)).unsqueeze(1)
        smps = smps.permute([0, 1, 4, 2, 3]).contiguous().squeeze(1)
        
        y = y * smps
        
        # F
        y = torch.fft.ifftshift(y, [-2, -1])
        y = torch.fft.fft2(y, norm='ortho')
        y = torch.fft.fftshift(y, [-2, -1])
        
        y = torch.view_as_real(y).permute([0, 4, 2, 3, 1]).contiguous()
    else:
        y = x
    
    return y


def ftran(y, smps, mask):
    """
    compute adjoint of fast MRI, x = smps^H F^H mask^H x

    :param y: under-sampled measurements, shape: batch, coils, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: zero-filled image
    """
    
    """
    y.shape: 1, 20, 320, 320
    mask.shape: 1, 1, 320, 320
    """
    if smps.shape == mask.shape: # ffhq case
        x = y*mask
    else:
        y = y.permute([0, 4, 2, 3, 1]).contiguous()
        y = torch.view_as_complex(y)

        y = y * mask
        
        smps = smps.permute([0, 1, 4, 2, 3]).contiguous().squeeze(1)

        # F^H
        y = torch.fft.ifftshift(y, [-2, -1])
        x = torch.fft.ifft2(y, norm='ortho')
        x = torch.fft.fftshift(x, [-2, -1])

        # smps^H
        x = x * torch.conj(smps)
        x = x.sum(1)
        # x = x.unsqueeze(1)
        x = torch.view_as_real(x).permute([0, 3, 1, 2]).contiguous()

    return x

def fmult(x, smps, mask):
    """
    compute forward of fast MRI, y = mask F smps x

    :param x: groundtruth or estimated image, shape: batch, width, height; dtype: complex
    :param smps: sensitivity maps, shape: batch, coils, width, height; dtype: complex
    :param mask: sampling mask, shape: batch, width, height; dtype: float/bool
    :return: undersampled measurement
    """

    if smps.shape == mask.shape: # ffhq case
        y = x*mask
        
    else:
        
        if (len(x.shape) == 5):
            raise ValueError()
            y = x.permute([0, 4, 2, 3, 1]).contiguous()
            y = torch.view_as_complex(y)

            # mask
            mask = mask.unsqueeze(1)
            
            y = y * mask
            
            y = torch.view_as_real(y).permute([0, 4, 2, 3, 1]).contiguous()

            return y

        else:
            y = x.permute([0, 2, 3, 1]).contiguous()
            y = (torch.view_as_complex(y)).unsqueeze(1)
            
            smps = smps.permute([0, 1, 4, 2, 3]).contiguous().squeeze(1)
            
            y = y * smps
            # F
            y = torch.fft.ifftshift(y, [-2, -1])
            y = torch.fft.fft2(y, norm='ortho')
            y = torch.fft.fftshift(y, [-2, -1])
            
            y = y * mask
            
        y = torch.view_as_real(y).permute([0, 4, 2, 3, 1]).contiguous()

    return y


def uniformly_cartesian_mask(img_size, acceleration_rate, cumulative_mask = None, acs_percentage: float = 0.2, randomly_return: bool = False):
    if cumulative_mask != None:
        raise ValueError("Maybe not intended condition.")
    if acceleration_rate == 0:
        return np.ones(img_size, dtype=np.float32)
    else:
        ny = img_size[-1]

        ACS_START_INDEX = (ny // 2) - 10
        ACS_END_INDEX = (ny // 2) + 10

        if ny % 2 == 0:
            ACS_END_INDEX -= 1

        mask = np.zeros(shape=(acceleration_rate,) + img_size, dtype=np.float32)
        
        # Generate a random starting offset
        random_offset = np.random.randint(0, acceleration_rate)
        
        mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1

        for i in range(ny):
            for j in range(acceleration_rate):
                if (i + random_offset) % acceleration_rate == j:
                    mask[j, ..., i] = 1

        if randomly_return:
            mask = mask[np.random.randint(0, acceleration_rate)]
        else:
            mask = mask[0]

        return mask
    
def randomly_cartesian_mask(img_size, acceleration_rate, cumulative_mask = None, acs_percentage: float = 0.2, randomly_return: bool = False):
    if acceleration_rate in [1]:
        return np.ones(img_size, dtype=np.float32)
    probability = 1/acceleration_rate
    
    ny = img_size[-1]

    ACS_START_INDEX = (ny // 2) - 10
    ACS_END_INDEX = (ny // 2) + 10

    if ny % 2 == 0:
        ACS_END_INDEX -= 1

    mask = np.zeros(shape=img_size, dtype=np.float32)
    mask[..., ACS_START_INDEX: (ACS_END_INDEX + 1)] = 1
    
    if cumulative_mask != None:
        cumulative_mask = cumulative_mask.squeeze(0).squeeze(0)
        num_zero_in_cumulative_mask = (cumulative_mask == 0).sum(dim=1).flatten()[0]
        for i in range(ny):
            if np.random.rand() < (1/(acceleration_rate)):
                mask[..., i] = 1
                
            mask_tensor = torch.from_numpy(mask)
            
            if num_zero_in_cumulative_mask <= (int(ny / acceleration_rate)+5): # Count number of zero
                if cumulative_mask[0, i] < 1:
                    mask[..., i] = 1
                else:
                    pass
            else:
                if cumulative_mask[0, i] < 1:
                    if np.random.rand() < (1/(acceleration_rate-1)):
                        mask[..., i] = 1
                else:
                    pass
    
    else:
        for i in range(ny):
            if np.random.rand() < probability:
                mask[..., i] = 1
                

    return mask


def mix_cartesian_mask(img_size, acceleration_rate, acs_percentage: float = 0.2, probability_randomly: float = 0.95):
    """
    Generate a mixed Cartesian mask with 90% probability of being randomly Cartesian and 10% of being uniform Cartesian.
    
    Parameters:
        img_size (tuple): Size of the image.
        acceleration_rate (int): Acceleration rate.
        acs_percentage (float): ACS region percentage. Default is 0.2.
        probability_randomly (float): Probability of using randomly Cartesian mask. Default is 0.9.
        
    Returns:
        np.ndarray: The generated mask.
    """
    if np.random.rand() < probability_randomly:
        # Generate a randomly Cartesian mask
        return randomly_cartesian_mask(img_size, acceleration_rate, acs_percentage)
    else:
        # Generate a uniform Cartesian mask
        return uniformly_cartesian_mask(img_size, acceleration_rate, acs_percentage)


_mask_fn = {
    'uniformly_cartesian': uniformly_cartesian_mask,
    'randomly_cartesian': randomly_cartesian_mask,
    'mix_cartesian': mix_cartesian_mask
}


def addwgn(x: torch.Tensor, input_snr):

    noiseNorm = torch.norm(x.flatten()) * 10 ** (-input_snr / 20)

    noise = torch.randn(x.size()).to(x.device)

    noise = noise / torch.norm(noise.flatten()) * noiseNorm

    y = x + noise

    return y


def check_and_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def np_normalize_minusoneto_plusone(x):
    x -= np.amin(x)
    x /= np.amax(x)

    x = (x * 2.)-1.

    return x

def np_torch_renormalize(x):
    x = (x + 1.)/2.
    return x

def np_normalize_to_uint8(x):
    x -= np.amin(x)
    x /= np.amax(x)

    x = x * 255
    x = x.astype(np.uint8)

    return x


def load_real_dataset_handle(
        idx,
        root: str,
        acceleration_rate: int = 1,
        is_return_y_smps_hat: bool = False,
        mask_pattern: str = 'uniformly_cartesian',
        smps_hat_method: str = 'eps',
        is_demo: bool = False,
):

    if not smps_hat_method == 'eps':
        raise NotImplementedError('smps_hat_method can only be eps now, but found %s' % smps_hat_method)

    x_hat_path = os.path.join(root, 'x_hat')

    x_hat_h5 = os.path.join(x_hat_path, INDEX2FILE(idx) + '.h5') if is_demo == False else os.path.join(x_hat_path, DEMO_INDEX2FILE(idx) + '.h5')

    smps_hat_path = os.path.join(root, 'smps_hat')

    smps_hat_h5 = os.path.join(smps_hat_path, INDEX2FILE(idx) + '.h5') if is_demo == False else os.path.join(smps_hat_path, DEMO_INDEX2FILE(idx) + '.h5')

    ret = {
        'x_hat': x_hat_h5
    }
    if is_return_y_smps_hat:
        ret.update({
            'smps_hat': smps_hat_h5,
        })

    return ret

def torch_complex_normalize(x):
    x_angle = torch.angle(x)
    x_abs = torch.abs(x)

    x_abs -= torch.min(x_abs)
    x_abs /= torch.max(x_abs)

    x = x_abs * np.exp(1j * x_angle)

    return x


def bshw_comp_to_bchws_real(data):
    assert len(data.shape) == 4
    data = torch.view_as_real(data)
    data = data.permute([0, 4, 2, 3, 1]).contiguous()
    return data

def bshw_comp_to_bchws_comp(data):
    assert len(data.shape) == 4
    data = data.permute([0, 2, 3, 1]).contiguous()
    data = data.unsqueeze(1)
    return data

def b1hw_comp_to_b2hw_real(data):
    assert len(data.shape) in [3,4]
    if len(data.shape) == 3:
        data = data.unsqueeze(1)
    assert data.shape[1] == 1
    data = data.squeeze(1)
    data = torch.view_as_real(data)
    data = data.permute([0, 3, 1, 2]).contiguous()
    return data

# Before torch_complex_normalize
def bchw_real_to_bhw_comp(data):
    assert len(data.shape) == 4
    data = data.permute([0, 2, 3, 1]).contiguous()
    data = torch.view_as_complex(data)
    return data

def bchws_real_to_bshw_comp(data):
    assert len(data.shape) == 5
    data = data.permute([0, 4, 2, 3, 1]).contiguous()
    data = torch.view_as_complex(data)
    return data

def b1hws_comp_to_bshw_comp(data):
    assert len(data.shape) == 5
    data = data.squeeze(1)
    data = data.permute([0, 3, 1, 2]).contiguous()
    return data


class fastMRI(Dataset):
    def __init__(
            self,
            mode,
            root,
            acceleration_rate,
            mask_pattern,
            image_size,
            noiselevel_on_measurement,
            is_demo,
            is_return_y_smps_hat: bool = True,
            smps_hat_method: str = 'eps',
            diffusion_model_type="uncond",
    ):

        if is_demo == False:
            if mode == 'train':
                idx_list = range(1355)
            elif mode == 'validation':
                raise ValueError("Not intended condition.")
            elif mode == 'test':
                idx_list = range(1355, 1377)
            else:
                raise ValueError(f"Unknown mode: {mode}")
        else:
            if mode == 'train':
                idx_list = [0]
            elif mode == 'validation':
                raise ValueError("Not intended condition.")
            elif mode == 'test':
                idx_list = [1]
            else:
                raise ValueError(f"Unknown mode: {mode}")
            
        self.diffusion_model_type = diffusion_model_type
        self.mask_pattern = mask_pattern
        self.__index_maps = []
        self.image_size = image_size
        self.noiselevel_on_measurement = noiselevel_on_measurement
        self.is_demo = is_demo

        for idx in idx_list:
            if INDEX2DROP(idx, is_demo = self.is_demo):
                continue

            ret = load_real_dataset_handle(
                idx = idx,
                root = root,
                acceleration_rate = 1,
                is_return_y_smps_hat = is_return_y_smps_hat,
                mask_pattern = mask_pattern,
                smps_hat_method = smps_hat_method,
                is_demo = self.is_demo
            )

            with h5py.File(ret['x_hat'], 'r') as f:
                num_slice = f['x_hat'].shape[0]

            if INDEX2SLICE_START(idx, is_demo = self.is_demo) is not None:
                slice_start = INDEX2SLICE_START(idx, is_demo = self.is_demo)
            else:
                slice_start = 0

            if INDEX2SLICE_END(idx, is_demo = self.is_demo) is not None:
                slice_end = INDEX2SLICE_END(idx, is_demo = self.is_demo)
            else:
                slice_end = num_slice - 5

            for s in range(slice_start, slice_end):
                self.__index_maps.append([ret, s])

            self.acceleration_rate = acceleration_rate

        self.is_return_y_smps_hat = is_return_y_smps_hat

    def __len__(self):
        return len(self.__index_maps)

    def __getitem__(self, item):

        ret, s = self.__index_maps[item]

        with h5py.File(ret['x_hat'], 'r', swmr=True) as f:
            x = f['x_hat'][s]

        with h5py.File(ret['smps_hat'], 'r', swmr=True) as f:
            smps_hat = f['smps_hat'][s]

        smps_hat = torch.from_numpy(smps_hat)

        x = torch.from_numpy(x)

        x = (crop_images(x.unsqueeze(0).unsqueeze(0), crop_width=self.image_size)).squeeze(1)        
        
        for i in range(x.shape[0]):
            x[i] = torch_complex_normalize(x[i])

        x = torch.view_as_real(x).permute([0, 3, 1, 2]).contiguous()
        
        smps_hat = (crop_images(smps_hat.unsqueeze(0), crop_width=self.image_size))#.squeeze(0)

        smps_hat = (smps_hat.permute([0, 2, 3, 1]).contiguous()).unsqueeze(1)
        
        return {'x': x, 'smps': smps_hat}


def normalize_complex(data, eps=0.):
    mag = np.abs(data)
    mag_std = mag.std()
    return data / (mag_std + eps), mag_std

def from_kspace_to_image(kspace, smps):
    kspace = ftran_non_mask(kspace, smps)
    return kspace

def from_image_to_kspace(image, smps):
    image = fmult_non_mask(image, smps)
    return image

def apply_mask_on_kspace(kspace, smps, mask):
    kspace = ftran(kspace, smps, mask)
    kspace = fmult(kspace, smps, mask)
    return kspace

def apply_mask_on_image(image, smps, mask):
    image = fmult(image, smps, mask)
    image = ftran_non_mask(image, smps = smps)

    return image

def apply_mask_on_kspace_wthout_ftranfmult(kspace, smps, mask):
    if smps.shape == mask.shape: # ffhq case
        kspace = kspace * mask
    else: # fastMRI case
        kspace = kspace.permute([0, 4, 2, 3, 1]).contiguous()
        kspace = torch.view_as_complex(kspace)
        kspace = kspace * mask
        kspace = torch.view_as_real(kspace).permute([0, 4, 2, 3, 1]).contiguous()
    return kspace

_mask_fn = {
    'uniformly_cartesian': uniformly_cartesian_mask,
    'randomly_cartesian': randomly_cartesian_mask,
    'mix_cartesian': mix_cartesian_mask
}

def get_fastmri_mask(batch_size, image_w_h, mask_pattern, dataset_name, acceleration_rate, cumulative_mask = None):
    assert (mask_pattern in ['randomly_cartesian', 'uniformly_cartesian', 'mix_cartesian', 'random_box']) and (dataset_name in ['ffhq', 'fastmri']) 
    masks = []
    for _ in range(batch_size):
        mask_np = _mask_fn[mask_pattern](img_size=(image_w_h, image_w_h), acceleration_rate=acceleration_rate, cumulative_mask=cumulative_mask)
        masks.append(mask_np)

    masks = np.stack(masks, axis=0)  # Stack along batch dimension
    masks = torch.from_numpy(masks).to(torch.float32).unsqueeze(1)  # Add channel dim
    
    return masks, masks
