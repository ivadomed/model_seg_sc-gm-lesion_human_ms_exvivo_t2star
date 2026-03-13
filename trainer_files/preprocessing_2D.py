
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
import numpy as np
import torch 

def _preprocess_data_gpu(data_batch: torch.Tensor, mag_prepro: bool = False, phase_prepro: bool = False) -> torch.Tensor:
    """
    Applies custom preprocessing DIRECTLY ON THE GPU using Kornia and PyTorch.
    Modified to handle single-channel (magnitude only) input.
    """
    # data_batch is already a GPU tensor, no need for more .to(device) calls here
    was_3dim = False
    if len(data_batch.size())==3 : ## the batch dim is missing
        was_3dim = True
        data_batch = data_batch.unsqueeze(0)
    
    # Check number of channels
    num_channels = data_batch.size(1)

    mag_batch = data_batch[:, 0:1, :, :]
    
    # 1. Create binary mask from magnitude
    thresholds_tensor = kornia.filters.otsu_threshold(mag_batch)[1]
    reshaped_thresholds = thresholds_tensor.view(-1, 1, 1, 1)
    mask = (mag_batch >= reshaped_thresholds).float()

    # 2. Apply morphological opening
    kernel = torch.ones(3, 3, device=mask.device) # Use mask.device for robustness
    mask = kornia.morphology.opening(mask, kernel)

    processed_mag = mag_batch * mask
    
    # Initialize processed_phase to None
    processed_phase = None

    if num_channels > 1:
        phase_batch = data_batch[:, 1:2, :, :]
        processed_phase = phase_batch * mask

    if mag_prepro:
        mag_min = processed_mag.amin(dim=(-2, -1), keepdim=True)
        mag_max = processed_mag.amax(dim=(-2, -1), keepdim=True)
        mag_normalized = (processed_mag - mag_min) / (mag_max - mag_min + 1e-6)
        mag_clahe = kornia.enhance.equalize_clahe(mag_normalized)
        processed_mag = mag_clahe * mask

    if phase_prepro and num_channels > 1:
        safe_mask = mask.bool()
        if safe_mask.any():
            masked_phase_values = torch.masked_select(phase_batch, safe_mask)
            p30 = torch.quantile(masked_phase_values, 0.30)
            p85 = torch.quantile(masked_phase_values, 0.85)
        else:
            p30, p85 = 0.0, 1.0

        phase_rescaled = (phase_batch - p30) / (p85 - p30 + 1e-6)
        phase_rescaled = torch.clamp(phase_rescaled, 0, 1)
        processed_phase = phase_rescaled * mask
    
    # 5. Combine channels
    if num_channels > 1:
        processed_batch = torch.cat([processed_mag, processed_phase], dim=1)
    else:
        processed_batch = processed_mag
        
    if was_3dim : 
        processed_batch = processed_batch.squeeze(0)
    return processed_batch


class GPUPreprocessingTransform(BasicTransform):
    def __init__(self, device: torch.device = torch.device('cuda'), mag_prepro: bool = False, phase_prepro: bool = False, no_otsu: bool = False):
        super(GPUPreprocessingTransform, self).__init__()
        self.device = device
        self.mag_prepro = mag_prepro
        self.phase_prepro = phase_prepro
        self.no_otsu = no_otsu
        
    def __call__(self, **data_dict):
        # Unique log file for each worker process
        data_for_gpu = data_dict['image'].clone()
        
        if self.no_otsu : 
            processed_data = data_for_gpu
        else :
            processed_data = _preprocess_data_gpu(data_for_gpu, self.mag_prepro, self.phase_prepro)
        data_dict['image'] = processed_data.cpu()

        return data_dict