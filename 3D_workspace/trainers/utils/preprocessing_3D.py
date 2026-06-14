import torch
import torch.nn.functional as F
import kornia


def preprocess_data_gpu_3d(data_batch: torch.Tensor, mag_prepro: bool = False, phase_prepro: bool = False) -> torch.Tensor:
    was_5dim = False
    if len(data_batch.size()) == 4: 
        was_5dim = True
        data_batch = data_batch.unsqueeze(0)
    
    b, c, d, h, w = data_batch.shape
    mag_batch = data_batch[:, 0:1, :, :, :] 
    
    # 1. 3D Otsu
    mag_flattened = mag_batch.view(b, 1, d * h, w)
    thresholds_tensor = kornia.filters.otsu_threshold(mag_flattened)[1]
    reshaped_thresholds = thresholds_tensor.view(b, 1, 1, 1, 1)
    mask = (mag_batch >= reshaped_thresholds).float()

    # 2. 3D Morphology
    kernel_size = 3
    padding = 1
    eroded_mask = -F.max_pool3d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
    opened_mask = F.max_pool3d(eroded_mask, kernel_size=kernel_size, stride=1, padding=padding)
    mask = opened_mask

    processed_mag = mag_batch * mask
    processed_phase = None
    if c > 1:
        phase_batch = data_batch[:, 1:2, :, :, :]
        processed_phase = phase_batch * mask

    # 3. Magnitude Preprocessing
    if mag_prepro:
        mag_min = processed_mag.amin(dim=(-3, -2, -1), keepdim=True)
        mag_max = processed_mag.amax(dim=(-3, -2, -1), keepdim=True)
        mag_normalized = (processed_mag - mag_min) / (mag_max - mag_min + 1e-6)
        
        mag_slices = mag_normalized.view(b * d, 1, h, w)
        mag_clahe_slices = kornia.enhance.equalize_clahe(mag_slices)
        mag_clahe = mag_clahe_slices.view(b, 1, d, h, w)
        processed_mag = mag_clahe * mask

    # 4. Phase Preprocessing
    if phase_prepro and c > 1:
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
    
    if c > 1:
        processed_batch = torch.cat([processed_mag, processed_phase], dim=1)
    else:
        processed_batch = processed_mag
        
    if was_5dim: processed_batch = processed_batch.squeeze(0)
    return processed_batch