import torch
import torch.nn.functional as F
import kornia

def get_otsu_mask(mag_tensor):
    """
    Computes binary mask using Otsu thresholding on the magnitude channel.
    Expects tensor shape (B, C, ...) or (C, ...).
    """
    # Flatten spatial dims for Otsu
    if mag_tensor.ndim == 4: # (B, C, H, W)
        flat = mag_tensor.view(mag_tensor.shape[0], 1, -1)
    elif mag_tensor.ndim == 5: # (B, C, D, H, W)
        flat = mag_tensor.view(mag_tensor.shape[0], 1, -1)
    else:
        flat = mag_tensor.flatten().view(1, 1, -1)

    thresholds = kornia.filters.otsu_threshold(flat)[1]
    
    # Reshape threshold to broadcast over original dimensions
    shape_broad = list(mag_tensor.shape)
    shape_broad[1] = 1 # Channel dim is 1
    for i in range(2, len(shape_broad)): shape_broad[i] = 1
    
    return (mag_tensor >= thresholds.view(shape_broad)).float()

def normalize_min_max_clahe(img_tensor, mask, is_3d=False):
    """
    Applies Min-Max normalization followed by CLAHE.
    """
    # Min-Max
    # Dim params for amin/amax: (-2, -1) for 2D, (-3, -2, -1) for 3D
    dims = (-3, -2, -1) if is_3d else (-2, -1)
    
    img_min = img_tensor.amin(dim=dims, keepdim=True)
    img_max = img_tensor.amax(dim=dims, keepdim=True)
    img_norm = (img_tensor - img_min) / (img_max - img_min + 1e-6)
    
    # CLAHE (Kornia CLAHE expects B,C,H,W)
    if is_3d:
        b, c, d, h, w = img_norm.shape
        # Reshape D into Batch for 2D CLAHE processing
        slices = img_norm.view(b * d, c, h, w)
        processed = kornia.enhance.equalize_clahe(slices)
        processed = processed.view(b, c, d, h, w)
    else:
        processed = kornia.enhance.equalize_clahe(img_norm)
        
    return processed * mask

def normalize_phase(phase_tensor, mask):
    """
    Rescales phase based on 30th and 85th percentiles of the foreground.
    """
    safe_mask = mask.bool()
    if safe_mask.any():
        masked_vals = torch.masked_select(phase_tensor, safe_mask)
        p30 = torch.quantile(masked_vals, 0.30)
        p85 = torch.quantile(masked_vals, 0.85)
    else:
        p30, p85 = 0.0, 1.0
        
    phase_rescaled = (phase_tensor - p30) / (p85 - p30 + 1e-6)
    phase_rescaled = torch.clamp(phase_rescaled, 0, 1)
    return phase_rescaled * mask

def preprocess_gpu(data, is_3d=False, mag_prepro=False, phase_prepro=False):
    """
    Robust GPU Preprocessing for both 2D and 3D.
    
    Args:
        data (torch.Tensor): Input tensor. 
            - 2D expects (C, H, W) or (B, C, H, W)
            - 3D expects (C, D, H, W) or (B, C, D, H, W)
        is_3d (bool): Whether the spatial data is volumetric.
    """
    # 1. STANDARDIZE DIMENSIONS to (B, C, ...)
    was_squeezed = False
    
    expected_ndim = 5 if is_3d else 4
    
    if data.ndim == (expected_ndim - 1):
        # Case: (C, H, W) or (C, D, H, W) -> Missing Batch
        data = data.unsqueeze(0)
        was_squeezed = True
    elif data.ndim != expected_ndim:
        raise ValueError(f"Input shape {data.shape} does not match expected {'3D' if is_3d else '2D'} dimensions.")

    # Now data is strictly (B, C, H, W) or (B, C, D, H, W). 
    # Channel is ALWAYS dim 1.
    mag = data[:, 0:1, ...]
    
    # 2. GENERATE MASK (Otsu + Morphology)
    mask = get_otsu_mask(mag)

    if is_3d:
        # 3D Opening: Erosion then Dilation using MaxPool trick
        # Equivalent to scipy.ndimage.binary_opening but on GPU
        kernel_size = 3
        padding = 1
        # Erosion: -MaxPool(-X)
        eroded = -F.max_pool3d(-mask, kernel_size, stride=1, padding=padding)
        # Dilation: MaxPool(X)
        mask = F.max_pool3d(eroded, kernel_size, stride=1, padding=padding)
    else:
        # 2D Opening: Kornia implementation
        kernel = torch.ones(3, 3, device=data.device)
        mask = kornia.morphology.opening(mask, kernel)

    # 3. PREPROCESS MAGNITUDE
    if mag_prepro:
        proc_mag = normalize_min_max_clahe(mag, mask, is_3d=is_3d)
    else:
        proc_mag = mag * mask

    # 4. PREPROCESS PHASE (if Channel 1 exists)
    if data.shape[1] > 1:
        phase = data[:, 1:2, ...]
        if phase_prepro:
            proc_phase = normalize_phase(phase, mask)
        else:
            proc_phase = phase * mask
        
        # Re-concatenate channels
        result = torch.cat([proc_mag, proc_phase], dim=1)
    else:
        result = proc_mag

    # 5. RESTORE DIMENSIONS
    if was_squeezed:
        result = result.squeeze(0)
        
    return result