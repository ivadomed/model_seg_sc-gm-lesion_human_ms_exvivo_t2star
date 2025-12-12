import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import kornia

# --- 1. Environment Setup ---
# Saving in a new subfolder as requested
OUTPUT_DIR = "./visuals/phase_preprocessing"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_and_prep_slice(path):
    """
    Loads a NIfTI, extracts the middle slice, and normalizes it 
    to [0, 1] for stable Kornia processing.
    """
    nii = nib.load(path)
    data = nii.get_fdata()
    
    # Extract middle slice (Assumes 3D volume or 4D with 1 channel)
    if data.ndim == 3:
        mid = data.shape[2] // 2
        slice_data = data[:, :, mid]
    elif data.ndim == 4:
        mid = data.shape[2] // 2
        slice_data = data[:, :, mid, 0] 
    else:
        slice_data = data

    # Min-Max Normalize to [0, 1] 
    # (Standardizes input before specific pipeline logic)
    d_min = np.min(slice_data)
    d_max = np.max(slice_data)
    if d_max - d_min > 1e-8:
        slice_data = (slice_data - d_min) / (d_max - d_min)
    else:
        slice_data = np.zeros_like(slice_data)
        
    # Convert to Tensor: (B, C, H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(slice_data).float().unsqueeze(0).unsqueeze(0)
    return tensor, os.path.basename(path)

def run_phase_prepro(mag_path, phase_path):
    print(f"Processing Pair:\n  Mag: {mag_path}\n  Phase: {phase_path}")
    
    # 1. Load Data
    mag_tensor, mag_name = load_and_prep_slice(mag_path)
    phase_tensor, phase_name = load_and_prep_slice(phase_path)
    
    # --- 2. PIPELINE LOGIC (Replicating _preprocess_data_gpu) ---
    
    # A. Create Mask from MAGNITUDE
    # Otsu Thresholding (Taking index 1 as requested)
    thresholds_tensor = kornia.filters.otsu_threshold(mag_tensor)[1]
    
    reshaped_thresholds = thresholds_tensor.view(-1, 1, 1, 1)
    mask = (mag_tensor >= reshaped_thresholds).float()

    # Morphological Opening
    kernel = torch.ones(3, 3) 
    mask = kornia.morphology.opening(mask, kernel)
    
    # B. Process PHASE
    # Logic extracted from the provided trainer code:
    safe_mask = mask.bool()
    
    if safe_mask.any():
        # Select only phase values falling inside the mask
        masked_phase_values = torch.masked_select(phase_tensor, safe_mask)
        
        # Calculate percentiles dynamically based on the masked region
        p30 = torch.quantile(masked_phase_values, 0.30)
        p85 = torch.quantile(masked_phase_values, 0.85)
    else:
        # Fallback if mask is empty
        p30, p85 = 0.0, 1.0

    # Rescale phase intensity based on these percentiles
    # (phase - p30) / (p85 - p30 + epsilon)
    phase_rescaled = (phase_tensor - p30) / (p85 - p30 + 1e-6)
    
    # Clamp values to [0, 1]
    phase_rescaled = torch.clamp(phase_rescaled, 0, 1)
    
    # Apply the mask to the final result
    processed_phase = phase_rescaled * mask
    
    # -----------------------------------------------------------

    # 3. Convert back to numpy for plotting
    img_phase_orig = phase_tensor.squeeze().numpy()
    img_phase_proc = processed_phase.squeeze().numpy()
    img_mask = mask.squeeze().numpy()

    # 4. Visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original Phase
    axs[0].imshow(img_phase_orig, cmap='gray')
    axs[0].set_title(f"\n ")
    axs[0].axis('off')

    # Processed Phase
    axs[1].imshow(img_phase_proc, cmap='gray')
    axs[1].set_title("\n ")
    axs[1].axis('off')
    
    plt.suptitle(f"Phase Preprocessing Steps", fontsize=16)
    plt.tight_layout()
    
    # 5. Save with specific suffix
    base_name = phase_name
    if base_name.endswith('.nii.gz'):
        base_name = base_name[:-7]
    elif base_name.endswith('.nii'):
        base_name = base_name[:-4]
        
    suffix = "___sub-PML019_S1_71.0cm_T2s_75i_TR45TE9_cor_18avg_900_slice-157"
    save_filename = f"{base_name}{suffix}.png"
    
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path)
    print(f"Saved visualization to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    mag_file = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/imagesTr/MagPhaseExp_simple_training_base_0000_0000.nii.gz"
    phase_file = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/imagesTr/MagPhaseExp_simple_training_base_0000_0001.nii.gz"
    
    # --- Dummy Data Generator ---
    if not os.path.exists(mag_file) or not os.path.exists(phase_file):
        print("Input files not found. Generating dummy NIfTI...")
        os.makedirs("./data", exist_ok=True)
        
        # Create Dummy Mag (for mask generation)
        grid_x, grid_y = np.mgrid[0:256, 0:256]
        center = (128, 128)
        mask_blob = (np.sqrt((grid_x - center[0])**2 + (grid_y - center[1])**2) < 90).astype(float)
        dummy_mag = (mask_blob * 0.8) + (np.random.rand(256, 256) * 0.1)
        
        # Create Dummy Phase (Gradient + Noise)
        dummy_phase = (grid_x / 256.0) + (np.random.rand(256, 256) * 0.2)
        
        dummy_mag = np.stack([dummy_mag]*10, axis=2) 
        dummy_phase = np.stack([dummy_phase]*10, axis=2)

        nib.save(nib.Nifti1Image(dummy_mag, np.eye(4)), mag_file)
        nib.save(nib.Nifti1Image(dummy_phase, np.eye(4)), phase_file)

    run_phase_prepro(mag_file, phase_file)