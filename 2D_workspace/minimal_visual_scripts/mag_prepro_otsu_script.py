import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import kornia

# --- 1. Environment Setup ---
# Saving in a new subfolder as requested
OUTPUT_DIR = "./visuals/mag_preprocessing"
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

    # Min-Max Normalize to [0, 1] (Standard for Neural Nets input)
    # This prepares the raw data before the specific pipeline normalization
    d_min = np.min(slice_data)
    d_max = np.max(slice_data)
    if d_max - d_min > 1e-8:
        slice_data = (slice_data - d_min) / (d_max - d_min)
    else:
        slice_data = np.zeros_like(slice_data)
        
    # Convert to Tensor: (B, C, H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(slice_data).float().unsqueeze(0).unsqueeze(0)
    return tensor, os.path.basename(path)

def run_mag_prepro(mag_path):
    print(f"Processing Magnitude: {mag_path}")
    
    # 1. Load Data
    mag_tensor, mag_name = load_and_prep_slice(mag_path)
    
    # --- 2. PIPELINE LOGIC (Replicating _preprocess_data_gpu) ---
    # We assume data is already (1, 1, H, W)
    
    # A. Otsu Thresholding
    # User modification: take element of index 1
    thresholds_tensor = kornia.filters.otsu_threshold(mag_tensor)[1]
    
    reshaped_thresholds = thresholds_tensor.view(-1, 1, 1, 1)
    mask = (mag_tensor >= reshaped_thresholds).float()

    # B. Morphological Opening
    kernel = torch.ones(3, 3) 
    mask = kornia.morphology.opening(mask, kernel)

    # C. Apply Mask (Intermediate)
    processed_mag = mag_tensor * mask
    
    # D. Magnitude Preprocessing (CLAHE)
    # 1. Min/Max normalization of the MASKED image
    mag_min = processed_mag.amin(dim=(-2, -1), keepdim=True)
    mag_max = processed_mag.amax(dim=(-2, -1), keepdim=True)
    
    # Avoid division by zero with 1e-6
    mag_normalized = (processed_mag - mag_min) / (mag_max - mag_min + 1e-6)
    
    # 2. Apply CLAHE
    # Note: Kornia defaults clip_limit=40, grid_size=(8,8) usually
    mag_clahe = kornia.enhance.equalize_clahe(mag_normalized)
    
    # 3. Final Masking
    final_processed_mag = mag_clahe * mask
    
    # -----------------------------------------------------------

    # 3. Convert back to numpy for plotting
    img_orig = mag_tensor.squeeze().numpy()
    img_proc = final_processed_mag.squeeze().numpy()
    img_mask = mask.squeeze().numpy()

    # 4. Visualization
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original
    axs[0].imshow(img_orig, cmap='gray')
    axs[0].set_title(f"\n ")
    axs[0].axis('off')

    # Processed (Otsu + CLAHE)
    axs[1].imshow(img_proc, cmap='gray')
    axs[1].set_title("\n ")
    axs[1].axis('off')
    
    plt.suptitle(f"Magnitude Preprocessing Steps", fontsize=16)
    plt.tight_layout()
    
    # 5. Save with specific suffix
    base_name = mag_name
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
    # REPLACE WITH YOUR ACTUAL MAGNITUDE PATH
    mag_file = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/imagesTr/MagPhaseExp_simple_training_base_0000_0000.nii.gz"
    
    # --- Dummy Data Generator ---
    if not os.path.exists(mag_file):
        print("Input file not found. Generating dummy NIfTI...")
        os.makedirs("./data", exist_ok=True)
        
        # Create Dummy Mag with some contrast to show off CLAHE
        grid_x, grid_y = np.mgrid[0:256, 0:256]
        center = (128, 128)
        mask_blob = (np.sqrt((grid_x - center[0])**2 + (grid_y - center[1])**2) < 90).astype(float)
        
        # Gradient inside the blob
        gradient = (grid_x + grid_y) / 512.0
        dummy_mag = (mask_blob * gradient) + (np.random.rand(256, 256) * 0.05)
        
        dummy_mag = np.stack([dummy_mag]*10, axis=2) # 3D
        nib.save(nib.Nifti1Image(dummy_mag, np.eye(4)), mag_file)

    run_mag_prepro(mag_file)