import os
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import kornia

# --- 1. Environment Setup ---
OUTPUT_DIR = "./visuals/otsu_masking" # Keeping the directory you requested
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
        slice_data = data[:, :, mid, 0] # Take first channel if 4D
    else:
        slice_data = data

    # Min-Max Normalize to [0, 1] (Essential for Neural Nets/Otsu stability)
    d_min = np.min(slice_data)
    d_max = np.max(slice_data)
    if d_max - d_min > 1e-8:
        slice_data = (slice_data - d_min) / (d_max - d_min)
    else:
        slice_data = np.zeros_like(slice_data)
        
    # Convert to Tensor: (B, C, H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(slice_data).float().unsqueeze(0).unsqueeze(0)
    return tensor, os.path.basename(path)

def run_otsu_masking(mag_path, phase_path):
    print(f"Processing Pair:\n  Mag: {mag_path}\n  Phase: {phase_path}")
    
    # 1. Load Data
    mag_tensor, mag_name = load_and_prep_slice(mag_path)
    phase_tensor, phase_name = load_and_prep_slice(phase_path)

    # 2. Apply Otsu Logic (Replicating _preprocess_data_gpu)
    # ---------------------------------------------------------
    # Note: Kornia operations expect (B, C, H, W)
    
    # A. Calculate Threshold on MAGNITUDE
    # kornia.filters.otsu_threshold returns the computed threshold values
    otsu_thresh = kornia.filters.otsu_threshold(mag_tensor)[1]
    # B. Create Binary Mask
    # (mag_batch >= reshaped_thresholds).float()
    mask = (mag_tensor >= otsu_thresh).float()

    # C. Apply Morphological Opening
    # kernel = torch.ones(3, 3)
    kernel = torch.ones(3, 3)
    mask = kornia.morphology.opening(mask, kernel)

    # D. Apply Mask to Both Channels
    processed_mag = mag_tensor * mask
    processed_phase = phase_tensor * mask
    # ---------------------------------------------------------

    # 3. Convert back to numpy for plotting
    img_mag_orig = mag_tensor.squeeze().numpy()
    img_phase_orig = phase_tensor.squeeze().numpy()
    img_mag_proc = processed_mag.squeeze().numpy()
    img_phase_proc = processed_phase.squeeze().numpy()
    img_mask = mask.squeeze().numpy()

    # 4. Visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    
    # Row 1: Originals
    axs[0, 0].imshow(img_mag_orig, cmap='gray')
    axs[0, 0].set_title(f"\n ")
    axs[0, 0].axis('off')

    axs[0, 1].imshow(img_phase_orig, cmap='gray')
    axs[0, 1].set_title(f"\n ")
    axs[0, 1].axis('off')

    # Row 2: Masked Results
    axs[1, 0].imshow(img_mag_proc, cmap='gray')
    axs[1, 0].set_title("\n ")
    axs[1, 0].axis('off')

    axs[1, 1].imshow(img_phase_proc, cmap='gray')
    axs[1, 1].set_title("\n ")
    axs[1, 1].axis('off')
    
    plt.suptitle(f"Otsu Preprocessing Visualization", fontsize=16)
    plt.tight_layout()
    
    save_filename = f"otsu_{mag_name.replace('.nii.gz', '')}.png"
    save_path = os.path.join(OUTPUT_DIR, save_filename[:-4] + "___sub-PML019_S1_71.0cm_T2s_75i_TR45TE9_cor_18avg_900_slice-157" + ".png")
    plt.savefig(save_path)
    print(f"Saved visualization to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    mag_file = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/imagesTr/MagPhaseExp_simple_training_base_0000_0000.nii.gz"
    phase_file = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/imagesTr/MagPhaseExp_simple_training_base_0000_0001.nii.gz"
    
    # --- Dummy Data Generator (For testing if files don't exist) ---
    if not os.path.exists(mag_file) or not os.path.exists(phase_file):
        print("Input files not found. Generating dummy NIfTI files for demonstration...")
        os.makedirs("./data", exist_ok=True)
        
        # Create Dummy Mag (Blob in center)
        grid_x, grid_y = np.mgrid[0:256, 0:256]
        center = (128, 128)
        mask_blob = (np.sqrt((grid_x - center[0])**2 + (grid_y - center[1])**2) < 80).astype(float)
        noise = np.random.rand(256, 256) * 0.2
        dummy_mag = (mask_blob * 0.8) + noise # Signal + Noise
        dummy_mag = np.stack([dummy_mag]*10, axis=2) # Make 3D
        
        # Create Dummy Phase (Random noise everywhere)
        dummy_phase = np.random.rand(256, 256, 10) 

        nib.save(nib.Nifti1Image(dummy_mag, np.eye(4)), mag_file)
        nib.save(nib.Nifti1Image(dummy_phase, np.eye(4)), phase_file)
        print("Dummy files created.")

    run_otsu_masking(mag_file, phase_file)