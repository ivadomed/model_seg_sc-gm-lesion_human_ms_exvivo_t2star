#!/usr/bin/env python

"""
Standalone Python Script for Testing MRI Preprocessing Logic from a nnU-Net Trainer.

This script automatically finds multiple magnitude-phase NIfTI pairs, applies the
custom preprocessing function to each, and saves a visualization of the results
to a unique PNG file for each pair.
"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.restoration import unwrap_phase
from skimage.exposure import equalize_adapthist as clahe, rescale_intensity
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, binary_opening

# =====================================================================================
# 1. MOCK nnU-Net BASE CLASSES (No changes)
# =====================================================================================

class MockLabelManager:
    def __init__(self):
        self.foreground_regions, self.has_regions = [], False
        self.foreground_labels, self.ignore_label = (1, 2, 3), None

class MockConfigurationManager:
    def __init__(self):
        self.patch_size, self.use_mask_for_norm = (128, 128), [False, False]

class nnUNetTrainer:
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device):
        self.plans, self.configuration, self.fold = plans, configuration, fold
        self.dataset_json, self.device = dataset_json, device
        self.label_manager = MockLabelManager()
        self.configuration_manager = MockConfigurationManager()
        self.is_cascaded = False

# =====================================================================================
# 2. CUSTOM TRAINER CLASS WITH PREPROCESSING LOGIC (No changes)
# =====================================================================================

class nnUNetTrainerWandb(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cpu')):
        super().__init__(plans, configuration, fold, dataset_json, device)

    def _preprocess_data(self, data_batch: torch.Tensor) -> torch.Tensor:
        batch_np = data_batch.cpu().numpy()
        processed_batch_np = np.zeros_like(batch_np)

        for i in range(batch_np.shape[0]):
            mag_channel, phase_channel = batch_np[i, 0], batch_np[i, 1]
            try:
                thresh = threshold_otsu(mag_channel[mag_channel > 0])
                mask = mag_channel > thresh
                mask = binary_fill_holes(mask)
                mask = binary_opening(mask, structure=np.ones((3, 3)))
            except ValueError:
                mask = np.ones_like(mag_channel, dtype=bool)

            vmin, vmax = mag_channel.min(), mag_channel.max()
            mag_normalized = (mag_channel - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(mag_channel)
            mag_clahe = clahe(mag_normalized, clip_limit=0.01)
            processed_batch_np[i, 0] = mag_clahe * mask

            unwrapped_phase = unwrap_phase(phase_channel)
            masked_unwrapped_phase = unwrapped_phase[mask]
            
            p2, p98 = np.percentile(masked_unwrapped_phase, (30, 85)) if masked_unwrapped_phase.size > 0 else (0, 0)
            phase_rescaled = rescale_intensity(unwrapped_phase, in_range=(p2, p98), out_range=(0, 1)) if p98 > p2 else np.zeros_like(unwrapped_phase)
            processed_batch_np[i, 1] = phase_rescaled * mask
        return torch.from_numpy(processed_batch_np).float()

# =====================================================================================
# 3. HELPER FUNCTIONS
# =====================================================================================

def load_and_combine_channels(mag_path: str, phase_path: str) -> torch.Tensor:
    """Loads separate magnitude and phase NIfTI files and combines them into a 2-channel tensor."""
    if not os.path.exists(phase_path):
        print(f"Warning: Corresponding phase file not found for {os.path.basename(mag_path)}. Skipping.")
        return None
        
    mag_img = nib.load(mag_path)
    mag_data = mag_img.get_fdata(dtype=np.float32)

    phase_img = nib.load(phase_path)
    phase_data = phase_img.get_fdata(dtype=np.float32)

    if mag_data.shape != phase_data.shape:
        print(f"Warning: Shape mismatch for {os.path.basename(mag_path)}. Skipping.")
        return None
    
    combined_data = np.stack([mag_data, phase_data], axis=-1)
    combined_data = np.transpose(combined_data, (2, 0, 1))
    data_batch = torch.from_numpy(combined_data).unsqueeze(0)
    
    return data_batch

def create_and_save_plot(original_batch, processed_batch, output_filename):
    """Generates and saves the 2x2 comparison plot."""
    original_mag, original_phase = original_batch[0, 0].numpy(), original_batch[0, 1].numpy()
    processed_mag, processed_phase = processed_batch[0, 0].numpy(), processed_batch[0, 1].numpy()

    fig, axs = plt.subplots(2, 2, figsize=(12, 12), constrained_layout=True)
    fig.suptitle(f'Preprocessing Output for\n{os.path.basename(output_filename)}', fontsize=16, weight='bold')

    # Plot Original Magnitude
    im1 = axs[0, 0].imshow(original_mag.T, cmap='gray')
    axs[0, 0].set_title('Original Magnitude', fontsize=14); axs[0, 0].axis('off')
    fig.colorbar(im1, ax=axs[0, 0], shrink=0.8)

    # Plot Original Phase
    im2 = axs[0, 1].imshow(original_phase.T, cmap='gray')
    axs[0, 1].set_title('Original Phase', fontsize=14); axs[0, 1].axis('off')
    fig.colorbar(im2, ax=axs[0, 1], shrink=0.8)

    # Plot Preprocessed Magnitude
    im3 = axs[1, 0].imshow(processed_mag.T, cmap='gray')
    axs[1, 0].set_title('Processed Magnitude (Masked + CLAHE)', fontsize=14); axs[1, 0].axis('off')
    fig.colorbar(im3, ax=axs[1, 0], shrink=0.8)

    # Plot Preprocessed Phase
    im4 = axs[1, 1].imshow(processed_phase.T, cmap='gray')
    axs[1, 1].set_title('Processed Phase (Unwrapped + Rescaled)', fontsize=14); axs[1, 1].axis('off')
    fig.colorbar(im4, ax=axs[1, 1], shrink=0.8)

    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"✅ Saved output image: {output_filename}")


# =====================================================================================
# 4. MAIN EXECUTION BLOCK (*** This section is updated ***)
# =====================================================================================

def main():
    """Main execution function to find and process multiple files."""
    
    # --- Configuration ---
    base_data_path = '../../datasets/processed_data_full_multichannel/data_slice'
    # Change this number to generate more or fewer images
    num_images_to_generate = 5
    
    # --- Find Files ---
    subject_anat_path = os.path.join(base_data_path, 'sub-PML014/anat')
    # Use glob to find all magnitude files
    search_pattern = os.path.join(subject_anat_path, '*_part-mag_*.nii.gz')
    magnitude_files = sorted(glob.glob(search_pattern))

    if not magnitude_files:
        print(f"Error: No magnitude files found at '{search_pattern}'. Please check the path.")
        return

    print(f"Found {len(magnitude_files)} magnitude files. Processing the first {num_images_to_generate}...")
    
    # --- Processing Loop ---
    trainer = nnUNetTrainerWandb(plans={}, configuration='2d', fold=0, dataset_json={}, device=torch.device('cpu'))

    for mag_path in magnitude_files[:num_images_to_generate]:
        print("-" * 50)
        # Derive the corresponding phase file path
        phase_path = mag_path.replace('_part-mag_', '_part-phase_')
        
        # Load the data
        original_batch = load_and_combine_channels(mag_path, phase_path)
        
        # If loading was successful, process and save the plot
        if original_batch is not None:
            # Apply the preprocessing
            processed_batch = trainer._preprocess_data(original_batch)
            
            # Define a unique output filename
            output_filename = f"preprocessing_output_{os.path.basename(mag_path).replace('.nii.gz', '.png')}"
            
            # Create and save the visualization
            create_and_save_plot(original_batch, processed_batch, output_filename)

if __name__ == "__main__":
    main()