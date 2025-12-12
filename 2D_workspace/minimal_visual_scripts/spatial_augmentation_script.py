import os
import random
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import kornia
from typing import Tuple

# --- 1. Mocks and Environment Setup ---
# Create the output directory
OUTPUT_DIR = "./visuals/spatial_augmentation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set seeds for deterministic behavior (Removing randomness as requested)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Mock BasicTransform so we don't need the batchgeneratorsv2 library dependency
class BasicTransform:
    def __call__(self, **data_dict):
        return data_dict

# --- 2. The Custom Transform (Extracted from your Trainer) ---
class CustomSpatialTransform(BasicTransform):
    """
    Extracted from your nnUNetTrainer. 
    """
    def __init__(self,
                 patch_size: Tuple[int, ...],
                 p_per_sample: float = 1.0,
                 degrees: Tuple[float, float] = (-15, 15),
                 scale: Tuple[float, float] = (0.6, 1.4),
                 translation_px: Tuple[int, ...] = (20, 20),
                 shear: Tuple[float, float] = (-10, 10),
                 perspective: float = 0.1,
                 data_key: str = "image",
                 label_key: str = "segmentation"):
        super(CustomSpatialTransform, self).__init__()
        self.patch_size = patch_size
        self.p_per_sample = p_per_sample
        
        # Note: resample is set to bilinear for image
        self.affine_augmenter_image = kornia.augmentation.RandomAffine(
            degrees=degrees, translate=tuple(float(t) / p for t, p in zip(translation_px, patch_size)),
            scale=scale, shear=shear, p=1.0, resample='bilinear'
        )
        self.perspective_augmenter_image = kornia.augmentation.RandomPerspective(
            distortion_scale=perspective, p=1.0, resample='bilinear'
        )

        self.data_key = data_key
        self.label_key = label_key

    def __call__(self, **data_dict):
        # Forced execution if p_per_sample is 1.0
        if random.random() > self.p_per_sample:
            return data_dict

        data = data_dict.get(self.data_key)
        
        # --- 1. Random Crop to Patch Size ---
        # In a visualization script, if image < patch_size, we might error out.
        # Logic here handles resizing or cropping.
        data_shape = data.shape[1:] # C, H, W -> H, W
        
        # Safety check for script: if image is smaller than patch, pad it locally
        if any(i < j for i, j in zip(data_shape, self.patch_size)):
             pad_h = max(0, self.patch_size[0] - data_shape[0])
             pad_w = max(0, self.patch_size[1] - data_shape[1])
             data = torch.nn.functional.pad(data, (0, pad_w, 0, pad_h))
             data_shape = data.shape[1:]

        starts = [random.randint(0, i - j) for i, j in zip(data_shape, self.patch_size)]
        slicing = [slice(None)] + [slice(s, s + p) for s, p in zip(starts, self.patch_size)]
        cropped_data = data[tuple(slicing)].clone()
        
        # --- 2. Generate Parameters ONCE ---
        # Generate random parameters using the image augmenter
        # (Even though it's "Random", seeding makes it deterministic)
        params_affine = self.affine_augmenter_image.generate_parameters(cropped_data.unsqueeze(0).shape)
        
        # --- 3. Apply Transformations to Image ---
        # Apply affine transform to image
        data_after_affine = self.affine_augmenter_image(cropped_data.unsqueeze(0), params=params_affine)
        
        # Generate perspective params and apply transform to image
        params_perspective = self.perspective_augmenter_image.generate_parameters(data_after_affine.shape)
        final_data = self.perspective_augmenter_image(data_after_affine, params=params_perspective).squeeze(0)
        
        data_dict[self.data_key] = final_data
        return data_dict

# --- 3. Main Execution and Visualization ---
def run_augmentation(nii_path):
    filename = os.path.basename(nii_path)
    print(f"Processing: {filename}")
    
    # 1. Load Image
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    
    # 2. Extract a middle slice to simulate 2D
    # Assumption: data is (H, W, D) -> take middle D
    if data.ndim == 3:
        mid_slice_idx = data.shape[2] // 2
        slice_data = data[:, :, mid_slice_idx]
    elif data.ndim == 4: # (H, W, D, C) or (C, H, W, D) depending on loading
        # Simple heuristic, take middle of last dim
        mid_slice_idx = data.shape[-1] // 2
        slice_data = data[..., mid_slice_idx]
        if slice_data.ndim == 3: slice_data = slice_data[:, :, 0] # reduce to 2D
    else:
        slice_data = data # Assume 2D

    # 3. Normalize to [0, 1] (Standard for Neural Nets/Kornia)
    # This prevents values from exploding or clipping weirdly during interpolation
    slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-8)
    
    # 4. Prepare Tensor: Shape (C, H, W). Kornia expects Float tensors.
    input_tensor = torch.from_numpy(slice_data).float().unsqueeze(0) # Add channel dim -> (1, H, W)
    
    # Define Patch Size (Simulating what nnU-Net would do)
    # We use the actual image size or a fixed size like 512x512
    patch_size = (input_tensor.shape[1], input_tensor.shape[2])
    
    # Calculate translation based on patch size (as per your trainer code)
    max_translate_dist = [int(p * 0.45) for p in patch_size]

    # 5. Initialize Transform
    # p_per_sample=1.0 forces the transform to happen
    transform = CustomSpatialTransform(
        patch_size=patch_size,
        p_per_sample=1.0,     
        degrees=(-90, 90),     
        scale=(0.7, 1.7),     
        translation_px=tuple(max_translate_dist),
        shear=(-35, 35),       
        perspective=0.35      
    )

    # 6. Apply Transform
    data_dict = {'image': input_tensor, 'segmentation': None}
    
    # Create copies for visualization
    original_img = input_tensor.clone().squeeze().numpy()
    
    print("Applying augmentation...")
    out_dict = transform(**data_dict)
    
    aug_img = out_dict['image'].squeeze().numpy()

    # 7. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    ax[0].imshow(original_img, cmap='gray')
    ax[0].set_title("Original Slice")
    ax[0].axis('off')
    
    ax[1].imshow(aug_img, cmap='gray')
    ax[1].set_title("Processed (Spatial Aug)")
    ax[1].axis('off')
    
    # Add Text info
    plt.suptitle(f"File: {filename}\nPath: {nii_path}", fontsize=10)
    
    save_path = os.path.join(OUTPUT_DIR, f"aug_{filename.replace('.nii.gz', '')}.png")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    target_nii_path = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/imagesTr/MagPhaseExp_simple_training_base_0000_0000.nii.gz" 

    # Create a dummy file if it doesn't exist for testing purposes
    if not os.path.exists(target_nii_path):
        print(f"File not found at {target_nii_path}. Creating dummy noise image.")
        dummy_data = np.random.rand(256, 256, 10)
        img = nib.Nifti1Image(dummy_data, np.eye(4))
        target_nii_path = "dummy_test.nii.gz"
        nib.save(img, target_nii_path)

    run_augmentation(target_nii_path)