import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import GaussianBlur
import nibabel as nib
import matplotlib.pyplot as plt

# --- 1. Environment Setup ---
OUTPUT_DIR = "./visuals/soft_loss_edges"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. Configuration (From Trainer) ---
# Exact parameters from your trainer code
EDGE_PARAMS = {
    0: {'edge_weight': 0.9, 'kernel_size': 7},
    1: {'edge_weight': 0.9, 'kernel_size': 7},
    2: {'edge_weight': 0.8, 'kernel_size': 5},  
    3: {'edge_weight': 0.7, 'kernel_size': 5},
    4: {'edge_weight': 0.5, 'kernel_size': 5}
}
BLUR_SIGMA = 1.0

# --- 3. The Logic extracted from DC_and_CE_with_Edge_Loss ---

def create_weight_map(target_one_hot: torch.Tensor, edge_params: dict, blur_sigma: float) -> torch.Tensor:
    """
    Standalone version of the _create_weight_map method from DC_and_CE_with_Edge_Loss.
    
    Expected Input Shape: (Num_Classes, Batch_Size, 1, H, W) 
    (This mimics the shape inside the loss function of nnUNet)
    """
    shape = target_one_hot.shape 
    # Logic from provided code to handle 2D vs 3D
    is_3d = (len(shape) == 5) and (shape[2] > 1) # Generally False for 2D slices

    if is_3d:
        pool_op = F.max_pool3d
        weight_map_shape = (shape[0], 1, shape[2], shape[3], shape[4])
    else:
        pool_op = F.max_pool2d
        
        # Squeeze the depth dim if it exists (standard nnUnet behavior)
        squeezed_target = target_one_hot
        if len(shape) == 5:
            squeezed_target = target_one_hot.squeeze(2)
        
        current_shape = squeezed_target.shape
        # current_shape is now (Num_Classes, Batch, H, W)
        
        batch_size = current_shape[1]
        spatial_dims = current_shape[2:] 
        weight_map_shape = (batch_size, 1, *spatial_dims) 

    # Initialize with 1.0 (default weight)
    weight_map = torch.ones(weight_map_shape, device=target_one_hot.device, dtype=torch.float32)
    
    for class_idx, params in edge_params.items():
        # Safety check if labels in params exceed actual classes in tensor
        if class_idx >= shape[0]: 
            continue            
        
        k = params.get('kernel_size', 3)
        w_edge = params.get('edge_weight', 1.0)
        pad = (k - 1) // 2 if k > 1 else 0
        
        # Extract specific class mask: (1, Batch, H, W)
        class_mask_sliced = squeezed_target[class_idx:class_idx+1, ...].float()
        
        # Permute for pooling: (Batch, 1, H, W)
        class_mask_for_pooling = class_mask_sliced.permute(1, 0, 2, 3)
        
        # Morphological Gradient (Dilate - Erode)
        dilated = pool_op(class_mask_for_pooling, kernel_size=k, stride=1, padding=pad)
        eroded = -pool_op(-class_mask_for_pooling, kernel_size=k, stride=1, padding=pad)
        
        edge_region = (dilated - eroded).bool()
        
        # Update weight map
        # Note: edge_region is (Batch, 1, H, W), weight_map is (Batch, 1, H, W)
        weight_map = torch.where(edge_region, torch.tensor(w_edge), weight_map)

    # Re-expand dimensions to match typical nnUNet flow if needed
    if len(shape) == 5 and not is_3d:
        weight_map = weight_map.unsqueeze(2) 

    # Apply Gaussian Blur
    if blur_sigma > 0:
        kernel_size = int(2 * round(3 * blur_sigma) + 1)
        blurrer = GaussianBlur(kernel_size=kernel_size, sigma=blur_sigma)
        
        if len(weight_map.shape) == 5:
            # Blur expects (B, C, H, W), so we squeeze/unsqueeze depth
            weight_map = blurrer(weight_map.squeeze(2)).unsqueeze(2)
        else:
            weight_map = blurrer(weight_map)
            
    return weight_map

from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_edge_map(base_image, edge_map, title="Edge Weight Map"):
    """
    Exact visualization function from the trainer.
    Crucially includes .T to match orientation.
    Updated to align colorbar height with image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Display the base image (e.g., preprocessed magnitude)
    ax.imshow(base_image.T, cmap='gray')
    
    # Overlay the edge map
    # We assign this to a variable 'overlay' so we can pass it to the colorbar
    overlay = ax.imshow(edge_map.T, cmap='hot_r', alpha=0.6, vmin=0.0, vmax=1.0)
    
    # --- NEW: Align Colorbar Height ---
    # This creates a divider for the existing axes 'ax'
    divider = make_axes_locatable(ax)
    
    # Append a new axis to the right of 'ax', with 5% width and 0.05 padding
    # This ensures the colorbar axis ('cax_bar') has the exact same height as the image
    cax_bar = divider.append_axes("right", size="5%", pad=0.05)
    
    fig.colorbar(overlay, cax=cax_bar, label="Edge Weight")
    # ----------------------------------

    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    return fig

# --- 4. Main Execution ---

def load_slice(path, is_label=False):
    nii = nib.load(path)
    data = nii.get_fdata()
    
    # Extract Middle Slice
    if data.ndim == 3:
        mid = data.shape[2] // 2
        slice_data = data[:, :, mid]
    elif data.ndim == 4:
        mid = data.shape[2] // 2
        slice_data = data[:, :, mid, 0]
    else:
        slice_data = data

    if is_label:
        return slice_data.astype(np.int64)
    else:
        # Normalize Mag to [0, 1] for viz
        d_min, d_max = np.min(slice_data), np.max(slice_data)
        if d_max - d_min > 1e-8:
            return (slice_data - d_min) / (d_max - d_min)
        return slice_data

def run_edge_viz(mag_path, label_path):
    print(f"Generating Edge Map Viz for:\n  Mag: {mag_path}\n  Label: {label_path}")
    
    # 1. Load Data
    mag_img = load_slice(mag_path, is_label=False)
    label_img = load_slice(label_path, is_label=True)
    
    # 2. Prepare Tensors for 'create_weight_map'
    # Need to convert label to One-Hot: (Num_Classes, Batch, 1, H, W)
    
    # A. Tensorify and add Batch/Depth dims: (1, 1, H, W)
    label_tensor = torch.from_numpy(label_img).long().unsqueeze(0).unsqueeze(0)
    
    # B. Determine Num Classes (Max label + 1 or fixed based on params)
    # Your params go up to index 4 (so 5 classes), plus potentially background.
    # Let's assume standard behavior: max label in map + 1
    num_classes = max(label_tensor.max().item() + 1, 6) # Ensure at least 6 for your params (0-5)
    
    # C. One Hot Encoding
    # F.one_hot output: (1, 1, H, W, Num_Classes)
    one_hot = F.one_hot(label_tensor.squeeze(1), num_classes=num_classes)
    
    # Permute to (Num_Classes, Batch, H, W)
    # one_hot is (Batch, 1, H, W, Num_C) -> Permute to (Num_C, Batch, 1, H, W)
    one_hot = one_hot.permute(3, 0, 1, 2).float()
    
    # 3. Generate Map
    weight_map_tensor = create_weight_map(one_hot, EDGE_PARAMS, BLUR_SIGMA)
    
    # 4. Extract for Plotting
    # weight_map_tensor shape: (Batch, 1, 1, H, W) -> Squeeze to (H, W)
    weight_map_np = weight_map_tensor.squeeze().cpu().numpy()
    
    # 5. Visualize
    fig = plot_edge_map(mag_img, weight_map_np, title="")
    
    # 6. Save
    mag_name = os.path.basename(mag_path).replace('.nii.gz', '').replace('.nii', '')
    
    suffix = "___sub-PML019_S1_71.0cm_T2s_75i_TR45TE9_cor_18avg_900_slice-157"
    save_filename = f"edges_{mag_name}{suffix}.png"
    
    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path)
    print(f"Saved to: {save_path}")
    plt.close()

if __name__ == "__main__":
    # --- CONFIGURATION ---
    mag_file = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/imagesTr/MagPhaseExp_simple_training_base_0000_0000.nii.gz" 
    label_file = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/labelsTr/MagPhaseExp_simple_training_base_0000.nii.gz"
    
    run_edge_viz(mag_file, label_file)