import matplotlib.pyplot as plt
import numpy as np

# Standard colors for segmentation classes
# 0:Background, 1:Cyan, 2:Lime, 3:Red, 4:Yellow, 5:Magenta
COLORS = ['black', 'cyan', 'lime', 'red', 'yellow', 'magenta', 'orange', 'purple']

def decode_bitmask(bitmask, num_classes=4):
    """
    Converts an integer bitmask (H, W) into one-hot channels (num_classes, H, W).
    Handles singleton dimensions if passed (1, H, W).
    """
    if bitmask is None: 
        return None
        
    # Squeeze singleton dimensions if (1, H, W) or (C, H, W) where C=1
    if bitmask.ndim == 3:
        if bitmask.shape[0] == 1:
            bitmask = bitmask[0]
        elif bitmask.shape[-1] == 1:
            bitmask = bitmask[..., 0]
            
    h, w = bitmask.shape
    multi_channel = np.zeros((num_classes, h, w), dtype=np.uint8)
    
    for i in range(1, num_classes + 1): 
        multi_channel[i-1] = (bitmask == i).astype(np.uint8)
        
    return multi_channel

def plot_overlay(ax, bg_img, mask, title, num_classes=4, alpha=0.3):
    """
    Plots a background image with segmentation contours overlaid.
    """
    # Ensure background is 2D
    if bg_img.ndim == 3:
        # If (C, H, W), take first channel (Magnitude)
        bg_img = bg_img[0]
        
    ax.imshow(bg_img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    
    if mask is not None:
        masks_mc = decode_bitmask(mask, num_classes)
        for c_idx in range(num_classes):
            # Map class index to color list (offset by 1 because 0 is bg)
            color_idx = c_idx + 1
            color = COLORS[color_idx % len(COLORS)]
            
            mask_c = masks_mc[c_idx]
            if np.any(mask_c):
                # Filled transparent contour
                ax.contourf(mask_c, levels=[0.5, 1.5], colors=[color], alpha=alpha)
                # Thin solid line for boundary
                ax.contour(mask_c, levels=[0.5], colors=[color], linewidths=0.8)

def save_ortho_view(img_data_3d, pred_data_3d, gt_data_3d, case_id, output_path):
    """
    Generates and saves a 2x3 grid of orthogonal views (Axial, Coronal, Sagittal).
    Top Row: Ground Truth
    Bottom Row: Prediction
    """
    # Ensure image is (D, H, W) or (C, D, H, W)
    # If 4D (C, D, H, W), take first channel for visualization
    img = img_data_3d[0] if img_data_3d.ndim == 4 else img_data_3d
    
    # Calculate center slices
    c_z, c_y, c_x = np.array(img.shape) // 2
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"3D View: {case_id}", fontsize=16)
    
    # Define Slices: (Image, GT, Pred, Title)
    # Note: Axes are (Z, Y, X) -> (0, 1, 2)
    slices = [
        # Axial (XY plane, slice Z)
        (
            img[c_z, :, :], 
            gt_data_3d[c_z, :, :] if gt_data_3d is not None else None, 
            pred_data_3d[c_z, :, :], 
            "Axial"
        ),
        # Coronal (XZ plane, slice Y)
        (
            img[:, c_y, :], 
            gt_data_3d[:, c_y, :] if gt_data_3d is not None else None, 
            pred_data_3d[:, c_y, :], 
            "Coronal"
        ),
        # Sagittal (YZ plane, slice X)
        (
            img[:, :, c_x], 
            gt_data_3d[:, :, c_x] if gt_data_3d is not None else None, 
            pred_data_3d[:, :, c_x], 
            "Sagittal"
        )
    ]
    
    for i, (im_s, gt_s, pred_s, title) in enumerate(slices):
        # Top Row: Ground Truth
        if gt_s is not None:
            plot_overlay(axs[0, i], im_s, gt_s, f"{title} - GT", num_classes=4)
        else:
            plot_overlay(axs[0, i], im_s, None, f"{title} - GT (None)", num_classes=4)
            
        # Bottom Row: Prediction
        plot_overlay(axs[1, i], im_s, pred_s, f"{title} - Pred", num_classes=4)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_2d_slice_viz(vol_slice_mag, vol_slice_phase, gt_2d, pred_slice, case_2d_id, output_path):
    """
    Visualizes a single 2D slice with separate Mag/Phase panels (if Phase exists).
    Replaces plot_2d_slice_eval in inference_3d.py
    """
    # Determine columns based on whether phase exists
    has_phase = vol_slice_phase is not None
    cols = 4 if has_phase else 3
    
    fig, axs = plt.subplots(1, cols, figsize=(6*cols, 6))
    
    # 1. Magnitude (Raw)
    axs[0].imshow(vol_slice_mag, cmap='gray')
    axs[0].set_title(f"{case_2d_id}\nMagnitude")
    axs[0].axis('off')

    idx_offset = 1
    
    # 2. Phase (Optional)
    if has_phase:
        axs[1].imshow(vol_slice_phase, cmap='gray')
        axs[1].set_title("Phase")
        axs[1].axis('off')
        idx_offset = 2

    # 3. Ground Truth Overlay
    plot_overlay(axs[idx_offset], vol_slice_mag, gt_2d, "Ground Truth", num_classes=4)
    
    # 4. Prediction Overlay
    plot_overlay(axs[idx_offset+1], vol_slice_mag, pred_slice, "Prediction", num_classes=4)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()