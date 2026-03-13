import argparse
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

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
        
    ax.imshow(bg_img, cmap='gray', interpolation='none')
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    
    if mask is not None:
        # Auto-detect number of classes if values exceed default
        max_val = int(np.max(mask))
        if max_val > num_classes:
             num_classes = max_val

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

def main():
    parser = argparse.ArgumentParser(description="Visualize GT overlay on Magnitude image (High Res).")
    parser.add_argument("--image", type=str, required=True, help="Path to magnitude image (nii.gz)")
    parser.add_argument("--gt", type=str, required=True, help="Path to ground truth mask (nii.gz)")
    parser.add_argument("--output", type=str, default="overlay_viz.png", help="Path to save the output visualization")
    args = parser.parse_args()

    # Load images
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.exists(args.gt):
        raise FileNotFoundError(f"GT not found: {args.gt}")

    print(f"Loading image: {args.image}")
    img_nii = nib.load(args.image)
    img_data = img_nii.get_fdata()

    print(f"Loading GT: {args.gt}")
    gt_nii = nib.load(args.gt)
    gt_data = gt_nii.get_fdata()
    
    # Handle NaNs
    img_data = np.nan_to_num(img_data)

    # Transpose data to match visualization_utils expectation
    # Nibabel loads (X, Y, Z). 
    # Logic in utils assumes (Z, Y, X) for Axial/Coronal/Sagittal slicing order.
    # Transposing to (Z, Y, X).
    img_data = img_data.transpose(2, 1, 0)
    gt_data = gt_data.transpose(2, 1, 0)

    # 3D Center slices
    c_z = img_data.shape[0] // 2
    c_y = img_data.shape[1] // 2
    c_x = img_data.shape[2] // 2
    
    print(f"Data shape (Z, Y, X): {img_data.shape}")
    print(f"Center slices - Axial(Z): {c_z}, Coronal(Y): {c_y}, Sagittal(X): {c_x}")

    # Create figure with high DPI for resolution
    # 1 row, 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
    
    # Axial (Slice Z)
    # img[c_z, :, :]
    plot_overlay(axs[0], img_data[c_z, :, :], gt_data[c_z, :, :], "Axial", num_classes=4)

    # Coronal (Slice Y)
    # img[:, c_y, :]
    plot_overlay(axs[1], img_data[:, c_y, :], gt_data[:, c_y, :], "Coronal", num_classes=4)

    # Sagittal (Slice X)
    # img[:, :, c_x]
    plot_overlay(axs[2], img_data[:, :, c_x], gt_data[:, :, c_x], "Sagittal", num_classes=4)
    
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight', pad_inches=0.1, dpi=300)
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
