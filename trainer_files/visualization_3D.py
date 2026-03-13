import numpy as np
import matplotlib.pyplot as plt


def decode_bitmask_to_multichannel(bitmask: np.ndarray, num_classes: int) -> list:
    """
    Decodes a label map into a list of binary masks per class.
    """
    masks = []
    # Assuming 0 is background, start from 1
    for i in range(1, num_classes + 1):
        masks.append(bitmask == i)
    return masks


def plot_3d_snapshot(mag_vol, phase_vol, gt_vol, pred_vol, num_classes, epoch, 
                     title_prefix="Validation", edge_vol=None):
    """
    Plots Axial, Coronal, and Sagittal views.
    Includes an optional 5th column for Edge/Weight Map if 'edge_vol' is provided.
    """
    shp = mag_vol.shape
    mid_z, mid_y, mid_x = shp[0] // 2, shp[1] // 2, shp[2] // 2
    
    views = [
        ('Coronal', lambda x: x[mid_z, :, :]), 
        ('Axial', lambda x: x[:, mid_y, :]), 
        ('Sagittal', lambda x: x[:, :, mid_x])
    ]
    
    n_cols = 5 if edge_vol is not None else 4
    
    fig, axs = plt.subplots(3, n_cols, figsize=(5 * n_cols, 15))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    fig.suptitle(f'{title_prefix} Snapshot - Epoch {epoch}', fontsize=16)
    
    colors = ['cyan', 'lime', 'red', 'yellow', 'magenta', 'orange', 'purple']

    def decode_bitmask_to_multichannel(bitmask, num_classes):
        masks = []
        for i in range(1, num_classes + 1):
            masks.append(bitmask == i)
        return masks

    for row_idx, (view_name, slicer) in enumerate(views):
        # 1. Magnitude
        axs[row_idx, 0].imshow(slicer(mag_vol).T, cmap='gray')
        axs[row_idx, 0].axis('off')
        axs[row_idx, 0].set_title(f"{view_name} - Mag")

        # 2. Phase
        if np.count_nonzero(phase_vol) == 0: 
            axs[row_idx, 1].text(0.5, 0.5, 'N/A', ha='center', va='center')
            axs[row_idx, 1].axis('off')
        else: 
            axs[row_idx, 1].imshow(slicer(phase_vol).T, cmap='gray')
            axs[row_idx, 1].axis('off')
        axs[row_idx, 1].set_title(f"{view_name} - Phase")

        # 3. Ground Truth
        axs[row_idx, 2].imshow(slicer(mag_vol).T, cmap='gray') 
        gt_slice = slicer(gt_vol)
        gt_channels = decode_bitmask_to_multichannel(gt_slice, num_classes)
        for i, channel in enumerate(gt_channels):
            if np.any(channel):
                color = colors[i % len(colors)]
                axs[row_idx, 2].contourf(channel.T, levels=[0.5, 1.5], colors=[color], alpha=0.3)
                axs[row_idx, 2].contour(channel.T, levels=[0.5], colors=[color], linewidths=1)
        axs[row_idx, 2].axis('off')
        axs[row_idx, 2].set_title(f"{view_name} - GT")

        # 4. Prediction
        axs[row_idx, 3].imshow(slicer(mag_vol).T, cmap='gray')
        pred_slice = slicer(pred_vol)
        pred_channels = decode_bitmask_to_multichannel(pred_slice, num_classes)
        for i, channel in enumerate(pred_channels):
            if np.any(channel):
                color = colors[i % len(colors)]
                axs[row_idx, 3].contourf(channel.T, levels=[0.5, 1.5], colors=[color], alpha=0.3)
                axs[row_idx, 3].contour(channel.T, levels=[0.5], colors=[color], linewidths=1)
        axs[row_idx, 3].axis('off')
        axs[row_idx, 3].set_title(f"{view_name} - Pred")
        
        # 5. Edge / Weight Map (FIXED LOGIC)
        if edge_vol is not None and n_cols > 4:
            # A. Plot Background
            axs[row_idx, 4].imshow(slicer(mag_vol).T, cmap='gray')
            
            # B. Get Slice
            edge_slice = slicer(edge_vol)
            
            # C. Mask ONLY the background (Standard Weight = 1.0)
            # This allows weights < 1.0 (0.9) and > 1.0 (2.0) to both appear.
            # We use isclose to handle floating point minor differences.
            background_value = 1.0
            masked_edge = np.ma.masked_where(np.isclose(edge_slice, background_value, atol=1e-3), edge_slice)
            
            # E. Plot
            im = axs[row_idx, 4].imshow(masked_edge.T, cmap='hot_r', alpha=0.6, vmin=0.0, vmax=1.0)
            
            axs[row_idx, 4].axis('off')
            axs[row_idx, 4].set_title(f"{view_name} - Soft Loss Map")

    return fig