
import numpy as np
import matplotlib.pyplot as plt


def decode_bitmask_to_7_channels(bitmask: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Decodes a single-channel integer bitmask into a 7-channel binary mask,
    where each channel corresponds to a unique foreground value (1 through num_classes).
    """
    h, w = bitmask.shape
    multi_channel_mask = np.zeros((num_classes, h, w), dtype=np.uint8)
    
    for i in range(1, num_classes + 1):
        channel_idx = i - 1
        multi_channel_mask[channel_idx][bitmask == i] = 1
        
    return multi_channel_mask


def plot_edge_map(base_image, edge_map, title="Edge Weight Map", epoch=0):
    """
    Creates a plot of the edge weight map overlaid on a base image.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Display the base image (e.g., preprocessed magnitude)
    ax.imshow(base_image.T, cmap='gray')
    
    # Overlay the edge map. 'hot' is a good colormap for weights.  
    cax = ax.imshow(edge_map[0].T, cmap='hot_r', alpha=0.6, vmin=0.0, vmax=1.0)
    
    fig.colorbar(cax, ax=ax, label="Edge Weight")
    ax.set_title(f"{title} (Epoch {epoch})")
    ax.axis('off')
    fig.tight_layout()
    return fig


def plot_comparison_and_segmentation(
    orig_mag, orig_phase,
    proc_mag, proc_phase,
    gt_masks, pred_masks_full, pred_masks_simplified,
    aug_mag=None, aug_phase=None 
):
    colors = ['black', 'cyan', 'lime', 'red', 'yellow', 'magenta', 'orange', 'purple']
    fig, axs = plt.subplots(4, 3, figsize=(18, 24), gridspec_kw={'wspace': 0.1, 'hspace': 0.2})
    fig.suptitle('Input Comparison and Segmentation Results', fontsize=20)
    
    if orig_mag is not None: 
        axs[0, 0].imshow(orig_mag.T, cmap='gray')
        axs[0, 0].set_title("Original Magnitude")
        axs[0, 0].axis('off')

    if orig_phase is not None: 
        axs[0, 1].imshow(orig_phase.T, cmap='gray')
        axs[0, 1].set_title("Original Phase")
        axs[0, 1].axis('off')
        axs[0, 2].axis('off')

    if aug_mag is not None:
        axs[1, 0].imshow(aug_mag.T, cmap='gray')
        axs[1, 0].set_title("Augmented Magnitude")
    axs[1, 0].axis('off')

    if aug_phase is not None:
        axs[1, 1].imshow(aug_phase.T, cmap='gray')
        axs[1, 1].set_title("Augmented Phase")
    axs[1, 1].axis('off')
    axs[1, 2].axis('off')

    if proc_mag is not None: 
        axs[2, 0].imshow(proc_mag.T, cmap='gray')
        axs[2, 0].set_title("Preprocessed Magnitude")
        axs[2, 0].axis('off')

    if proc_phase is not None: 
        axs[2, 1].imshow(proc_phase.T, cmap='gray')
        axs[2, 1].set_title("Preprocessed Phase")
        axs[2, 1].axis('off')
        axs[2, 2].axis('off')

    segmentation_plots = [
        (gt_masks, "Ground Truth"),
        (pred_masks_full, "Prediction (Multi-Label)"),
        (pred_masks_simplified, "Prediction (Winner-Takes-All)")
    ]

    for i, (masks, title) in enumerate(segmentation_plots):
        ax = axs[3, i]
        if aug_mag is not None :
            ax.imshow(aug_mag.T, cmap='gray')
        else : 
            ax.imshow(proc_mag.T, cmap='gray')
        ax.set_title(title)
        ax.axis('off')

        if masks is not None and masks.ndim == 3:
            for class_idx in range(masks.shape[0]):
                adjustment = 1 if i == 0 else 0
                color = colors[(class_idx + adjustment) % len(colors)]
                if np.any(masks[class_idx]):
                    ax.contourf(masks[class_idx].T, levels=[0.5, 1.5], colors=[color], alpha=0.3)
                    ax.contour(masks[class_idx].T, levels=[0.5], colors=[color], linewidths=1.5)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig