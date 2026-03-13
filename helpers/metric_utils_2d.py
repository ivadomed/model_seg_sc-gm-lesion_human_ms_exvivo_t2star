import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion, generate_binary_structure

def _ensure_2d_binary(mask):
    """
    Squeezes dimensions to ensure (H, W) shape for 2D operations.
    Converts to boolean.
    """
    mask = np.asanyarray(mask)
    # Handle (1, H, W) or (C, H, W) where we want the first channel
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    elif mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask[..., 0]
    
    return mask.astype(bool)

def _get_border_points(mask, spacing):
    """
    Extracts border points from a binary mask and scales them by spacing.
    """
    # 2D structure for 2D surface distance
    struct = generate_binary_structure(2, 1)
    
    eroded = binary_erosion(mask, structure=struct)
    border = np.logical_xor(mask, eroded)
    points = np.argwhere(border)
    
    if len(points) == 0:
        return np.empty((0, 2))
    
    # Scale points by spacing to get mm coordinates
    return points * np.array(spacing)

def compute_binary_dice_2d(pred, gt):
    """
    Computes standard Dice coefficient for 2D masks.
    """
    pred = _ensure_2d_binary(pred)
    gt = _ensure_2d_binary(gt)
    
    pred_sum = pred.sum()
    gt_sum = gt.sum()
    
    if gt_sum == 0:
        return np.nan if pred_sum == 0 else 0.0
    
    intersection = np.logical_and(pred, gt).sum()
    return (2.0 * intersection) / (pred_sum + gt_sum + 1e-8)

def compute_surface_distances_2d(pred, gt, spacing):
    """
    Computes the raw list of surface distances (bidirectional).
    
    Args:
        pred: Prediction mask (2D or 3D with singleton dim)
        gt: Ground Truth mask (2D or 3D with singleton dim)
        spacing: Tuple of (x, y) or (x, y, z) spacing from SimpleITK.
    
    Returns:
        np.array: List of all surface distances (concatenated pred->gt and gt->pred).
                  Returns [0.0] if both are empty (perfect match).
                  Returns None if one is empty and the other is not (infinite penalty).
    """
    pred = _ensure_2d_binary(pred)
    gt = _ensure_2d_binary(gt)
    
    pred_empty = not np.any(pred)
    gt_empty = not np.any(gt)
    
    # Case 1: Both empty - Perfect match
    if pred_empty and gt_empty:
        return np.array([0.0])
    
    # Case 2: One empty - Penalty (Caller decides whether to treat as NaN or Max)
    if pred_empty or gt_empty:
        return None

    # Handle Spacing: Align SimpleITK (x,y) with Numpy (y,x)
    # Numpy images are [y, x]. Spacing input is usually [x, y] or [x, y, z].
    # We need spacing to be [spacing_y, spacing_x]
    if spacing is not None:
        if len(spacing) >= 2:
            # Take first two dimensions and swap them: (x, y, ...) -> (y, x)
            current_spacing = (spacing[1], spacing[0])
        else:
            current_spacing = (1.0, 1.0)
    else:
        current_spacing = (1.0, 1.0)

    # Double check dimensions
    if pred.ndim != len(current_spacing):
        return None

    pred_border = _get_border_points(pred, current_spacing)
    gt_border = _get_border_points(gt, current_spacing)

    if len(pred_border) == 0 or len(gt_border) == 0:
        return None

    tree_gt = cKDTree(gt_border)
    tree_pred = cKDTree(pred_border)

    d_pred_to_gt, _ = tree_gt.query(pred_border, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt_border, k=1)

    return np.concatenate([d_pred_to_gt, d_gt_to_pred])

def calculate_hd95(distances_array):
    """
    Computes the 95th percentile from a list of raw distances.
    Useful for both single-slice HD95 and global case-wise HD95 (by aggregating raw arrays).
    """
    if distances_array is None or len(distances_array) == 0:
        return np.nan
    
    # Check for the [0.0] "perfect empty match" marker
    if len(distances_array) == 1 and distances_array[0] == 0.0:
        return 0.0
        
    return np.percentile(distances_array, 95)