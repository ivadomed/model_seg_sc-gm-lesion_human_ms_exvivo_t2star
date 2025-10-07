import torch
import os
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import kornia
import torch.nn.functional as F
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn

# --- 1. CONFIGURATION ---
DEVICE = torch.device('cuda:0')

MODEL_FOLDER = "/home/ge.polymtl.ca/pahoa/nih_project/model_seg_sc-gm-lesion_human_ms_exvivo_t2star/Dataset503_MagPhaseExp_no_edges/nnUNetTrainerWandb__nnUNetPlans__2d"

PATH_PROCESSED = Path("/home/ge.polymtl.ca/pahoa/nih_project/datasets/processed_data_full_multichannel")
SLICED_DATA_DIR = PATH_PROCESSED / "data_slice"
NNUNET_RAW_DIR = PATH_PROCESSED / "nnunet_raw"
NNUNET_PREPROCESSED_DIR = PATH_PROCESSED / "nnunet_preprocessed"
DATASET_NAME_OR_ID = "Dataset503_MagPhaseExp"
TASK_NAME = "MagPhaseExp"

# --- Input/Output Paths ---
OUTPUT_FOLDER = Path(MODEL_FOLDER) / "validation_png_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 2. HELPER FUNCTIONS (ADAPTED FROM YOUR TRAINER & NEW) ---

def _preprocess_data_gpu(data_batch: torch.Tensor) -> torch.Tensor:
    mag_batch, phase_batch = data_batch[:, 0:1, :, :], data_batch[:, 1:2, :, :]
    thresholds = kornia.filters.otsu_threshold(mag_batch)[1].view(-1, 1, 1, 1)
    
    mask = (mag_batch >= thresholds).float()
    kernel = torch.ones(3, 3, device=mask.device)
    mask = kornia.morphology.opening(mask, kernel)
    
    mag_min, mag_max = mag_batch.amin(dim=(-2, -1), keepdim=True), mag_batch.amax(dim=(-2, -1), keepdim=True)
    mag_normalized = (mag_batch - mag_min) / (mag_max - mag_min + 1e-6)
    mag_clahe = kornia.enhance.equalize_clahe(mag_normalized)
    processed_mag = mag_clahe * mask
    
    masked_phase = torch.masked_select(phase_batch, mask.bool())
    p30, p85 = (torch.quantile(masked_phase, q).item() for q in [0.30, 0.85]) if masked_phase.numel() > 0 else (0.0, 1.0)
    phase_rescaled = torch.clamp((phase_batch - p30) / (p85 - p30 + 1e-6), 0, 1)
    
    return torch.cat([processed_mag, phase_rescaled * mask], dim=1)

def create_mosaic_from_single_image(image_tensor: torch.Tensor, target_size: tuple = (384, 384)) -> torch.Tensor:
    mosaic = image_tensor.repeat(1, 1, 2, 2)
    return mosaic

def unmosaic_and_resize(mosaic_tensor: torch.Tensor, original_shape: tuple, target_size: tuple = (384, 384)) -> torch.Tensor:
    """Reverses the mosaicing process for a prediction tensor."""
    large_mosaic_size = (target_size[0] * 2, target_size[1] * 2)
    # 1. Resize the small prediction back to the large mosaic grid size    
    large_mosaic_pred = F.interpolate(mosaic_tensor, size=large_mosaic_size, mode='bilinear', align_corners=False)
    # 2. Extract the top-left quadrant, which corresponds to the original image
    original_pred_in_grid = large_mosaic_pred[:, :, :target_size[0], :target_size[1]]
    # 3. Resize this quadrant to the original image's dimensions
    final_pred = F.interpolate(original_pred_in_grid, size=original_shape, mode='bilinear', align_corners=False)
    return final_pred

def calculate_dice_scores(pred_mask_np: np.ndarray, gt_mask_np: np.ndarray, class_labels: dict) -> dict:
    """Calculates Dice scores for each foreground class."""
    scores = {}
    pred_one_hot = torch.from_numpy(decode_bitmask_to_7_channels(pred_mask_np, len(class_labels) - 1)).unsqueeze(0)
    gt_one_hot = torch.from_numpy(decode_bitmask_to_7_channels(gt_mask_np, len(class_labels) - 1)).unsqueeze(0)
    
    tp, fp, fn, _ = get_tp_fp_fn_tn(pred_one_hot, gt_one_hot, axes=[0, 2, 3])
    
    # Dice = 2TP / (2TP + FP + FN)
    dice_vals = (2 * tp) / (2 * tp + fp + fn + 1e-8)
    
    for i, class_name in enumerate(class_labels.keys()):
        if class_name == "background": continue
        scores[class_name] = dice_vals[i-1].item() # Indexing matches foreground classes
    return scores

def decode_bitmask_to_7_channels(bitmask: np.ndarray, num_classes: int) -> np.ndarray:
    """Decodes a single-channel integer bitmask into a multi-channel binary mask."""
    # (Adapted from your trainer)
    multi_channel_mask = np.zeros((num_classes, bitmask.shape[0], bitmask.shape[1]), dtype=np.uint8)
    for i in range(1, num_classes + 1):
        multi_channel_mask[i-1][bitmask == i] = 1
    return multi_channel_mask

def create_side_by_side_comparison_plot(base_image, gt_mask, pred_mask, dice_scores, title_str, class_labels):
    """Generates the final PNG output figure."""
    colors = ['cyan', 'lime', 'red', 'yellow']
    fig, axs = plt.subplots(1, 3, figsize=(20, 8), gridspec_kw={'width_ratios': [1, 1, 0.5]})
    fig.suptitle(title_str, fontsize=16)

    # Plot Ground Truth
    axs[0].imshow(base_image.T, cmap='gray')
    axs[0].set_title("Ground Truth", fontsize=14)
    gt_decoded = decode_bitmask_to_7_channels(gt_mask, len(class_labels) - 1)
    for i in range(gt_decoded.shape[0]):
        if np.any(gt_decoded[i]):
            axs[0].contour(gt_decoded[i].T, levels=[0.5], colors=[colors[i]], linewidths=1.5)

    # Plot Prediction
    axs[1].imshow(base_image.T, cmap='gray')
    axs[1].set_title("Prediction (Winner-Takes-All)", fontsize=14)
    pred_decoded = decode_bitmask_to_7_channels(pred_mask, len(class_labels) - 1)
    for i in range(pred_decoded.shape[0]):
        if np.any(pred_decoded[i]):
            axs[1].contour(pred_decoded[i].T, levels=[0.5], colors=[colors[i]], linewidths=1.5)

    for ax in [axs[0], axs[1]]: ax.axis('off')

    # Display Dice Scores
    axs[2].axis('off')
    dice_text = "Dice Scores:\n\n"
    for name, score in dice_scores.items():
        dice_text += f"{name}: {score:.4f}\n"
    axs[2].text(0.05, 0.95, dice_text, transform=axs[2].transAxes, fontsize=12, verticalalignment='top')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# --- 3. SCRIPT EXECUTION ---

# Define class labels from your trainer for plotting and Dice calculation
class_labels = {"background": 0, "WM": 1, "GM": 2, "lesion_WM": 3, "lesion_GM": 4}

manifest_path = NNUNET_RAW_DIR / DATASET_NAME_OR_ID / "inference_manifest.json"
print(f"Loading inference manifest from: {manifest_path}")
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

splits_file_path = NNUNET_PREPROCESSED_DIR / DATASET_NAME_OR_ID / 'splits_final.json'
with open(splits_file_path, 'r') as f: splits = json.load(f)
validation_case_ids = splits[0]['val']
print(f"Found {len(validation_case_ids)} validation cases for fold 0.")

predictor = nnUNetPredictor(device=DEVICE)
predictor.initialize_from_trained_model_folder(MODEL_FOLDER, use_folds=(0,), checkpoint_name='checkpoint_final.pth')
network = predictor.network.to(DEVICE).eval()
print(f"Predictor initialized using model from: {MODEL_FOLDER}")

with torch.no_grad():
    for case_id in tqdm(validation_case_ids, desc="Running Inference & Plotting"):
        
        # Use the manifest to get the original filename for the current validation case_id
        original_name_key = manifest.get(case_id)
        if not original_name_key:
            print(f"⚠️ WARNING: Could not find case {case_id} in manifest. Skipping.")
            continue
        
        subject_id = original_name_key.split('_')[0]
        mag_path = SLICED_DATA_DIR / subject_id / 'anat' / f"{original_name_key}_part-mag.nii.gz"
        phase_path = SLICED_DATA_DIR / subject_id / 'anat' / f"{original_name_key}_part-phase.nii.gz"
        label_path = NNUNET_RAW_DIR / DATASET_NAME_OR_ID / 'labelsTr' / f"{case_id}.nii.gz"
        aligned_mag_path = NNUNET_RAW_DIR / DATASET_NAME_OR_ID / 'imagesTr' / f"{case_id}_0000.nii.gz"

        mag_nii = nib.load(mag_path)
        original_shape = mag_nii.shape[:2]
        mag_data = torch.from_numpy(mag_nii.get_fdata().squeeze()).float().unsqueeze(0)
        phase_data = torch.from_numpy(nib.load(phase_path).get_fdata().squeeze()).float().unsqueeze(0)
        image_tensor = torch.cat([mag_data, phase_data], dim=0).unsqueeze(0).to(DEVICE)
        
        # Preprocess -> Mosaic -> Predict
        preprocessed_tensor = _preprocess_data_gpu(image_tensor)
        mosaic_tensor = create_mosaic_from_single_image(preprocessed_tensor)
        mosaic_logits = network(mosaic_tensor)
        mosaic_probs_tensor = torch.sigmoid(mosaic_logits)

        pred_probs_resized = unmosaic_and_resize(mosaic_probs_tensor, original_shape)
        pred_mask_np = torch.argmax(pred_probs_resized, dim=1).squeeze().cpu().numpy().astype(np.uint8)

        gt_mask_np = nib.load(label_path).get_fdata().squeeze().astype(np.uint8)
        
        dice_scores = calculate_dice_scores(pred_mask_np, gt_mask_np, class_labels)
        
        # Generate and save the final PNG plot using the aligned base image
        title = f"File: {original_name_key}"
        base_image_for_plot = nib.load(aligned_mag_path).get_fdata().squeeze()
        
        fig = create_side_by_side_comparison_plot(
            base_image_for_plot, gt_mask_np, pred_mask_np, dice_scores, title, class_labels
        )
        
        output_filename = OUTPUT_FOLDER / f"{case_id}_comparison.png"
        fig.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

print("\n✅ Inference and plotting complete!")
print(f"Comparison PNGs saved in: {OUTPUT_FOLDER}")