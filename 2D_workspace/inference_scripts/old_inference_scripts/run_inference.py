import torch
import os
import argparse
import json
from pathlib import Path
import nibabel as nib
import numpy as np
import kornia
import torch.nn.functional as F
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd 
from torch.nn.functional import pad

# --- 1. CONFIGURATION & HELPERS ---

def log_tensor_stats(name, tensor, log_file):
    """
    Logs tensor statistics to a text file for debugging.
    """
    t = tensor.detach().cpu().float()
    with open(log_file, "a") as f:
        f.write(f"--- {name} ---\n")
        f.write(f"Shape: {tuple(t.shape)}\n")
        f.write(f"Mean:  {t.mean().item():.6f}\n")
        f.write(f"Std:   {t.std().item():.6f}\n")
        f.write(f"Min:   {t.min().item():.6f}\n")
        f.write(f"Max:   {t.max().item():.6f}\n")
        try:
            p05, p95 = np.percentile(t.numpy(), [5, 95])
            f.write(f"5th %: {p05:.6f}\n")
            f.write(f"95th%: {p95:.6f}\n")
        except:
            pass
        f.write("\n")

def resize_and_pad_to_size(input_tensor, target_size):
    """
    1. Resizes input_tensor (B, C, H, W) maintaining aspect ratio so it fits inside target_size.
    2. Pads the result to match target_size exactly.
    Returns: padded_tensor, (padding_tuple, resized_shape)
    """
    b, c, h, w = input_tensor.shape
    target_h, target_w = target_size

    # 1. Calculate Scaling Factor
    # We want the image to fit entirely, so we take the min ratio
    scale_h = target_h / h
    scale_w = target_w / w
    scale = min(scale_h, scale_w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    # 2. Resize
    # Use bilinear for images (input)
    resized_tensor = F.interpolate(input_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

    # 3. Pad
    diff_h = target_h - new_h
    diff_w = target_w - new_w

    pad_l = diff_w // 2
    pad_r = diff_w - pad_l
    pad_t = diff_h // 2
    pad_b = diff_h - pad_t

    padded_tensor = F.pad(resized_tensor, (pad_l, pad_r, pad_t, pad_b), mode='constant', value=0)
    
    # Return tensor and info needed to reverse the process
    # info: (padding_tuple, spatial_shape_after_resize)
    return padded_tensor, ((pad_l, pad_r, pad_t, pad_b), (new_h, new_w))

def reverse_resize_and_pad(output_tensor, transformation_info, original_shape):
    """
    Reverses the padding and resizing to get back to original_shape.
    """
    padding, resized_shape = transformation_info
    pad_l, pad_r, pad_t, pad_b = padding
    
    # 1. Unpad (Crop)
    h_curr, w_curr = output_tensor.shape[2], output_tensor.shape[3]
    cropped = output_tensor[..., pad_t : h_curr - pad_b, pad_l : w_curr - pad_r]
    
    # 2. Resize back to original resolution
    # Use bilinear for probabilities
    original_h, original_w = original_shape
    final_output = F.interpolate(cropped, size=(original_h, original_w), mode='bilinear', align_corners=False)
    
    return final_output

def get_experiment_settings(exp_name: str):
    settings = {
        "mag_only": False,
        "mag_prepro": False,
        "phase_prepro": False,
        "do_mosaic": False,
        "input_patch_size": (224, 320) 
    }

    if exp_name == "exp_base": pass
    elif exp_name == "exp_mosaic": settings["do_mosaic"] = True
    elif exp_name == "exp_mag_only": settings["mag_only"] = True
    elif exp_name == "exp_phase_prepro": settings["phase_prepro"] = True
    elif exp_name == "exp_mag_prepro": settings["mag_prepro"] = True
    return settings

def get_class_configuration(model_folder: Path):
    dataset_json_path = model_folder / "dataset.json"
    if not dataset_json_path.exists():
        dataset_json_path = model_folder.parent / "dataset.json"
    
    if dataset_json_path.exists():
        with open(dataset_json_path, 'r') as f:
            info = json.load(f)
        labels = info.get('labels', {})
        if labels and isinstance(list(labels.values())[0], int):
             pass 
        elif labels and isinstance(list(labels.keys())[0], int):
             labels = {v: int(k) for k, v in labels.items()}
        num_classes = len(labels)
    else:
        num_classes = 5

    print(f"ℹ️ Detected {num_classes} classes in model configuration.")
    if num_classes == 5:
        return {"background": 0, "WM": 1, "GM": 2, "lesion_WM": 3, "lesion_GM": 4}
    elif num_classes == 3:
        return {"background": 0, "WM": 1, "lesion": 2}
    else:
        return labels if dataset_json_path.exists() else {f"class_{i}": i for i in range(num_classes)}

def _preprocess_data_gpu(data_batch: torch.Tensor, mag_prepro: bool = False, phase_prepro: bool = False) -> torch.Tensor:
    if len(data_batch.size()) == 3: 
        data_batch = data_batch.unsqueeze(0)
        
    mag_batch, phase_batch = data_batch[:, 0:1, :, :], data_batch[:, 1:2, :, :]

    # 1. Binary Mask
    thresholds = kornia.filters.otsu_threshold(mag_batch)[1].view(-1, 1, 1, 1)
    mask = (mag_batch >= thresholds).float()
    mask = kornia.morphology.opening(mask, torch.ones(3, 3, device=mask.device))

    processed_mag = mag_batch * mask
    processed_phase = phase_batch * mask

    if mag_prepro:
        mag_min = processed_mag.amin(dim=(-2, -1), keepdim=True)
        mag_max = processed_mag.amax(dim=(-2, -1), keepdim=True)
        mag_norm = (processed_mag - mag_min) / (mag_max - mag_min + 1e-6)
        processed_mag = kornia.enhance.equalize_clahe(mag_norm) * mask

    if phase_prepro:
        safe_mask = mask.bool()
        if safe_mask.any():
            masked_phase = torch.masked_select(phase_batch, safe_mask)
            p30, p85 = torch.quantile(masked_phase, torch.tensor([0.30, 0.85], device=masked_phase.device))
        else:
            p30, p85 = 0.0, 1.0
        phase_rescaled = torch.clamp((phase_batch - p30) / (p85 - p30 + 1e-6), 0, 1)
        processed_phase = phase_rescaled * mask
    
    return torch.cat([processed_mag, processed_phase], dim=1)

def apply_mosaic_transform(input_tensor, base_size):
    resized = F.interpolate(input_tensor, size=base_size, mode='bilinear', align_corners=False)
    top_row = torch.cat([resized, resized], dim=3)
    return torch.cat([top_row, top_row], dim=2)

def reverse_mosaic_transform(output_tensor, target_size):
    h_half, w_half = output_tensor.shape[2] // 2, output_tensor.shape[3] // 2
    top_left = output_tensor[:, :, :h_half, :w_half]
    return F.interpolate(top_left, size=target_size, mode='bilinear', align_corners=False)

def decode_bitmask(bitmask: np.ndarray, num_classes: int) -> np.ndarray:
    multi_channel_mask = np.zeros((num_classes, bitmask.shape[0], bitmask.shape[1]), dtype=np.uint8)
    for i in range(1, num_classes + 1):
        multi_channel_mask[i-1][bitmask == i] = 1
    return multi_channel_mask

def calculate_dice_numpy(pred_one_hot: torch.Tensor, gt_one_hot: torch.Tensor, class_labels: dict) -> dict:
    pred_np = pred_one_hot.squeeze(0).cpu().numpy().astype(bool)
    gt_np = gt_one_hot.squeeze(0).cpu().numpy().astype(bool)
    scores = {}
    class_names = [name for name in class_labels.keys() if name != "background"]
    
    for i, class_name in enumerate(class_names):
        if i >= pred_np.shape[0]: break
        p, g = pred_np[i], gt_np[i]
        
        if not np.any(g) and not np.any(p): scores[class_name] = 1.0
        elif not np.any(g): scores[class_name] = 0.0 
        else:
            intersection = np.sum(p & g)
            scores[class_name] = (2.0 * intersection) / (np.sum(p) + np.sum(g) + 1e-8)

    lesion_indices = [i for i, name in enumerate(class_names) if "lesion" in name.lower()]
    if lesion_indices:
        p_lesion = np.zeros_like(pred_np[0], dtype=bool)
        g_lesion = np.zeros_like(gt_np[0], dtype=bool)
        for idx in lesion_indices:
            if idx < pred_np.shape[0]: p_lesion |= pred_np[idx]
            if idx < gt_np.shape[0]: g_lesion |= gt_np[idx]
        
        if not np.any(g_lesion) and not np.any(p_lesion): scores['lesion_ALL'] = 1.0
        elif not np.any(g_lesion): scores['lesion_ALL'] = 0.0
        else:
            intersection = np.sum(p_lesion & g_lesion)
            scores['lesion_ALL'] = (2.0 * intersection) / (np.sum(p_lesion) + np.sum(g_lesion) + 1e-8)
        
    return scores

def create_comparison_plot(base_image, preprocessed_phase, gt_mask, pred_mask, dice_scores, title_str, class_labels):
    colors = ['cyan', 'lime', 'red', 'yellow', 'magenta', 'orange']
    class_names = [name for name in class_labels.keys() if name != "background"]
    num_fg_classes = len(class_names)
    
    gt_decoded = decode_bitmask(gt_mask, num_fg_classes)
    pred_decoded = decode_bitmask(pred_mask, num_fg_classes)
    
    fig, axs = plt.subplots(3, 5, figsize=(28, 16))
    fig.suptitle(title_str, fontsize=20, y=0.98)

    axs[0, 0].imshow(base_image.T, cmap='gray'); axs[0,0].set_title("Mag")
    axs[0, 1].imshow(preprocessed_phase.T, cmap='gray'); axs[0,1].set_title("Phase")
    axs[0, 2].imshow(base_image.T, cmap='gray'); axs[0,2].set_title("GT All")
    axs[0, 3].imshow(base_image.T, cmap='gray'); axs[0,3].set_title("Pred All")
    
    for i in range(num_fg_classes):
        c = colors[i % len(colors)]
        if np.any(gt_decoded[i]): axs[0, 2].contour(gt_decoded[i].T, colors=[c], linewidths=1)
        if np.any(pred_decoded[i]): axs[0, 3].contour(pred_decoded[i].T, colors=[c], linewidths=1)

    axs[0, 4].axis('off')
    txt = "\n".join([f"{k}: {v:.4f}" for k,v in dice_scores.items()])
    axs[0, 4].text(0, 1, txt, va='top', transform=axs[0, 4].transAxes, fontsize=12)

    for i in range(min(num_fg_classes, 5)):
        c = colors[i % len(colors)]
        axs[1, i].imshow(base_image.T, cmap='gray')
        if np.any(gt_decoded[i]): axs[1, i].contourf(gt_decoded[i].T, colors=[c], alpha=0.4)
        axs[2, i].imshow(base_image.T, cmap='gray')
        if np.any(pred_decoded[i]): axs[2, i].contourf(pred_decoded[i].T, colors=[c], alpha=0.4)

    for ax in axs.ravel(): ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

# --- 3. AGGREGATION LOGIC ---

def aggregate_experiment_results(output_folder_root: Path, folds: list, experiment_name: str):
    print(f"\n📊 Aggregating results for '{experiment_name}' across folds: {folds}")
    all_fold_stats = []
    
    for fold in folds:
        fold_dir_name = f"{experiment_name}_fold{fold}"
        fold_path = output_folder_root / fold_dir_name
        if not fold_path.exists(): continue

        csv_files = list(fold_path.rglob("metrics_fold_summary.csv"))
        if not csv_files: continue
            
        df = pd.read_csv(csv_files[0], index_col=0)
        if 'Mean' in df.index:
            stats = df.loc['Mean'].to_dict()
            stats['fold'] = fold
            all_fold_stats.append(stats)
    
    if not all_fold_stats:
        print("❌ No data found to aggregate.")
        return

    df_results = pd.DataFrame(all_fold_stats).set_index('fold')
    exp_mean = df_results.mean()
    exp_std = df_results.std()

    report_file = output_folder_root / f"{experiment_name}_FINAL_REPORT.txt"
    with open(report_file, 'w') as f:
        f.write(f"EXPERIMENT REPORT: {experiment_name}\n")
        f.write("="*50 + "\n\n")
        f.write("--- Per Fold Dice Averages ---\n")
        f.write(df_results.to_string(float_format="%.4f"))
        f.write("\n\n")
        f.write("--- Overall Experiment (Mean ± Std) ---\n")
        for col in df_results.columns:
            f.write(f"{col:<15}: {exp_mean[col]:.4f} ± {exp_std[col]:.4f}\n")
    
    print(f"✅ Aggregation Report saved to: {report_file}")


# --- 4. EXECUTION MODES ---

def run_volume_inference(args, exp_settings):
    # Setup
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_FOLDER, OUTPUT_FOLDER = Path(args.model_folder), Path(args.output_folder)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Debug log file
    debug_log = OUTPUT_FOLDER / "debug_inference_log.txt"
    if args.debug and debug_log.exists(): os.remove(debug_log)
    
    predictor = nnUNetPredictor(device=DEVICE)
    predictor.initialize_from_trained_model_folder(str(MODEL_FOLDER), use_folds=(args.fold,), checkpoint_name='checkpoint_best.pth')
    network = predictor.network.to(DEVICE).eval()
    
    mag_nii = nib.load(args.volume_mag_path)
    phase_nii = nib.load(args.volume_phase_path)
    mag_data, phase_data = mag_nii.get_fdata(), phase_nii.get_fdata()
    
    all_preds = []
    print(f"🚀 Processing Volume with settings: {exp_settings}")
    
    with torch.no_grad():
        for z in tqdm(range(mag_data.shape[1])):
            # 1. Create & Transpose (1, C, W, H)
            slice_tensor = torch.stack([
                torch.from_numpy(mag_data[:, z, :]), 
                torch.from_numpy(phase_data[:, z, :])
            ]).float().unsqueeze(0).transpose(3, 2).to(DEVICE)
            
            original_slice_shape = slice_tensor.shape[2:]
            
            if exp_settings['mag_only']: slice_tensor[:, 1] = slice_tensor[:, 0]
            
            # 2. Pad
            padded_tensor, transform_info = resize_and_pad_to_size(slice_tensor, exp_settings['input_patch_size'])
            
            # 3. Normalize (Explicit Scaling)
            padded_tensor[:, 0] = (padded_tensor[:, 0] - padded_tensor[:, 0].mean()) / (padded_tensor[:, 0].std() + 1e-8) * 0.924017
            padded_tensor[:, 1] = (padded_tensor[:, 1] - padded_tensor[:, 1].mean()) / (padded_tensor[:, 1].std() + 1e-8) * 0.924017

            # Debug
            if args.debug and z < 3: # Log first 3 slices
                 log_tensor_stats(f"Vol Slice {z} Normalized (Mag)", padded_tensor[0, 0], debug_log)

            # 4. Preprocess
            prep = _preprocess_data_gpu(padded_tensor, exp_settings['mag_prepro'], exp_settings['phase_prepro'])
            
            if exp_settings['do_mosaic']: 
                inp = apply_mosaic_transform(prep, exp_settings['input_patch_size'])
            else: 
                inp = prep
                
            logits = network(inp)
            probs = torch.sigmoid(logits)
            
            if exp_settings['do_mosaic']: 
                probs_unpadded = reverse_mosaic_transform(probs, (mag_data.shape[0], mag_data.shape[2])) # Note: Dims might need transpose checking for mosaic
            else: 
                probs_restored = reverse_resize_and_pad(probs, transform_info, original_slice_shape)

            # 5. Transpose Back (1, C, H, W)
            probs_final = probs_unpadded.transpose(3, 2)
                
            all_preds.append(torch.argmax(probs_final[0], dim=0).cpu().numpy().astype(np.uint8))
            
    # Save
    out_nii = nib.Nifti1Image(np.stack(all_preds, axis=1), mag_nii.affine)
    nib.save(out_nii, OUTPUT_FOLDER / "volume_seg.nii.gz")
    print("✅ Volume Saved.")

def run_plot_inference(args, exp_settings):
    target_patch_size = (224, 320)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    MODEL_FOLDER, OUTPUT_FOLDER = Path(args.model_folder), Path(args.output_folder)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Debug log file
    debug_log = OUTPUT_FOLDER / "debug_inference_log.txt"
    if args.debug and debug_log.exists(): os.remove(debug_log)

    if args.test_set_path:
        NNUNET_RAW_DATASET_PATH = Path(args.test_set_path)
        DATASET_NAME = NNUNET_RAW_DATASET_PATH.name
        RAW_DIR = NNUNET_RAW_DATASET_PATH.parent
        SLICED_DIR = NNUNET_RAW_DATASET_PATH.parent.parent / 'data_slice'
        manifest_path = RAW_DIR / DATASET_NAME / "inference_manifest.json"
        with open(manifest_path) as f: manifest = json.load(f)
        case_ids = list(manifest.keys())
        print(f"🧠 Test Mode: {len(case_ids)} cases.")
    else:
        PATH_PROCESSED = Path(args.path_processed)
        DATASET_NAME = MODEL_FOLDER.parent.name
        RAW_DIR = PATH_PROCESSED / "nnunet_raw"
        SLICED_DIR = PATH_PROCESSED / "data_slice"
        SPLIT_FILE = PATH_PROCESSED / "nnunet_preprocessed" / DATASET_NAME / 'splits_final.json'
        
        with open(SPLIT_FILE) as f: splits = json.load(f)
        case_ids = splits[args.fold]['val']
        with open(RAW_DIR / DATASET_NAME / "inference_manifest.json") as f: manifest = json.load(f)
        print(f"📝 Validation Mode (Fold {args.fold}): {len(case_ids)} cases.")

    predictor = nnUNetPredictor(device=DEVICE)
    predictor.initialize_from_trained_model_folder(str(MODEL_FOLDER), use_folds=(args.fold,), checkpoint_name='checkpoint_best.pth')
    network = predictor.network.to(DEVICE).eval()
    
    class_labels = get_class_configuration(MODEL_FOLDER)
    num_fg_classes = len(class_labels) - 1
    metric_data = []

    for case_id in tqdm(case_ids):
        original_name_key = manifest.get(case_id)
        if not original_name_key: continue
        try:
            subject_id, rest = original_name_key.split('/', 1)
            name_key, slice_id = rest.split('_slice-')
        except ValueError: continue

        mag_path = SLICED_DIR / subject_id / 'anat' / f"{name_key}_part-mag_slice-{slice_id}.nii.gz"
        phase_path = SLICED_DIR / subject_id / 'anat' / f"{name_key}_part-phase_slice-{slice_id}.nii.gz"
        label_path = RAW_DIR / DATASET_NAME / 'labelsTr' / f"{case_id}.nii.gz"
        
        if not all(p.exists() for p in [mag_path, phase_path, label_path]): continue

        # Load Raw
        mag_nii = nib.load(mag_path)
        original_shape = mag_nii.shape[:2]
        
        # 1. Create & Transpose: (H,W) -> (1, C, W, H)
        img_tensor = torch.cat([
            torch.from_numpy(mag_nii.get_fdata().squeeze()).float().unsqueeze(0),
            torch.from_numpy(nib.load(phase_path).get_fdata().squeeze()).float().unsqueeze(0)
        ], dim=0).unsqueeze(0).transpose(3, 2).to(DEVICE)
        print("img tensor: ", img_tensor.size())
        
        original_slice_shape = img_tensor.shape[2:]
        
        if exp_settings['mag_only']: img_tensor[:, 1] = img_tensor[:, 0]

        # 2. Pad
        padded_tensor, padding_info = resize_and_pad_to_size(img_tensor, target_patch_size)

        # 3. Normalize (Exact Formula)
        padded_tensor[:, 0] = (padded_tensor[:, 0] - padded_tensor[:, 0].mean()) / (padded_tensor[:, 0].std() + 1e-8) * 0.924017
        padded_tensor[:, 1] = (padded_tensor[:, 1] - padded_tensor[:, 1].mean()) / (padded_tensor[:, 1].std() + 1e-8) * 0.924017
        print("padded_tensor: ", padded_tensor.size(), case_id)
        # Debug
        if args.debug:
            log_tensor_stats(f"Normalized Input (Mag) - {case_id}", padded_tensor[0, 0], debug_log)
            log_tensor_stats(f"Normalized Input (Phase) - {case_id}", padded_tensor[0, 1], debug_log)

        # 4. Preprocess
        prep_tensor = _preprocess_data_gpu(padded_tensor, exp_settings['mag_prepro'], exp_settings['phase_prepro'])
        
        if exp_settings['do_mosaic']: 
            input_tensor = apply_mosaic_transform(prep_tensor, exp_settings['input_patch_size'])
        else:
            input_tensor = prep_tensor
            
        with torch.no_grad():
            logits = network(input_tensor)
            probs = torch.sigmoid(logits)
            
        if exp_settings['do_mosaic']:
            probs_unpadded = reverse_mosaic_transform(probs, original_shape) # Warning: Mosaic logic needs review for transposed dims
        else:
            probs_unpadded = reverse_resize_and_pad(probs, padding_info, original_slice_shape)
        
        # 5. Transpose Back: (1, C, W, H) -> (1, C, H, W)
        probs_final = probs_unpadded.transpose(3, 2)
            
        # Metrics
        pred_onehot = (probs_final[:, 1:] > 0.5)
        gt_mask = nib.load(label_path).get_fdata().squeeze().astype(np.uint8)
        gt_onehot = torch.from_numpy(decode_bitmask(gt_mask, num_fg_classes)).unsqueeze(0).to(DEVICE)
        
        scores = calculate_dice_numpy(pred_onehot, gt_onehot, class_labels)
        
        row = {'case_id': case_id}
        row.update(scores)
        metric_data.append(row)

        pred_mask_plot = np.argmax(probs_final.squeeze(0).cpu().numpy(), axis=0).astype(np.uint8)
        
        # Note: We must also transpose back the input image for plotting so it looks like the original
        plot_img = prep_tensor.transpose(3, 2)
        
        fig = create_comparison_plot(
            plot_img.squeeze(0)[0].cpu().numpy(), 
            plot_img.squeeze(0)[1].cpu().numpy(),
            gt_mask, pred_mask_plot, scores, f"{case_id} ({args.experiment})", class_labels
        )
        fig.savefig(OUTPUT_FOLDER / f"{case_id}_comparison.png", dpi=100, bbox_inches='tight')
        plt.close(fig)

    if metric_data:
        df = pd.DataFrame(metric_data).set_index('case_id')
        df.to_csv(OUTPUT_FOLDER / "metrics_per_case.csv")
        summary = df.agg(['mean', 'std'])
        summary.index = ['Mean', 'Std']
        summary.to_csv(OUTPUT_FOLDER / "metrics_fold_summary.csv")
        print(f"✅ Saved CSV tables to: {OUTPUT_FOLDER}")
    else:
        print("⚠️ No metrics collected.")

# --- MAIN ---

def str_to_bool(value):
    if isinstance(value, bool): return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'): return True
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['plot', 'volume', 'aggregate'])
    parser.add_argument('--experiment', type=str, default='exp_base')
    parser.add_argument('--model_folder', type=str, default=None)
    parser.add_argument('--output_folder', type=str, default=None)
    
    parser.add_argument('--folds_to_aggregate', nargs='+', type=int, help="List of folds")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--test_set_path', type=str)
    parser.add_argument('--path_processed', type=str)
    parser.add_argument('--use_split_file', type=str_to_bool, default=True)
    parser.add_argument('--volume_mag_path', type=str)
    parser.add_argument('--volume_phase_path', type=str)
    
    # NEW DEBUG FLAG
    parser.add_argument('--debug', action='store_true', help="Enable verbose tensor stat logging")

    args = parser.parse_args()

    if args.mode == 'aggregate':
        if not args.output_folder or not args.folds_to_aggregate:
            parser.error("Aggregate mode requires --output_folder and --folds_to_aggregate")
        aggregate_experiment_results(Path(args.output_folder), args.folds_to_aggregate, args.experiment)
    else:
        if not args.model_folder: parser.error("Model folder required.")
        if not args.output_folder: args.output_folder = str(Path(args.model_folder) / "inference_outputs")
        
        settings = get_experiment_settings(args.experiment)
        
        if args.mode == 'volume':
            if not args.volume_mag_path: parser.error("Volume mode requires file paths.")
            run_volume_inference(args, settings)
        elif args.mode == 'plot':
            if not args.path_processed and not args.test_set_path: parser.error("Plot mode requires dataset path.")
            run_plot_inference(args, settings)


# import torch
# import torch.nn as nn
# import argparse
# import os
# import shutil
# from pathlib import Path
# import kornia
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
# from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy

# # --- 1. CUSTOM PREPROCESSING (From your Trainer) ---
# # We verify this matches your nnUnetTrainerWandb.py logic exactly.

# def _preprocess_data_gpu(data_batch: torch.Tensor) -> torch.Tensor:
#     """
#     Applies custom preprocessing DIRECTLY ON THE GPU using Kornia.
#     Expected Input: (B, C, H, W) - Z-Scored by standard nnU-Net before reaching here.
#     """
#     # Ensure correct dimensions
#     if len(data_batch.size()) == 3: 
#         data_batch = data_batch.unsqueeze(0)
        
#     # Slicing channels (assuming 0=Mag, 1=Phase based on your code)
#     mag_batch = data_batch[:, 0:1, :, :]
#     phase_batch = data_batch[:, 1:2, :, :]

#     # 1. Create binary mask from magnitude (Otsu)
#     # Note: Kornia Otsu expects normalized values 0-1 usually, but works on distributions.
#     # Since input is Z-scored, we rely on the implementation being robust or the distribution being bimodal.
#     thresholds = kornia.filters.otsu_threshold(mag_batch)[1]
    
#     # Reshape thresholds to broadcast: (B, 1, 1, 1)
#     thresholds = thresholds.view(-1, 1, 1, 1)
#     mask = (mag_batch >= thresholds).float()

#     # 2. Apply morphological opening
#     # Use kernel on same device as mask
#     kernel = torch.ones(3, 3, device=mask.device) 
#     mask = kornia.morphology.opening(mask, kernel)

#     # 3. Phase Prepro (Percentile Scaling)
#     safe_mask = mask.bool()
#     if safe_mask.any():
#         masked_phase = torch.masked_select(phase_batch, safe_mask)
#         p30 = torch.quantile(masked_phase, 0.30)
#         p85 = torch.quantile(masked_phase, 0.85)
#     else:
#         p30, p85 = 0.0, 1.0

#     phase_rescaled = torch.clamp((phase_batch - p30) / (p85 - p30 + 1e-6), 0, 1)
    
#     # Masking
#     processed_mag = mag_batch * mask
#     processed_phase = phase_rescaled * mask
    
#     # Recombine
#     return torch.cat([processed_mag, processed_phase], dim=1)

# # --- 2. NETWORK WRAPPER ---
# # This is the "Magic Fix". We wrap the trained network so that every time 
# # nnU-Net tries to predict a sliding window tile, we force our preprocessing first.

# class PreprocessingHook(nn.Module):
#     def __init__(self, original_network):
#         super().__init__()
#         self.original_network = original_network

#     def forward(self, x):
#         # x is the sliding window tile (B, C, H, W)
#         # It has already been Z-scored by nnU-Net's standard loading pipeline.
#         # We apply your specific GPU transforms here.
#         x_preprocessed = _preprocess_data_gpu(x)
#         return self.original_network(x_preprocessed)
    
#     # Proxy other attributes to the original network (e.g., deep_supervision status)
#     def __getattr__(self, name):
#         try:
#             return super().__getattr__(name)
#         except AttributeError:
#             return getattr(self.original_network, name)

# # --- 3. MAIN INFERENCE PIPELINE ---

# def run_standard_inference(args):
#     # Setup Paths
#     model_folder = Path(args.model_folder)
#     output_folder = Path(args.output_folder)
#     output_folder.mkdir(exist_ok=True, parents=True)

#     # 1. Initialize Predictor
#     print(f"🚀 Initializing nnUNetPredictor...")
#     predictor = nnUNetPredictor(
#         tile_step_size=0.5,
#         use_gaussian=True,
#         use_mirroring=False, # Faster, enable if you want slightly better precision
#         device=torch.device('cuda', args.gpu_id),
#         verbose=True
#     )
    
#     # 2. Load Model Weights
#     predictor.initialize_from_trained_model_folder(
#         str(model_folder),
#         use_folds=(args.fold,),
#         checkpoint_name='checkpoint_best.pth'
#     )

#     # 3. INJECT CUSTOM PREPROCESSING
#     # We replace the internal network with our wrapper
#     print(f"💉 Injecting Custom Otsu/Phase Preprocessing Hook...")
#     predictor.network = PreprocessingHook(predictor.network)

#     # 4. Prepare Inputs
#     # nnU-Net expects a list of lists: [[case1_channel0, case1_channel1], [case2...]]
#     # Since we are doing a single volume inference here:
#     input_files = [[args.volume_mag_path, args.volume_phase_path]]
    
#     # Prepare Output Filename (nnU-Net outputs based on input name usually, but we can manage it)
#     # We will use a temp generic name and rename it later to match your old script's output
#     temp_case_id = "inference_case"
#     output_files = [str(output_folder / f"{temp_case_id}.nii.gz")]

#     # 5. Run Prediction
#     print(f"🧠 Running Inference on: {args.volume_mag_path}")
#     predictor.predict_from_files(
#         list_of_lists_or_source_folder=input_files,
#         output_folder_or_list_of_truncated_output_files=output_files,
#         save_probabilities=False,
#         overwrite=True,
#         num_processes_preprocessing=2,
#         num_processes_segmentation_export=2,
#         folder_with_segs_from_prev_stage=None,
#         num_parts=1,
#         part_id=0
#     )

#     # 6. Cleanup / Rename to match expected output "volume_seg.nii.gz"
#     generated_file = output_folder / f"{temp_case_id}.nii.gz"
#     final_file = output_folder / "volume_seg.nii.gz"
    
#     if generated_file.exists():
#         shutil.move(str(generated_file), str(final_file))
#         print(f"✅ Success! Segmentation saved to: {final_file}")
#     else:
#         print("❌ Error: Output file was not generated.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_folder', type=str, required=True, help="Path to the specific model fold folder")
#     parser.add_argument('--output_folder', type=str, required=True)
#     parser.add_argument('--volume_mag_path', type=str, required=True, help="Path to Magnitude NIfTI")
#     parser.add_argument('--volume_phase_path', type=str, required=True, help="Path to Phase NIfTI")
#     parser.add_argument('--fold', type=int, default=0)
#     parser.add_argument('--gpu_id', type=int, default=0)
    
#     args = parser.parse_args()
    
#     run_standard_inference(args)