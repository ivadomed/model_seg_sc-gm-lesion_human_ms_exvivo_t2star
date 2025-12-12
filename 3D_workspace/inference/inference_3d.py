import argparse
import os
import sys
import json
import re
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from collections import defaultdict
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, isfile, load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# =================================================================================================
# HELPER FUNCTIONS
# =================================================================================================

def normalize_string(s: str) -> str:
    s = os.path.basename(s)
    s = s.replace('.nii.gz', '').replace('.nii', '')
    s = s.replace('_part-mag', '').replace('part-mag', '') 
    return s.strip()

def parse_manifests(manifest_2d_path: str, manifest_3d_path: str):
    print(f"📂 Parsing manifests...")
    with open(manifest_3d_path, 'r') as f:
        man_3d_list = json.load(f)
        
    map_orig_to_nnunet = {}
    for entry in man_3d_list:
        nn_id = entry['nnunet_id']
        orig_fname = entry['original_filename']
        subj = entry['subject']
        norm_fname = normalize_string(orig_fname)
        key = f"{subj}/{norm_fname}"
        map_orig_to_nnunet[key] = nn_id

    with open(manifest_2d_path, 'r') as f:
        man_2d = json.load(f)

    nnunet_to_2d_map = defaultdict(list)
    matches_found = 0
    for case_2d_id, path_str in man_2d.items():
        match = re.search(r"(.*)_slice-(\d+)$", path_str)
        if match:
            full_vol_path = match.group(1) 
            slice_idx = int(match.group(2))
            if '/' in full_vol_path:
                subj_part, file_part = full_vol_path.split('/', 1)
                norm_file_part = normalize_string(file_part)
                query_key = f"{subj_part}/{norm_file_part}"
                if query_key in map_orig_to_nnunet:
                    nn_id = map_orig_to_nnunet[query_key]
                    nnunet_to_2d_map[nn_id].append({
                        'case_2d_id': case_2d_id,
                        'slice_idx': slice_idx,
                        'orig_vol_name': full_vol_path
                    })
                    matches_found += 1
    
    print(f"✅ Successfully linked {matches_found} 2D slices to 3D volumes.")
    return nnunet_to_2d_map

def compute_dice(pred, gt, labels=[1, 2, 3, 4]):
    """Computes Dice per case (returns dict of floats)."""
    metrics = {}
    for label in labels:
        p_bin = (pred == label)
        g_bin = (gt == label)
        
        if g_bin.sum() == 0:
            dice = np.nan if p_bin.sum() == 0 else 0.0
        else:
            inter = np.logical_and(p_bin, g_bin).sum()
            union = p_bin.sum() + g_bin.sum()
            dice = (2.0 * inter) / (union + 1e-8)
        metrics[f'Dice_{label}'] = dice
    return metrics

def compute_counts(pred, gt, labels=[1, 2, 3, 4]):
    """Computes raw TP, FP, FN counts for Global Dice calculation."""
    counts = {}
    for label in labels:
        p_bin = (pred == label)
        g_bin = (gt == label)
        
        tp = np.logical_and(p_bin, g_bin).sum()
        fp = np.logical_and(p_bin, ~g_bin).sum()
        fn = np.logical_and(~p_bin, g_bin).sum()
        
        counts[label] = {'TP': tp, 'FP': fp, 'FN': fn}
    return counts

# =================================================================================================
# VISUALIZATION FUNCTIONS
# =================================================================================================

COLORS = ['black', 'cyan', 'lime', 'red', 'yellow', 'magenta', 'orange', 'purple']

def decode_bitmask_to_channels(bitmask: np.ndarray, num_classes: int = 4) -> np.ndarray:
    if bitmask is None: return None
    h, w = bitmask.shape
    multi_channel = np.zeros((num_classes, h, w), dtype=np.uint8)
    for i in range(1, num_classes + 1): 
        multi_channel[i-1] = (bitmask == i).astype(np.uint8)
    return multi_channel

def _plot_contours_on_ax(ax, img, mask_int, title, num_classes=4):
    if img.ndim == 2:
        ax.imshow(img, cmap='gray')
    else:
        ax.imshow(img[0], cmap='gray')
        
    ax.set_title(title)
    ax.axis('off')
    
    if mask_int is not None:
        masks_mc = decode_bitmask_to_channels(mask_int, num_classes)
        for c_idx in range(num_classes):
            color_idx = c_idx + 1
            color = COLORS[color_idx % len(COLORS)]
            mask_c = masks_mc[c_idx]
            if np.any(mask_c):
                ax.contourf(mask_c, levels=[0.5, 1.5], colors=[color], alpha=0.3)
                ax.contour(mask_c, levels=[0.5], colors=[color], linewidths=1.0)

def plot_ortho_view(img_data_3d, pred_data_3d, gt_data_3d, case_id, output_path):
    shape = img_data_3d.shape 
    img = img_data_3d[0] if img_data_3d.ndim == 4 else img_data_3d
    c_x, c_y, c_z = img.shape[2]//2, img.shape[1]//2, img.shape[0]//2
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"3D View: {case_id}", fontsize=16)
    
    slices = [
        (img[c_z, :, :], gt_data_3d[c_z, :, :] if gt_data_3d is not None else None, pred_data_3d[c_z, :, :], "Axial"),
        (img[:, c_y, :], gt_data_3d[:, c_y, :] if gt_data_3d is not None else None, pred_data_3d[:, c_y, :], "Coronal"),
        (img[:, :, c_x], gt_data_3d[:, :, c_x] if gt_data_3d is not None else None, pred_data_3d[:, :, c_x], "Sagittal")
    ]
    
    for i, (im_s, gt_s, pred_s, title) in enumerate(slices):
        _plot_contours_on_ax(axs[0, i], im_s, gt_s, f"{title} - GT", num_classes=4)
        _plot_contours_on_ax(axs[1, i], im_s, pred_s, f"{title} - Pred", num_classes=4)
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_2d_slice_eval(vol_slice_mag, vol_slice_phase, gt_2d, pred_slice, case_2d_id, output_path):
    cols = 4 if vol_slice_phase is not None else 3
    fig, axs = plt.subplots(1, cols, figsize=(6*cols, 6))
    
    axs[0].imshow(vol_slice_mag, cmap='gray')
    axs[0].set_title(f"{case_2d_id}\nMagnitude")
    axs[0].axis('off')

    idx_offset = 1
    if vol_slice_phase is not None:
        axs[1].imshow(vol_slice_phase, cmap='gray')
        axs[1].set_title("Phase")
        axs[1].axis('off')
        idx_offset = 2

    _plot_contours_on_ax(axs[idx_offset], vol_slice_mag, gt_2d, "Ground Truth", num_classes=4)
    _plot_contours_on_ax(axs[idx_offset+1], vol_slice_mag, pred_slice, "Prediction", num_classes=4)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# =================================================================================================
# MAIN CLASS
# =================================================================================================

class Inference3DRunner:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
        self.use_tta = args.use_tta.lower() in ['true', '1', 'yes']
        self.ensemble = args.ensemble.lower() in ['true', '1', 'yes']
        
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=self.use_tta,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            allow_tqdm=True
        )
        
        self.io = SimpleITKIO()
        self.nnunet_to_2d_map = parse_manifests(args.manifest_2d, args.manifest_3d)
        
        # Load Splits for Validation
        self.splits = []
        if args.mode == 'validation':
            possible_paths = [
                join(args.model_folder, 'splits_final.json'),
                join(os.path.dirname(args.model_folder), 'splits_final.json'),
                join(args.input_path_raw.replace('nnUNet_raw', 'nnUNet_preprocessed'), 'splits_final.json')
            ]
            for p in possible_paths:
                if isfile(p):
                    print(f"📂 Loaded splits from: {p}")
                    self.splits = load_json(p)
                    break
            if not self.splits:
                print("⚠️ Warning: splits_final.json not found. Validation mode might process all files.")

    def run(self):
        if self.args.mode == 'validation':
            inp_dir = join(self.args.input_path_raw, 'imagesTr')
            gt_dir_3d = join(self.args.input_path_raw, 'labelsTr')
        else:
            inp_dir = join(self.args.input_path_raw, 'imagesTs')
            gt_dir_3d = join(self.args.input_path_raw, 'labelsTs') 
        
        if not os.path.isdir(inp_dir):
            raise ValueError(f"Input directory not found: {inp_dir}")

        all_files = subfiles(inp_dir, suffix='.nii.gz', join=False)
        all_cases = sorted(list(set([f.split('_0000.nii.gz')[0] for f in all_files if '_0000.nii.gz' in f])))
        print(f"Found {len(all_cases)} total cases in directory.")

        if self.ensemble:
            inference_loops = [("ensemble_predictions", self.args.folds)]
        else:
            inference_loops = [(f"fold_{f}", [f]) for f in self.args.folds]

        # Stats Accumulators for Final Summary (Cross-Fold)
        cross_fold_metrics_2d_means = []
        cross_fold_metrics_3d_means = []
        cross_fold_global_2d = []
        cross_fold_global_3d = []

        # --- Main Loop ---
        for folder_name, current_folds in inference_loops:
            print(f"\n========================================================")
            print(f"🚀 Running: {folder_name} (Folds: {current_folds})")
            print(f"========================================================")

            target_cases = []
            if self.args.mode == 'validation' and self.splits:
                val_keys = set()
                for f in current_folds:
                    if f < len(self.splits):
                        val_keys.update(self.splits[f]['val'])
                target_cases = [k for k in all_cases if k in val_keys]
                print(f"   Subset: {len(target_cases)} cases (Validation split for folds {current_folds})")
            else:
                target_cases = all_cases
                print(f"   Subset: Processing ALL {len(target_cases)} cases.")

            if not target_cases:
                print("   ⚠️ No cases to process for this loop. Skipping.")
                continue

            self.predictor.initialize_from_trained_model_folder(
                self.args.model_folder,
                use_folds=current_folds,
                checkpoint_name='checkpoint_best.pth'
            )

            current_output_dir = join(self.args.output_root, folder_name)
            dir_pred_3d = join(current_output_dir, "predictions_3d")
            dir_viz_3d = join(current_output_dir, "visualizations_3d")
            dir_viz_2d = join(current_output_dir, "visualizations_2d_slices")
            
            maybe_mkdir_p(dir_pred_3d)
            maybe_mkdir_p(dir_viz_3d)
            maybe_mkdir_p(dir_viz_2d)

            # Metrics for this fold
            metrics_3d_list = []
            metrics_2d_list = []
            
            # Global Count Accumulators (Per Fold)
            global_counts_3d = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
            global_counts_2d = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})

            # 4. Processing Loop
            for case_id in target_cases:
                print(f"   Processing: {case_id}")
                
                # Load Data
                fpath_c0 = join(inp_dir, f"{case_id}_0000.nii.gz")
                fpath_c1 = join(inp_dir, f"{case_id}_0001.nii.gz")
                files_to_read = [fpath_c0]
                if isfile(fpath_c1): files_to_read.append(fpath_c1)
                img_list, props = self.io.read_images(files_to_read)
                
                # Channel Duplication Logic
                expected_channels = len(self.predictor.dataset_json['channel_names'].keys())
                if self.args.experiment == "mag_only" and expected_channels == 2 and img_list.shape[0] == 1:
                    img_list = np.concatenate([img_list, img_list], axis=0)

                # Inference
                output = self.predictor.predict_single_npy_array(img_list, props, None, None, True)
                if isinstance(output, (list, tuple)):
                    pred_seg = output[0].astype(np.uint8)
                else:
                    pred_seg = np.argmax(output, axis=0).astype(np.uint8)

                if pred_seg.ndim == 4 and pred_seg.shape[0] == 1:
                    pred_seg = pred_seg[0]

                if self.args.save_predictions:
                    self.io.write_seg(pred_seg, join(dir_pred_3d, f"{case_id}.nii.gz"), props)

                # --- 3D EVALUATION ---
                gt_path = join(gt_dir_3d, f"{case_id}.nii.gz")
                gt_3d = None
                if isfile(gt_path):
                    gt_3d, _ = self.io.read_images([gt_path])
                    gt_3d = gt_3d[0]
                    
                    # Case-wise Dice
                    m_3d = compute_dice(pred_seg, gt_3d)
                    m_3d['Case_ID'] = case_id
                    metrics_3d_list.append(m_3d)
                    
                    # Global Counts Accumulation
                    c_3d = compute_counts(pred_seg, gt_3d)
                    for lbl, counts in c_3d.items():
                        global_counts_3d[lbl]['TP'] += counts['TP']
                        global_counts_3d[lbl]['FP'] += counts['FP']
                        global_counts_3d[lbl]['FN'] += counts['FN']
                    
                    plot_ortho_view(img_list, pred_seg, gt_3d, case_id, join(dir_viz_3d, f"{case_id}_ortho.png"))
                else:
                    plot_ortho_view(img_list, pred_seg, None, case_id, join(dir_viz_3d, f"{case_id}_ortho.png"))

                # --- 2D EVALUATION ---
                if case_id in self.nnunet_to_2d_map:
                    linked_slices = self.nnunet_to_2d_map[case_id]
                    for item in linked_slices:
                        case_2d_id = item['case_2d_id']
                        slice_idx = item['slice_idx']
                        label_fname = re.sub(r'_\d{4}$', '', case_2d_id)
                        gt_2d_path = join(self.args.gt_path_2d, f"{label_fname}.nii.gz")
                        if not isfile(gt_2d_path):
                             gt_2d_path = join(self.args.gt_path_2d, f"{case_2d_id}.nii.gz")
                            
                        if isfile(gt_2d_path):
                            gt_2d, _ = self.io.read_images([gt_2d_path])
                            gt_2d = gt_2d[0]
                            if gt_2d.ndim == 3: gt_2d = gt_2d[0]
                            
                            if 0 <= slice_idx < pred_seg.shape[0]:
                                pred_slice = pred_seg[slice_idx, :, :]
                                input_mag = img_list[0, slice_idx, :, :]
                                input_phase = img_list[1, slice_idx, :, :] if img_list.shape[0] > 1 else None

                                # Case-wise Dice
                                m_2d = compute_dice(pred_slice, gt_2d)
                                m_2d['Case_2D_ID'] = case_2d_id
                                m_2d['NNUNet_3D_ID'] = case_id
                                m_2d['Slice_Index'] = slice_idx
                                metrics_2d_list.append(m_2d)
                                
                                # Global Counts Accumulation
                                c_2d = compute_counts(pred_slice, gt_2d)
                                for lbl, counts in c_2d.items():
                                    global_counts_2d[lbl]['TP'] += counts['TP']
                                    global_counts_2d[lbl]['FP'] += counts['FP']
                                    global_counts_2d[lbl]['FN'] += counts['FN']
                                
                                plot_2d_slice_eval(input_mag, input_phase, gt_2d, pred_slice, case_2d_id, join(dir_viz_2d, f"{case_2d_id}.png"))

            # --- PROCESS FOLD RESULTS ---
            
            # 1. Save Case-Wise Metrics
            if metrics_3d_list:
                df_3d = pd.DataFrame(metrics_3d_list)
                df_3d.to_csv(join(current_output_dir, "metrics_3d_casewise.csv"), index=False)
                # Store Mean for Cross-Fold summary
                mean_3d = df_3d.mean(numeric_only=True).to_dict()
                mean_3d['Fold'] = folder_name
                cross_fold_metrics_3d_means.append(mean_3d)

            if metrics_2d_list:
                df_2d = pd.DataFrame(metrics_2d_list)
                df_2d.to_csv(join(current_output_dir, "metrics_2d_casewise.csv"), index=False)
                # Store Mean for Cross-Fold summary
                mean_2d = df_2d.mean(numeric_only=True).to_dict()
                mean_2d['Fold'] = folder_name
                cross_fold_metrics_2d_means.append(mean_2d)

            # 2. Compute and Save Global Dice (Aggregated Counts)
            def save_global_metrics(global_counts, filename):
                data = []
                for lbl, c in global_counts.items():
                    denom = (2 * c['TP']) + c['FP'] + c['FN']
                    dice = (2 * c['TP']) / denom if denom > 0 else 0.0
                    entry = {'Label': lbl, 'Global_Dice': dice, 'TP': c['TP'], 'FP': c['FP'], 'FN': c['FN']}
                    data.append(entry)
                if data:
                    df = pd.DataFrame(data)
                    df.to_csv(join(current_output_dir, filename), index=False)
                    return {f"Dice_{x['Label']}": x['Global_Dice'] for x in data} # Return simplified dict for history
                return {}

            g_dice_3d = save_global_metrics(global_counts_3d, "metrics_3d_global.csv")
            g_dice_2d = save_global_metrics(global_counts_2d, "metrics_2d_global.csv")

            if g_dice_3d: 
                g_dice_3d['Fold'] = folder_name
                cross_fold_global_3d.append(g_dice_3d)
            if g_dice_2d:
                g_dice_2d['Fold'] = folder_name
                cross_fold_global_2d.append(g_dice_2d)

        # --- FINAL CROSS-FOLD SUMMARY (If not ensemble) ---
        if not self.ensemble:
            print("\nComputing Cross-Fold Statistics from saved CSVs on disk...")
            
            # 1. Identify all fold directories in output_root
            # We look for folders starting with "fold_"
            fold_dirs = [join(self.args.output_root, d) for d in os.listdir(self.args.output_root) if os.path.isdir(join(self.args.output_root, d)) and d.startswith('fold_')]
            print(f"   Found data for folds: {[os.path.basename(d) for d in fold_dirs]}")

            # Storage for aggregation
            agg_means_3d = []
            agg_means_2d = []
            agg_global_3d = []
            agg_global_2d = []

            for f_dir in fold_dirs:
                fold_name = os.path.basename(f_dir)
                
                # --- 3D Metrics ---
                p_3d_case = join(f_dir, "metrics_3d_casewise.csv")
                if isfile(p_3d_case):
                    df = pd.read_csv(p_3d_case)
                    mean_dict = df.mean(numeric_only=True).to_dict()
                    mean_dict['Fold'] = fold_name
                    agg_means_3d.append(mean_dict)
                
                p_3d_glob = join(f_dir, "metrics_3d_global.csv")
                if isfile(p_3d_glob):
                    df = pd.read_csv(p_3d_glob)
                    # Convert dataframe rows to a single dict for summary
                    glob_dict = {f"Dice_{row['Label']}": row['Global_Dice'] for _, row in df.iterrows()}
                    glob_dict['Fold'] = fold_name
                    agg_global_3d.append(glob_dict)

                # --- 2D Metrics ---
                p_2d_case = join(f_dir, "metrics_2d_casewise.csv")
                if isfile(p_2d_case):
                    df = pd.read_csv(p_2d_case)
                    mean_dict = df.mean(numeric_only=True).to_dict()
                    mean_dict['Fold'] = fold_name
                    agg_means_2d.append(mean_dict)

                p_2d_glob = join(f_dir, "metrics_2d_global.csv")
                if isfile(p_2d_glob):
                    df = pd.read_csv(p_2d_glob)
                    glob_dict = {f"Dice_{row['Label']}": row['Global_Dice'] for _, row in df.iterrows()}
                    glob_dict['Fold'] = fold_name
                    agg_global_2d.append(glob_dict)

            # Helper to save summary
            def save_summary(agg_list, suffix):
                if agg_list:
                    df_agg = pd.DataFrame(agg_list)
                    summary = pd.DataFrame({
                        'Mean': df_agg.mean(numeric_only=True),
                        'Std': df_agg.std(numeric_only=True)
                    })
                    out_name = join(self.args.output_root, f"summary_cross_fold_{suffix}.csv")
                    summary.to_csv(out_name)
                    print(f"   Saved: {os.path.basename(out_name)}")

            save_summary(agg_means_3d, "casewise_3d")
            save_summary(agg_means_2d, "casewise_2d")
            save_summary(agg_global_3d, "global_3d")
            save_summary(agg_global_2d, "global_2d")
        
        print("\nAll inference loops completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment', type=str, required=True)
    parser.add_argument('-model_folder', type=str, required=True)
    parser.add_argument('-mode', type=str, default='validation')
    parser.add_argument('-input_path_raw', type=str, required=True)
    parser.add_argument('-gt_path_2d', type=str, required=True)
    parser.add_argument('-manifest_2d', type=str, required=True)
    parser.add_argument('-manifest_3d', type=str, required=True)
    parser.add_argument('-output_root', type=str, required=True)
    parser.add_argument('-folds', nargs='+', type=int, default=[0])
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--use_tta', type=str, required=True)
    parser.add_argument('--ensemble', type=str, required=True)
    
    args = parser.parse_args()
    
    runner = Inference3DRunner(args)
    runner.run()