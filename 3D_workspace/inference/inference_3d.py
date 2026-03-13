import argparse
import os
import sys
import json
import re
import numpy as np
import torch
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import trimesh 
from typing import List, Tuple, Dict
from collections import defaultdict
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles, isfile, load_json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
import kornia
import torch.nn.functional as F
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion, generate_binary_structure
from skimage.measure import marching_cubes # Required for smoothness

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent.parent  # goes up from inference_scripts -> 3D_workspace -> Root
sys.path.append(str(project_root))
print("[Info] Project root added to sys.path:", project_root)

from helpers.metric_utils_2d import compute_surface_distances_2d, compute_binary_dice_2d, calculate_hd95
from helpers.preprocessing_utils import preprocess_gpu
from helpers.stats_utils import MetricTracker
from helpers.visualization_utils import save_ortho_view, save_2d_slice_viz

# =================================================================================================
# HELPER FUNCTIONS
# =================================================================================================
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



class SmoothnessEvaluator:
    """
    Evaluates geometric smoothness and slice-consistency metrics.
    Updated: Hardcoded for Z-axis stacking and added Sphericity.
    """
    def __init__(self, pixel_spacing=(1.0, 1.0, 1.0)):
        self.spacing = np.array(pixel_spacing)
        # Hardcoded to 0 (Z-axis) based on "horizontal artifacts in coronal/sagittal views"
        self.axis = 0 

    # def _get_mesh(self, mask):
    #     # Pad to ensure closed surface at boundaries (prevents open meshes)
    #     padded_mask = np.pad(mask, 1, mode='constant', constant_values=0)
    #     try:
    #         verts, faces, normals, values = marching_cubes(padded_mask, spacing=tuple(self.spacing))
    #         # process=True fixes normals and degenerate faces, critical for curvature
    #         return trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    #     except (ValueError, RuntimeError):
    #         return None

    def _inter_slice_dice(self, mask):
        # Calculates Dice between index i and i+1 along the Z-axis
        if mask.shape[0] < 2: return np.nan
        
        s1 = mask[:-1].astype(bool)
        s2 = mask[1:].astype(bool)
        
        inter = (s1 & s2).sum(axis=(1,2))
        total = s1.sum(axis=(1,2)) + s2.sum(axis=(1,2))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            d_vals = 2.0 * inter / total
            d_vals[total == 0] = 1.0 # Both empty = match
        
        valid_transitions = total > 0

        if valid_transitions.sum() == 0:
            return 1.0

        # Only average the scores where the object actually exists
        return np.mean(d_vals[valid_transitions])

    def _total_variation_z(self, mask):
        # TV along Axis 0 (Z)
        diff_z = np.abs(np.diff(mask.astype(np.float32), axis=0))
        total_pixels = np.sum(mask)
        return np.sum(diff_z) / total_pixels if total_pixels > 0 else 0.0

    # def _geometric_metrics(self, mesh, mask_volume):
    #     if mesh is None or mesh.is_empty:
    #         return np.nan, np.nan, np.nan
        
    #     voxel_vol = np.prod(self.spacing)
    #     vol_mm3 = np.sum(mask_volume) * voxel_vol
    #     area_mm2 = mesh.area
        
    #     # 1. SA:V Ratio (Lower is smoother)
    #     sa_v_ratio = area_mm2 / vol_mm3 if vol_mm3 > 0 else np.nan

    #     # 2. Sphericity (COMBINED METRIC: Combined Volume & Area)
    #     # Range: 0 to 1. 1.0 is a perfect sphere. Jaggies reduce this score.
    #     # Formula: (pi^(1/3) * (6*V)^(2/3)) / A
    #     if vol_mm3 > 0 and area_mm2 > 0:
    #         sphericity = (np.pi**(1/3) * (6 * vol_mm3)**(2/3)) / area_mm2
    #     else:
    #         sphericity = np.nan

    #     # 3. Mean Curvature
    #     try:
    #         # Requires simple closed surface. 
    #         # If this fails, install `scipy` or check if mesh is watertight.
    #         curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, 1.0)
    #         avg_curvature = np.mean(np.abs(curvature))
    #     except Exception as e:
    #         # Print error to help debug why it's missing
    #         # print(f"[Warning] Curvature failed: {e}") 
    #         avg_curvature = np.nan
            
    #     return sa_v_ratio, avg_curvature, sphericity

    def evaluate_volume(self, volume, volume_source_name, labels=[1, 2, 3, 4]):
        results = []
        valid_labels = [l for l in labels if np.any(volume == l)]
        
        for cls in valid_labels:
            binary = (volume == cls)
            if np.sum(binary) == 0: continue
            
            # Slice Consistency
            is_dice = self._inter_slice_dice(binary)
            tv_z = self._total_variation_z(binary)
            
            # Geometric
            # mesh = self._get_mesh(binary)
            # sav, curv, sphericity = self._geometric_metrics(mesh, binary)
            
            results.append({
                "Source": volume_source_name,
                "Class": str(cls),
                "InterSlice_Dice": is_dice,
                "TotalVariation_Z": tv_z,
                # "SA_V_Ratio": sav,
                # "Mean_Curvature": curv,
                # "Sphericity": sphericity
            })
            
        if results:
            df_temp = pd.DataFrame(results)
            avg_row = df_temp.mean(numeric_only=True).to_dict()
            avg_row['Source'] = volume_source_name
            avg_row['Class'] = 'Average'
            results.append(avg_row)
            
        return pd.DataFrame(results)

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

def parse_test_manifest(manifest_2d_path: str):
    """
    Parses manifest for test set which maps 'TEST_SET_ID' -> 'sub-X/filename_slice-Y'.
    Returns a dict: { 'subject/filename': [ {'case_2d_id': ..., 'slice_idx': ...}, ... ] }
    """
    print(f"📂 Parsing TEST manifest: {manifest_2d_path}")
    with open(manifest_2d_path, 'r') as f:
        man_2d = json.load(f)

    vol_to_2d_map = defaultdict(list)
    matches = 0
    for case_2d_id, path_str in man_2d.items():
        # Parsing "sub-TNU025/S4_..._slice-310"
        match = re.search(r"(.*)_slice-(\d+)$", path_str)
        if match:
            full_vol_rel_path = match.group(1) # e.g. "sub-TNU025/S4_..."
            slice_idx = int(match.group(2))
            
            vol_to_2d_map[full_vol_rel_path].append({
                'case_2d_id': case_2d_id,
                'slice_idx': slice_idx,
                'orig_vol_rel_path': full_vol_rel_path
            })
            matches += 1
            
    print(f"✅ Found {len(vol_to_2d_map)} unique 3D volumes containing {matches} 2D slice targets.")
    return vol_to_2d_map

def compute_hd95_score(pred_bin, gt_bin, spacing):
    """
    Computes HD95 by aggregating 2D slice distances using the helper.
    """
    dists_acc = []
    
    # Iterate over Z-axis
    for z in range(pred_bin.shape[0]):
        p_slice = pred_bin[z, :, :]
        g_slice = gt_bin[z, :, :]
        
        # Skip if completely empty to save time (helper handles it, but this is an optimization)
        if not np.any(p_slice) and not np.any(g_slice):
            continue
            
        # Use HELPER function
        d = compute_surface_distances_2d(p_slice, g_slice, spacing)
        
        if d is not None:
            dists_acc.append(d)
            
    if not dists_acc:
        # If volume is empty in both, HD95 is 0.0. If one empty/mismatch, NaN.
        if not np.any(pred_bin) and not np.any(gt_bin):
            return 0.0
        return np.nan

    all_dists = np.concatenate(dists_acc)
    
    # Use HELPER function for the final percentile
    return calculate_hd95(all_dists)

def compute_hd95_raw_2d(pred_bin, gt_bin, spacing):
    """
    Computes list of distances for 2D, handling spacing like inference_2.py.
    Returns the raw list of surface distances (concatenated both ways).
    """
    # Ensure inputs are 2D (H, W)
    if pred_bin.ndim == 3 and pred_bin.shape[0] == 1:
        pred_bin = pred_bin[0]
    elif pred_bin.ndim == 3 and pred_bin.shape[-1] == 1:
        pred_bin = pred_bin[..., 0]
    if gt_bin.ndim == 3 and gt_bin.shape[0] == 1:
        gt_bin = gt_bin[0]
    elif gt_bin.ndim == 3 and gt_bin.shape[-1] == 1:
        gt_bin = gt_bin[..., 0]
    
    pred_empty = not np.any(pred_bin)
    gt_empty = not np.any(gt_bin)
    
    if pred_empty and gt_empty:
        return np.array([0.0]) # Perfect match
    if pred_empty or gt_empty:
        return None # One empty, one not -> penalty

    # Alignment with inference_2.py logic for spacing
    # Spacing is (x, y, z). We want (y, x) for numpy 2D slice processing
    if spacing is not None:
        if len(spacing) == 3:
            current_spacing = (spacing[1], spacing[0]) 
        elif len(spacing) == 2:
            current_spacing = (spacing[1], spacing[0])
        else:
            current_spacing = spacing
    else:
        current_spacing = (1.0, 1.0)
        
    pred_border = get_border_points(pred_bin, current_spacing)
    gt_border = get_border_points(gt_bin, current_spacing)
    
    if len(pred_border) == 0 or len(gt_border) == 0:
        return None
    
    tree_gt = cKDTree(gt_border)
    tree_pred = cKDTree(pred_border)
    d_pred_to_gt, _ = tree_gt.query(pred_border, k=1)
    d_gt_to_pred, _ = tree_pred.query(gt_border, k=1)
    
    return np.concatenate([d_pred_to_gt, d_gt_to_pred])

def compute_metrics(pred, gt, spacing, labels=[1, 2, 3, 4], is_3d=True):
    metrics = {}
    
    for label in labels:
        p_bin = (pred == label)
        g_bin = (gt == label)
        
        # Calculate Counts for Global Aggregation
        tp = np.logical_and(p_bin, g_bin).sum()
        fp = np.logical_and(p_bin, ~g_bin).sum()
        fn = np.logical_and(~p_bin, g_bin).sum()
        
        metrics[f'TP_{label}'] = tp
        metrics[f'FP_{label}'] = fp
        metrics[f'FN_{label}'] = fn

        if is_3d:
            inter = np.logical_and(p_bin, g_bin).sum()
            union = p_bin.sum() + g_bin.sum()
            dice = (2.0 * inter) / (union + 1e-8) if (p_bin.sum() + g_bin.sum()) > 0 else (np.nan if p_bin.sum() == 0 else 0.0)
            metrics[f'Dice_{label}'] = dice
            
            # Use the refactored 3D HD95 function
            metrics[f'HD95_{label}'] = compute_hd95_score(p_bin, g_bin, spacing)
        else:
            # 2D case (Slice extraction) - Use the HELPER directly
            metrics[f'Dice_{label}'] = compute_binary_dice_2d(p_bin, g_bin)
            
            # HD95 for 2D Slice
            sp_2d = (1.0, 1.0)
            if hasattr(spacing, '__len__') and len(spacing) >= 2:
                # Assuming spacing is (x, y, z) and image is (y, x)
                sp_2d = (spacing[1], spacing[0])
            
            d = compute_surface_distances_2d(p_bin, g_bin, sp_2d)
            if d is not None:
                metrics[f'HD95_{label}'] = calculate_hd95(d)
            else:
                if not np.any(p_bin) and not np.any(g_bin):
                    metrics[f'HD95_{label}'] = 0.0
                else:
                    metrics[f'HD95_{label}'] = np.nan

    return metrics

def compute_counts(pred, gt, labels=[1, 2, 3, 4]):
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

        if args.mode != 'test':
             self.nnunet_to_2d_map = parse_manifests(args.manifest_2d, args.manifest_3d)
        else:
             # In test mode, we parse directly inside run (or here using parse_test_manifest)
             self.nnunet_to_2d_map = {}
        
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

    def _run_test_mode(self):
        print("🚀 Running in TEST mode")
        
        # Load Manifest
        test_vol_map = parse_test_manifest(self.args.manifest_2d)
        sorted_vols = sorted(list(test_vol_map.keys()))
        
        # 1. Determine Inference Loops (Looping over folds vs single ensemble)
        if self.ensemble:
            folder_suffix = "_TTA_Ensemble" if self.use_tta else "_Ensemble"
            folder_name = f"inference_results_test{folder_suffix}"
            inference_loops = [(folder_name, self.args.folds)]
        else:
            inference_loops = []
            base_folder = "inference_results_test_TTA" if self.use_tta else "inference_results_test"
            for f in self.args.folds:
                name = join(base_folder, f"fold_{f}")
                inference_loops.append((name, [f]))

        # 2. Iterate through the folds (or the single ensemble)
        for rel_folder, current_folds in inference_loops:
            print(f"\n========================================================")
            print(f"🚀 Running TEST: {rel_folder} (Folds: {current_folds})")
            print(f"========================================================")

            # Initialize Predictor for these specific folds
            self.predictor.initialize_from_trained_model_folder(
                self.args.model_folder,
                use_folds=current_folds,
                checkpoint_name='checkpoint_best.pth'
            )
            preprocessor = self.predictor.configuration_manager.preprocessor_class(verbose=False)
            
            # Metrics storage for this specific loop
            tracker_2d = MetricTracker()
            metrics_2d_list = []
            metrics_2d_casewise_list = []
            smoothness_list = []
            
            # Output directory for test results
            out_root = join(self.args.output_root, rel_folder)
            out_vol_dir = join(out_root, "volumes_3d")
            dir_viz_2d = join(out_root, "visualizations_2d_slices")
            maybe_mkdir_p(out_vol_dir)
            maybe_mkdir_p(dir_viz_2d)
            
            # Iterate Volumes
            for vol_key in tqdm(sorted_vols, desc=f"Test Volumes ({rel_folder})"):
                parts = vol_key.split('/')
                if len(parts) >= 2:
                    subj = parts[0]
                    fname = parts[1]
                    fpath_mag = join(self.args.input_path_raw, subj, 'anat', fname + '_part-mag.nii.gz')
                    fpath_phase = join(self.args.input_path_raw, subj, 'anat', fname + '_part-phase.nii.gz')
                else:
                    print(f"Skipping malformed key: {vol_key}")
                    continue
                
                # Fallback search if exact path construction fails
                if not isfile(fpath_mag):
                     candidates = subfiles(join(self.args.input_path_raw, subj, 'anat'), prefix=fname, suffix='part-mag.nii.gz', join=True)
                     if candidates:
                         fpath_mag = candidates[0]
                         fpath_phase = fpath_mag.replace('part-mag', 'part-phase')
                     else:
                         print(f"❌ Mag file not found for {vol_key} at {fpath_mag}")
                         continue
                
                files = [fpath_mag]
                if isfile(fpath_phase):
                    files.append(fpath_phase)
                    
                print(f"   Inference on: {fname}")

                # Run Inference
                data_npy, seg_npy, data_properties = preprocessor.run_case(
                    files, None, self.predictor.plans_manager, 
                    self.predictor.configuration_manager, self.predictor.dataset_json
                )
                
                data_tensor = torch.from_numpy(data_npy).float().to(self.device).unsqueeze(0) 

                if ("otsu" in self.args.experiment and "no_otsu" not in self.args.experiment) or \
                   "phase_prepro" in self.args.experiment or "mag_prepro" in self.args.experiment:
                    phase_prepro = "phase_prepro" in self.args.experiment
                    mag_prepro = "mag_prepro" in self.args.experiment
                    data_tensor = preprocess_gpu(data_tensor, phase_prepro=phase_prepro, mag_prepro=mag_prepro, is_3d=True)

                predicted_logits = self.predictor.predict_logits_from_preprocessed_data(data_tensor.squeeze(0)).cpu()
                pred_seg = convert_predicted_logits_to_segmentation_with_correct_shape(
                    predicted_logits, self.predictor.plans_manager, 
                    self.predictor.configuration_manager, self.predictor.label_manager,
                    data_properties, return_probabilities=False
                )
                
                # Save 3D Prediction
                save_name = f"{subj}_{fname}.nii.gz"
                self.io.write_seg(pred_seg, join(out_vol_dir, save_name), data_properties)
                
                # --- SMOOTHNESS EVALUATION ---
                spacing = data_properties['spacing']
                smooth_eval = SmoothnessEvaluator(pixel_spacing=spacing)
                df_smooth_pred = smooth_eval.evaluate_volume(pred_seg, "Prediction")
                if not df_smooth_pred.empty:
                    df_smooth_pred['Case_ID'] = f"{subj}_{fname}"
                    smoothness_list.append(df_smooth_pred)

                # Evaluate Slices
                targets = test_vol_map[vol_key]
                orig_mag_img = sitk.ReadImage(fpath_mag)
                orig_mag_npy = sitk.GetArrayFromImage(orig_mag_img) 
                
                if len(files) > 1:
                     orig_phase_img = sitk.ReadImage(fpath_phase)
                     orig_phase_npy = sitk.GetArrayFromImage(orig_phase_img)
                else:
                     orig_phase_npy = None

                case_raw_dists = defaultdict(list)
                case_dice_scores = defaultdict(list)

                for t in targets:
                    slice_idx = t['slice_idx']
                    case_2d_id = t['case_2d_id']
                    
                    gt_path = join(self.args.gt_path_2d, f"{case_2d_id}.nii.gz")
                    if not isfile(gt_path):
                         print(f"Missing GT: {gt_path}")
                         continue
                    
                    gt_obj = sitk.ReadImage(gt_path) 
                    gt_npy = sitk.GetArrayFromImage(gt_obj) 
                    if gt_npy.ndim == 3: gt_npy = gt_npy[0] 

                    gt_shape = gt_npy.shape
                    vol_shape = pred_seg.shape
                    
                    current_slice = None
                    best_axis = -1

                    if self.args.slice_axis != -1:
                        ax = self.args.slice_axis
                        if ax == 0 and slice_idx < vol_shape[0]:
                            current_slice = pred_seg[slice_idx, :, :]
                            best_axis = 0
                        elif ax == 1 and slice_idx < vol_shape[1]:
                            current_slice = pred_seg[:, slice_idx, :]
                            best_axis = 1
                        elif ax == 2 and slice_idx < vol_shape[2]:
                            current_slice = pred_seg[:, :, slice_idx]
                            best_axis = 2
                    else:
                        if vol_shape[1:] == gt_shape:
                            if slice_idx < vol_shape[0]:
                                current_slice = pred_seg[slice_idx, :, :]
                                best_axis = 0
                        elif (vol_shape[0], vol_shape[2]) == gt_shape:
                            if slice_idx < vol_shape[1]:
                                current_slice = pred_seg[:, slice_idx, :]
                                best_axis = 1
                        elif (vol_shape[0], vol_shape[1]) == gt_shape:
                            if slice_idx < vol_shape[2]:
                                current_slice = pred_seg[:, :, slice_idx]
                                best_axis = 2

                    if current_slice is None:
                        print(f"Skipping {case_2d_id}: Index {slice_idx} out of bounds or shape mismatch. Vol: {vol_shape}, GT: {gt_shape}")
                        continue

                    m = compute_metrics(current_slice, gt_npy, data_properties['spacing'], labels=[1,2,3,4], is_3d=False)
                    
                    m['case_id'] = case_2d_id
                    m['best_orientation'] = f"Axis_{best_axis}"
                    
                    tracker_2d.add_case_metric(m)
                    tracker_2d.update_counts(current_slice, gt_npy, labels=[1, 2, 3, 4])
                    
                    for k, v in m.items():
                        if 'Dice_' in k:
                            try:
                                lbl = int(k.split('_')[-1])
                                case_dice_scores[lbl].append(v)
                            except ValueError: pass

                    labels = [1, 2, 3, 4]
                    for lbl in labels:
                        p_bin = (current_slice == lbl)
                        g_bin = (gt_npy == lbl)
                        
                        dists = compute_surface_distances_2d(p_bin, g_bin, spacing)
                    
                        if dists is not None:
                            case_raw_dists[lbl].extend(dists)
                            m[f'HD95_{lbl}'] = calculate_hd95(dists)
                        else:
                            m[f'HD95_{lbl}'] = np.nan

                    metrics_2d_list.append(m)
                    
                    d_sc = m.get('Dice_1', 0)
                    d_gm = m.get('Dice_2', 0)
                    score = np.nanmean([d_sc, d_gm]) if (not np.isnan(d_sc) and not np.isnan(d_gm)) else 0
                    print(f"   [{case_2d_id}] Processed. (Disc+GM Dice: {score:.3f})")
                    
                    try:
                        viz_path = join(dir_viz_2d, f"{case_2d_id}.png")
                        
                        v_mag = None
                        v_phase = None
                        
                        if orig_mag_npy is not None:
                            if best_axis == 0 and slice_idx < orig_mag_npy.shape[0]:
                                v_mag = orig_mag_npy[slice_idx, :, :]
                            elif best_axis == 1 and slice_idx < orig_mag_npy.shape[1]:
                                v_mag = orig_mag_npy[:, slice_idx, :]
                            elif best_axis == 2 and slice_idx < orig_mag_npy.shape[2]:
                                v_mag = orig_mag_npy[:, :, slice_idx]

                        if orig_phase_npy is not None:
                            if best_axis == 0 and slice_idx < orig_phase_npy.shape[0]:
                                v_phase = orig_phase_npy[slice_idx, :, :]
                            elif best_axis == 1 and slice_idx < orig_phase_npy.shape[1]:
                                v_phase = orig_phase_npy[:, slice_idx, :]
                            elif best_axis == 2 and slice_idx < orig_phase_npy.shape[2]:
                                v_phase = orig_phase_npy[:, :, slice_idx]
                        
                        if v_mag is not None:
                             plot_2d_slice_eval(v_mag, v_phase, gt_npy, current_slice, case_2d_id, viz_path)
                        else:
                             print(f"Skipping viz for {case_2d_id}: Could not extract slice {slice_idx} along axis {best_axis}.")
                             
                    except Exception as e:
                        print(f"Viz error {case_2d_id}: {e}")

                case_metrics_agg = {'Case_ID': f"{subj}_{fname}"}
                labels = [1, 2, 3, 4]
                for lbl in labels:
                    dices = case_dice_scores[lbl]
                    valid_dices = [d for d in dices if not np.isnan(d)]
                    if valid_dices:
                        case_metrics_agg[f'Dice_{lbl}'] = np.mean(valid_dices)
                    else:
                        case_metrics_agg[f'Dice_{lbl}'] = np.nan
                    
                    raw_d = case_raw_dists[lbl]
                    if raw_d:
                        case_metrics_agg[f'HD95_{lbl}'] = np.percentile(np.array(raw_d), 95)
                    else:
                        case_metrics_agg[f'HD95_{lbl}'] = np.nan
                
                metrics_2d_casewise_list.append(case_metrics_agg)

            # Save Summary for this fold/ensemble
            if metrics_2d_list:
                if smoothness_list:
                    df_smooth_all = pd.concat(smoothness_list, ignore_index=True)
                    keep_cols = ['Case_ID', 'Source', 'Class', 'InterSlice_Dice', 'TotalVariation_Z']
                    final_cols = [c for c in keep_cols if c in df_smooth_all.columns]
                    df_smooth_all = df_smooth_all[final_cols]
                    df_smooth_all.to_csv(join(out_root, "smoothness_casewise.csv"), index=False)
                    
                    numeric_cols = df_smooth_all.select_dtypes(include=np.number).columns.tolist()
                    df_smooth_global = df_smooth_all.groupby(['Source', 'Class'])[numeric_cols].agg(['mean', 'std']).reset_index()
                    df_smooth_global.columns = ['Source', 'Class'] + [f"{c[0]}_{c[1]}" for c in df_smooth_global.columns[2:]]
                    df_smooth_global.to_csv(join(out_root, "smoothness_global.csv"), index=False)
                
                df_2d_slice = pd.DataFrame(metrics_2d_list)
                df_2d_slice.to_csv(join(out_root, "metrics_2d_slicewise.csv"), index=False)
                
                if metrics_2d_casewise_list:
                    df_2d_case = pd.DataFrame(metrics_2d_casewise_list)
                    df_2d_case.to_csv(join(out_root, "metrics_2d_casewise.csv"), index=False)
                
                tracker_2d.save_global_summary_csv(join(out_root, "metrics_2d_global.csv"))
                
                print(f"✅ Saved metrics to {out_root}")
            else:
                print(f"❌ No metrics computed for {rel_folder}.")

    def run(self):
        if self.args.mode == 'test':
            self._run_test_mode()
            return
            
        if self.args.mode == 'validation':
            inp_dir = join(self.args.input_path_raw, 'imagesTr')
            gt_dir_3d = join(self.args.input_path_raw, 'labelsTr')
        else:
            inp_dir = join(self.args.input_path_raw, 'imagesTs')
            gt_dir_3d = join(self.args.input_path_raw, 'labelsTs') 
        
        if not os.path.isdir(inp_dir) and self.args.mode != 'single_volume':
            raise ValueError(f"Input directory not found: {inp_dir}")

        if self.args.mode == 'single_volume':
            self.run_single_volume()
            return

        all_files = subfiles(inp_dir, suffix='.nii.gz', join=False)
        all_cases = sorted(list(set([f.split('_0000.nii.gz')[0] for f in all_files if '_0000.nii.gz' in f])))
        print(f"Found {len(all_cases)} total cases in directory.")

        if self.ensemble:
            base_name = "ensemble_predictions"
            if self.use_tta:
                base_name += "_TTA"
            inference_loops = [(base_name, self.args.folds)]
        else:
            inference_loops = []
            for f in self.args.folds:
                name = f"fold_{f}"
                if self.use_tta:
                    name += "_TTA"
                inference_loops.append((name, [f]))

        cross_fold_metrics_2d_means = []
        cross_fold_metrics_3d_means = []
        cross_fold_global_2d = []
        cross_fold_global_3d = []

        # --- Main Loop ---
        if not self.args.summary_only:
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
                else:
                    target_cases = all_cases

                if not target_cases:
                    print("   ⚠️ No cases to process for this loop. Skipping.")
                    continue

                self.predictor.initialize_from_trained_model_folder(
                    self.args.model_folder,
                    use_folds=current_folds,
                    checkpoint_name='checkpoint_best.pth'
                )

                preprocessor = self.predictor.configuration_manager.preprocessor_class(
                    verbose=self.predictor.verbose
                )

                current_output_dir = join(self.args.output_root, folder_name)
                dir_pred_3d = join(current_output_dir, "predictions_3d")
                dir_viz_3d = join(current_output_dir, "visualizations_3d")
                dir_viz_2d = join(current_output_dir, "visualizations_2d_slices")
                maybe_mkdir_p(dir_pred_3d)
                maybe_mkdir_p(dir_viz_3d)
                maybe_mkdir_p(dir_viz_2d)

                metrics_3d_list = []
                metrics_2d_list = []
                metrics_2d_casewise_list = [] # True casewise (aggregated)
                smoothness_list = []  # <--- NEW LIST

                tracker_3d = MetricTracker()
                tracker_2d = MetricTracker()

                for case_id in target_cases:
                    print(f"   Processing: {case_id}")
                    
                    fpath_c0 = join(inp_dir, f"{case_id}_0000.nii.gz")
                    fpath_c1 = join(inp_dir, f"{case_id}_0001.nii.gz")
                    files_to_read = [fpath_c0]
                    if isfile(fpath_c1): files_to_read.append(fpath_c1)
                    
                    expected_channels = len(self.predictor.dataset_json['channel_names'].keys())
                    if self.args.experiment == "mag_only" and expected_channels == 2 and len(files_to_read) == 1:
                        files_to_read.append(fpath_c0)

                    data_npy, seg_npy, data_properties = preprocessor.run_case(
                        files_to_read, None, self.predictor.plans_manager, 
                        self.predictor.configuration_manager, self.predictor.dataset_json
                    )
                    spacing = data_properties['spacing']

                    data_tensor = torch.from_numpy(data_npy).float().to(self.device)
                    data_tensor = data_tensor.unsqueeze(0)
                    
                    if self.args.experiment == "mag_only" and data_tensor.shape[1] > 1:
                        data_tensor[:, 1] = data_tensor[:, 0]

                    if ("otsu" in self.args.experiment and "no_otsu" not in self.args.experiment) or "phase_prepro" in self.args.experiment or "mag_prepro" in self.args.experiment:
                        phase_prepro = "phase_prepro" in self.args.experiment
                        mag_prepro = "mag_prepro" in self.args.experiment
                        data_tensor = preprocess_gpu(data_tensor, phase_prepro=phase_prepro, mag_prepro=mag_prepro, is_3d=True)

                    data_for_net = data_tensor.squeeze(0)
                    predicted_logits = self.predictor.predict_logits_from_preprocessed_data(data_for_net).cpu()

                    pred_seg = convert_predicted_logits_to_segmentation_with_correct_shape(
                        predicted_logits, self.predictor.plans_manager, 
                        self.predictor.configuration_manager, self.predictor.label_manager,
                        data_properties, return_probabilities=False
                    )

                    if self.args.save_predictions:
                        self.io.write_seg(pred_seg, join(dir_pred_3d, f"{case_id}.nii.gz"), data_properties)

                    # --- VISUALIZATION & METRICS ---
                    viz_files = [fpath_c0]
                    if isfile(fpath_c1): viz_files.append(fpath_c1)
                    img_list_viz, _ = self.io.read_images(viz_files)
                    if self.args.experiment == "mag_only" and img_list_viz.shape[0] == 1 and expected_channels == 2:
                        img_list_viz = np.concatenate([img_list_viz, img_list_viz], axis=0)
                    
                    # 3D Evaluation
                    gt_path = join(gt_dir_3d, f"{case_id}.nii.gz")
                    gt_3d = None
                    if isfile(gt_path):
                        gt_3d, _ = self.io.read_images([gt_path])
                        gt_3d = gt_3d[0]
                        
                        m_3d = compute_metrics(pred_seg, gt_3d, spacing)
                        m_3d = compute_metrics(pred_seg, gt_3d, spacing)

                        # Add to tracker
                        tracker_3d.add_case_metric(m_3d)
                        tracker_3d.update_counts(pred_seg, gt_3d, labels=[1, 2, 3, 4])
                        
                        save_ortho_view(img_list_viz, pred_seg, gt_3d, case_id, join(dir_viz_3d, f"{case_id}_ortho.png"))
                    else:
                        save_ortho_view(img_list_viz, pred_seg, None, case_id, join(dir_viz_3d, f"{case_id}_ortho.png"))

                    # --- SMOOTHNESS EVALUATION (New Block) ---
                    smooth_eval = SmoothnessEvaluator(pixel_spacing=spacing)
                    
                    # 1. Evaluate Prediction
                    df_smooth_pred = smooth_eval.evaluate_volume(pred_seg, "Prediction")
                    if not df_smooth_pred.empty:
                        df_smooth_pred['Case_ID'] = case_id
                        smoothness_list.append(df_smooth_pred)

                    # 2. Evaluate GT (if available)
                    if gt_3d is not None:
                        df_smooth_gt = smooth_eval.evaluate_volume(gt_3d, "GroundTruth")
                        if not df_smooth_gt.empty:
                            df_smooth_gt['Case_ID'] = case_id
                            smoothness_list.append(df_smooth_gt)

                    # 2D Evaluation
                    if case_id in self.nnunet_to_2d_map:
                        linked_slices = self.nnunet_to_2d_map[case_id]
                        
                        # Aggregators for this case
                        case_raw_dists = defaultdict(list)
                        case_dice_scores = defaultdict(list)

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
                                    input_mag = img_list_viz[0, slice_idx, :, :]
                                    input_phase = img_list_viz[1, slice_idx, :, :] if img_list_viz.shape[0] > 1 else None

                                    # Compute Dice (slice-wise) - is_3d=False avoids calling 3D HD95
                                    m_2d_slice = compute_metrics(pred_slice, gt_2d, spacing, is_3d=False)
                                    
                                    tracker_2d.add_case_metric(m_2d_slice)
                                    tracker_2d.update_counts(pred_slice, gt_2d, labels=[1, 2, 3, 4])
                                    
                                    # Collect Dices (Restored)
                                    for k, v in m_2d_slice.items():
                                        if 'Dice_' in k:
                                            try:
                                                lbl = int(k.split('_')[-1])
                                                case_dice_scores[lbl].append(v)
                                            except ValueError: pass

                                    # Compute Raw HD95 distances
                                    labels = [1, 2, 3, 4]
                                    for lbl in labels:
                                        p_bin = (pred_slice == lbl)
                                        g_bin = (gt_2d == lbl)
                                        
                                        dists = compute_surface_distances_2d(p_bin, g_bin, spacing)
                                    
                                        if dists is not None:
                                            case_raw_dists[lbl].extend(dists)
                                            m_2d_slice[f'HD95_{lbl}'] = calculate_hd95(dists)
                                        else:
                                            m_2d_slice[f'HD95_{lbl}'] = np.nan
                                    
                                    # Append modified m_2d_slice to list AFTER computing HD95
                                    metrics_2d_list.append(m_2d_slice)

                                    save_2d_slice_viz(
                                        vol_slice_mag=input_mag, 
                                        vol_slice_phase=input_phase, 
                                        gt_2d=gt_2d, 
                                        pred_slice=pred_slice, 
                                        case_2d_id=case_2d_id, 
                                        output_path=join(dir_viz_2d, f"{case_2d_id}.png")
                                    )

                        # End of slice loop for this case. Now Aggregate for Case.
                        case_metrics_agg = {'Case_ID': case_id}
                        labels = [1, 2, 3, 4]
                        for lbl in labels:
                            # Dice Aggregation (Mean)
                            dices = case_dice_scores[lbl]
                            # Filter NaNs first to avoid "Mean of empty slice" warning
                            valid_dices = [d for d in dices if not np.isnan(d)]
                            if valid_dices:
                                case_metrics_agg[f'Dice_{lbl}'] = np.mean(valid_dices)
                            else:
                                case_metrics_agg[f'Dice_{lbl}'] = np.nan
                            
                            # HD95 Aggregation (Percentile of all raw distances)
                            raw_d = case_raw_dists[lbl]
                            if raw_d:
                                case_metrics_agg[f'HD95_{lbl}'] = np.percentile(np.array(raw_d), 95)
                            else:
                                case_metrics_agg[f'HD95_{lbl}'] = np.nan
                        
                        metrics_2d_casewise_list.append(case_metrics_agg)

                # --- Saving Fold Results ---
                
                if smoothness_list:
                    # Casewise
                    df_smooth_all = pd.concat(smoothness_list, ignore_index=True)
                    # Updated columns to include Sphericity
                    keep_cols = ['Case_ID', 'Source', 'Class', 
                                 'InterSlice_Dice', 'TotalVariation_Z'] 
                                #  'SA_V_Ratio', 'Mean_Curvature', 'Sphericity']
                    
                    # Only keep columns that actually exist (handles cases where Curvature might fail globally)
                    final_cols = [c for c in keep_cols if c in df_smooth_all.columns]
                    df_smooth_all = df_smooth_all[final_cols]
                    
                    df_smooth_all.to_csv(join(current_output_dir, "smoothness_casewise.csv"), index=False)
                    
                    # Global
                    # Aggregating all numeric columns automatically catches Sphericity
                    numeric_cols = df_smooth_all.select_dtypes(include=np.number).columns.tolist()
                    df_smooth_global = df_smooth_all.groupby(['Source', 'Class'])[numeric_cols].agg(['mean', 'std']).reset_index()
                    df_smooth_global.columns = ['Source', 'Class'] + [f"{c[0]}_{c[1]}" for c in df_smooth_global.columns[2:]]
                    df_smooth_global.to_csv(join(current_output_dir, "smoothness_global.csv"), index=False)
                    print(f"   Saved smoothness metrics to: {current_output_dir}")

                if metrics_3d_list:
                    df_3d = pd.DataFrame(metrics_3d_list)
                    df_3d.to_csv(join(current_output_dir, "metrics_3d_casewise.csv"), index=False)
                    mean_3d = df_3d.mean(numeric_only=True).to_dict()
                    mean_3d['Fold'] = folder_name
                    cross_fold_metrics_3d_means.append(mean_3d)

                if metrics_2d_list:
                    # Save Slice-wise
                    df_2d_slice = pd.DataFrame(metrics_2d_list)
                    df_2d_slice.to_csv(join(current_output_dir, "metrics_2d_slicewise.csv"), index=False)
                    
                    # Save Casewise (Aggregated)
                    if metrics_2d_casewise_list:
                        df_2d_case = pd.DataFrame(metrics_2d_casewise_list)
                        df_2d_case.to_csv(join(current_output_dir, "metrics_2d_casewise.csv"), index=False)
                        
                        # Use aggregated stats for cross-fold summary
                        mean_2d = df_2d_case.mean(numeric_only=True).to_dict()
                        mean_2d['Fold'] = folder_name
                        cross_fold_metrics_2d_means.append(mean_2d)
                    else:
                        # Fallback if somehow empty but slice list wasn't? Unlikely.
                        pass

                tracker_3d.save_casewise_csv(join(current_output_dir, "metrics_3d_casewise.csv"))
                tracker_3d.save_global_summary_csv(join(current_output_dir, "metrics_3d_global.csv"))

                # Save 2D results
                tracker_2d.save_casewise_csv(join(current_output_dir, "metrics_2d_slicewise.csv"))
                tracker_2d.save_global_summary_csv(join(current_output_dir, "metrics_2d_global.csv"))

        # --- FINAL CROSS-FOLD SUMMARY ---
        if not self.ensemble and not self.args.no_summary:
            print("\nComputing Cross-Fold Statistics from saved CSVs on disk...")
            fold_dirs = [join(self.args.output_root, d) for d in os.listdir(self.args.output_root) if os.path.isdir(join(self.args.output_root, d)) and d.startswith('fold_')]
            
            agg_means_3d, agg_means_2d = [], []
            agg_global_3d, agg_global_2d = [], []
            agg_smoothness = []  # <--- NEW LIST

            for f_dir in fold_dirs:
                fold_name = os.path.basename(f_dir)
                
                # 1. Load Standard Metrics (Casewise)
                for fname, agg_list in [("metrics_3d_casewise.csv", agg_means_3d), ("metrics_2d_casewise.csv", agg_means_2d)]:
                    p = join(f_dir, fname)
                    if isfile(p):
                        df = pd.read_csv(p)
                        mean_dict = df.mean(numeric_only=True).to_dict()
                        mean_dict['Fold'] = fold_name
                        agg_list.append(mean_dict)
                
                # 2. Load Standard Metrics (Global)
                for fname, agg_list in [("metrics_3d_global.csv", agg_global_3d), ("metrics_2d_global.csv", agg_global_2d)]:
                    p = join(f_dir, fname)
                    if isfile(p):
                        df = pd.read_csv(p)
                        glob_dict = {}
                        for _, row in df.iterrows():
                            glob_dict[f"Dice_{int(row['Label'])}"] = row['Global_Dice']
                            if 'Mean_HD95' in row:
                                glob_dict[f"HD95_{int(row['Label'])}"] = row['Mean_HD95']
                        glob_dict['Fold'] = fold_name
                        agg_list.append(glob_dict)

                # 3. Load Smoothness Metrics (Global) <--- NEW BLOCK
                p_smooth = join(f_dir, "smoothness_global.csv")
                if isfile(p_smooth):
                    df_s = pd.read_csv(p_smooth)
                    # We simply append the whole dataframe, adding a Fold column
                    df_s['Fold'] = fold_name
                    agg_smoothness.append(df_s)

            # --- SAVE SUMMARY (Standard) ---
            def save_summary(agg_list, suffix):
                if not agg_list: return
                df_agg = pd.DataFrame(agg_list)
                means = df_agg.mean(numeric_only=True)
                stds = df_agg.std(numeric_only=True)
                
                summary_data = {}
                for key in means.index:
                    parts = key.split('_')
                    if len(parts) < 2: continue
                    metric_type = parts[0]
                    try: label = int(parts[-1])
                    except ValueError: continue
                        
                    if label not in summary_data: summary_data[label] = {'Label': label}
                    summary_data[label][f'{metric_type}_Mean'] = means[key]
                    summary_data[label][f'{metric_type}_Std'] = stds[key]
                
                final_df = pd.DataFrame(list(summary_data.values()))
                if not final_df.empty:
                    final_df = final_df.sort_values('Label')
                    out_name = join(self.args.output_root, f"summary_cross_fold_{suffix}.csv")
                    final_df.to_csv(out_name, index=False)
                    print(f"   Saved: {os.path.basename(out_name)}")

            save_summary(agg_means_3d, "casewise_3d")
            save_summary(agg_means_2d, "casewise_2d")
            save_summary(agg_global_3d, "global_3d")
            save_summary(agg_global_2d, "global_2d")

            # --- SAVE SUMMARY (Smoothness) <--- NEW BLOCK
            if agg_smoothness:
                df_all_smooth = pd.concat(agg_smoothness, ignore_index=True)
                
                # We filter for columns containing 'mean', 'std', 'Ratio', 'Variation', 'Dice', OR 'Sphericity'
                # This ensures we pick up the new metric from the global files
                valid_keywords = ['mean', 'std', 'Ratio', 'Variation', 'Dice'] #, 'Sphericity', 'Curvature']
                numeric_cols = [c for c in df_all_smooth.columns if any(k in c for k in valid_keywords)]
                
                # Group by Source and Class, averaging across folds
                summary_smooth = df_all_smooth.groupby(['Source', 'Class'])[numeric_cols].mean(numeric_only=True).reset_index()
                
                out_name_s = join(self.args.output_root, "summary_cross_fold_smoothness.csv")
                summary_smooth.to_csv(out_name_s, index=False)
                print(f"   Saved: {os.path.basename(out_name_s)}")
                
        print("\nAll inference loops completed.")

    def run_single_volume(self):
        print("🚀 Running in single volume mode.")
        if not self.args.single_volume_path or not isfile(self.args.single_volume_path):
            raise ValueError(f"File not found for single volume mode: {self.args.single_volume_path}")

        # --- Determine file paths and case ID ---
        vol_path = self.args.single_volume_path
        if 'part-mag' in vol_path:
            fpath_c0 = vol_path
            fpath_c1 = vol_path.replace('part-mag', 'part-phase')
        elif 'part-phase' in vol_path:
            fpath_c0 = vol_path.replace('part-phase', 'part-mag')
            fpath_c1 = vol_path
        else:
            fpath_c0 = vol_path
            fpath_c1 = None
        
        files_to_read = [fpath_c0]
        if fpath_c1 and isfile(fpath_c1):
            files_to_read.append(fpath_c1)
        
        print(f"   Input files: {files_to_read}")

        # Find the nnU-Net case_id from the manifest
        subj = os.path.basename(os.path.dirname(os.path.dirname(vol_path)))
        norm_fname = normalize_string(os.path.basename(fpath_c0))
        key = f"{subj}/{norm_fname}"
        
        case_id = None
        # Reload manifest to find ID
        for entry in load_json(self.args.manifest_3d):
            entry_key = f"{entry['subject']}/{normalize_string(entry['original_filename'])}"
            if key == entry_key:
                case_id = entry['nnunet_id']
                break
        
        if not case_id:
            print(f"⚠️ Could not find a matching nnU-Net ID for volume: {vol_path}. Using filename as ID.")
            case_id = normalize_string(os.path.basename(fpath_c0))
        else:
            print(f"   Found matching nnU-Net ID: {case_id}")

        # --- Setup for inference ---
        folder_name = "ensemble_predictions" if self.ensemble else f"fold_{self.args.folds[0]}"
        current_output_dir = join(self.args.output_root, folder_name, f"single_{case_id}")
        
        dir_pred_3d = join(current_output_dir, "predictions_3d")
        dir_viz_3d = join(current_output_dir, "visualizations_3d")
        maybe_mkdir_p(dir_pred_3d)
        maybe_mkdir_p(dir_viz_3d)

        self.predictor.initialize_from_trained_model_folder(
            self.args.model_folder,
            use_folds=self.args.folds,
            checkpoint_name='checkpoint_best.pth'
        )
        preprocessor = self.predictor.configuration_manager.preprocessor_class(verbose=self.predictor.verbose)

        # --- Run Inference ---
        # Note: We pass None for seg because we are in inference mode
        data_npy, seg_npy, data_properties = preprocessor.run_case(
            files_to_read, None, self.predictor.plans_manager, 
            self.predictor.configuration_manager, self.predictor.dataset_json
        )
        spacing = data_properties['spacing']
        data_tensor = torch.from_numpy(data_npy).float().to(self.device).unsqueeze(0)

        # Apply custom preprocessing if flags are set
        if ("otsu" in self.args.experiment and "no_otsu" not in self.args.experiment) or "phase_prepro" in self.args.experiment or "mag_prepro" in self.args.experiment:
            phase_prepro = "phase_prepro" in self.args.experiment
            mag_prepro = "mag_prepro" in self.args.experiment
            print('[INFO] Applying custom preprocessing')
            data_tensor = preprocess_gpu(data_tensor, phase_prepro=phase_prepro, mag_prepro=mag_prepro, is_3d=True)

        predicted_logits = self.predictor.predict_logits_from_preprocessed_data(data_tensor.squeeze(0)).cpu()
        pred_seg = convert_predicted_logits_to_segmentation_with_correct_shape(
            predicted_logits, self.predictor.plans_manager, 
            self.predictor.configuration_manager, self.predictor.label_manager,
            data_properties, return_probabilities=False
        )

        if self.args.save_predictions:
            self.io.write_seg(pred_seg, join(dir_pred_3d, f"{case_id}.nii.gz"), data_properties)

        # --- Visualization & Metrics ---
        img_list_viz, _ = self.io.read_images(files_to_read)
        
        # Try to find GT in labelsTr (Training Labels)
        gt_dir_3d = join(self.args.input_path_raw, 'labelsTr') 
        gt_path = join(gt_dir_3d, f"{case_id}.nii.gz")
        gt_3d = None
        
        smooth_eval = SmoothnessEvaluator(pixel_spacing=spacing)
        smoothness_results = []

        if isfile(gt_path):
            gt_3d, _ = self.io.read_images([gt_path])
            gt_3d = gt_3d[0]
            
            # Standard Metrics
            m_3d = compute_metrics(pred_seg, gt_3d, spacing)
            m_3d['Case_ID'] = case_id
            pd.DataFrame([m_3d]).to_csv(join(current_output_dir, "metrics_3d.csv"), index=False)
            print(f"   3D Metrics saved to {current_output_dir}/metrics_3d.csv")
            
            # Smoothness GT
            df_gt = smooth_eval.evaluate_volume(gt_3d, "GroundTruth")
            if not df_gt.empty:
                df_gt['Case_ID'] = case_id
                smoothness_results.append(df_gt)
        
        # Smoothness Pred
        df_pred = smooth_eval.evaluate_volume(pred_seg, "Prediction")
        if not df_pred.empty:
            df_pred['Case_ID'] = case_id
            smoothness_results.append(df_pred)

        # Save Smoothness
        if smoothness_results:
            df_smooth = pd.concat(smoothness_results, ignore_index=True)
            df_smooth.to_csv(join(current_output_dir, "smoothness_single.csv"), index=False)
            print(f"   Smoothness metrics saved to {current_output_dir}/smoothness_single.csv")

        # Visualization
        save_ortho_view(img_list_viz, pred_seg, gt_3d, case_id, join(dir_viz_3d, f"{case_id}_ortho.png"))
        print(f"   3D Visualization saved to {dir_viz_3d}")
        
        print(f"✅ Single volume inference complete for {case_id}.")

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
    parser.add_argument('--no_summary', action='store_true', help="Skip cross-fold summary generation")
    parser.add_argument('--summary_only', action='store_true', help="Skip inference and only run summary generation")
    parser.add_argument('--single_volume_path', type=str, default=None, help="Path to a single volume for inference when mode is 'single_volume'")
    parser.add_argument('--slice_axis', type=int, default=-1, help="Force axis for 2D slicing (0=Axial(Z in sitk), 1=Coronal(Y), 2=Sagittal(X)). -1 = auto-detect.")
    
    args = parser.parse_args()
    
    runner = Inference3DRunner(args)
    runner.run()