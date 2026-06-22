import argparse
import os
import sys
import gc
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import kornia
import contextlib 
from pathlib import Path
import scipy.ndimage
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion, generate_binary_structure
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json, isfile
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from concurrent.futures import ThreadPoolExecutor

current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent.parent  # 2D_workspace/inference_2D.py -> repo root (for helpers import)
sys.path.append(str(project_root))
print("[Info] Project root added to sys.path:", project_root)

from helpers.metric_utils_2d import compute_surface_distances_2d, compute_binary_dice_2d, calculate_hd95
from helpers.preprocessing_utils import preprocess_gpu
from helpers.stats_utils import MetricTracker
from helpers.visualization_utils import save_2d_slice_viz

OLD_TRAINER_NAME = True

# =================================================================================================
# HELPER FUNCTIONS
# =================================================================================================

def _create_mosaic_tensor(data_tensor: torch.Tensor) -> torch.Tensor:
    row = torch.cat([data_tensor, data_tensor], dim=3)
    grid = torch.cat([row, row], dim=2)
    return grid

def _get_raw_data_folder_from_model_folder(model_folder):
    try:
        raw = model_folder.replace('nnunet_results', 'nnunet_raw')
        if 'nnUNetTrainer' in raw:
            raw = raw.split('nnUNetTrainer')[0]
        return raw.rstrip('/')
    except:
        return None

def _merge_masks(anat, path):
    merged = np.zeros_like(anat)
    mask_wm = (anat == 1)
    mask_gm = (anat == 2)
    
    lesion_label = 2
    uniques = np.unique(path)
    if 2 not in uniques :
        lesion_label = 999
            
    mask_lesion = (path == lesion_label)
    
    merged[mask_wm & ~mask_lesion] = 1
    merged[mask_gm & ~mask_lesion] = 2
    merged[mask_wm & mask_lesion] = 3
    merged[mask_gm & mask_lesion] = 4
    
    return merged

# =================================================================================================
# MAIN INFERENCE CLASS
# =================================================================================================

class CustomInferenceRunner:
    def __init__(self, model_folder: str, experiment: str, folds: Tuple[int, ...] = (0,), device: str = 'cuda', use_tta: bool = False, ensemble: bool = False, model_folder_2: str = None):
        self.experiment_name = experiment
        self.folds = folds
        self.model_folder = model_folder
        self.model_folder_2 = model_folder_2
        
        if not torch.cuda.is_available():
            print("WARNING: CUDA is not available. Inference will run on CPU and be very slow.")
            self.device_obj = torch.device('cpu')
        else:
            self.device_obj = torch.device(device, 0)
            
        self.use_tta = use_tta
        self.ensemble = ensemble
        self.volume_props = None 
        
        self.exp_mag_only = (experiment in ["mag_only", "prepro_mag"]) 
        self.exp_phase_prepro = (experiment in ["phase_prepro", "full_prepro"])
        self.exp_mag_prepro = (experiment in ["mag_prepro", "full_prepro"])
        self.exp_mag_one_channel = (experiment == "mag-one-channel")
        self.exp_mosaic = (experiment == "mosaic")
        self.exp_base_no_otsu = (experiment == "base_no_otsu")
        self.exp_stacking = (experiment == "stacking")

        if self.exp_stacking and not self.model_folder_2:
            raise ValueError("Stacking experiment requires 'model_folder_2' argument.")

        self.predictor = nnUNetPredictor(
            tile_step_size=0.5, 
            use_gaussian=self.use_tta,
            use_mirroring=self.use_tta,
            perform_everything_on_device=True,
            device=self.device_obj,
            verbose=False, 
            verbose_preprocessing=False, 
            allow_tqdm=False 
        )
        self.preprocessor_class = None
        self.viz_cache = [] 
        self.splits = None

    def _cpu_preprocess_worker(self, args):
        a, p, preprocessor, plans, config, dataset_json, target_val = args
        p_for_prepro = p.copy()
        p_for_prepro['spacing'] = [float(target_val)] * len(p['spacing'])
        data, seg, p_out = preprocessor.run_case_npy(
            a, None, p_for_prepro, plans, config, dataset_json
        )
        p_out['spacing'] = p['spacing']
        return data, p_out

    def _custom_iterator(self, list_of_input_arrs, list_of_input_props):
        self.viz_cache = [] 
        preprocessor = self.preprocessor_class(verbose=False)
        plans = self.predictor.plans_manager
        configuration_manager = self.predictor.configuration_manager
        dataset_json = self.predictor.dataset_json
        target_spacing = list(configuration_manager.spacing)[::-1]
        target_val = target_spacing[0]

        # Per-slice CPU preprocessing is the volume-mode bottleneck (the GPU sits idle waiting on it).
        # Scale workers to the machine so the predictor stays fed (override with NNUNET_PREPROC_WORKERS).
        _workers = int(os.environ.get("NNUNET_PREPROC_WORKERS", min((os.cpu_count() or 8), 12)))
        with ThreadPoolExecutor(max_workers=_workers) as executor:
            futures = []
            for a, p in zip(list_of_input_arrs, list_of_input_props):
                args = (a, p, preprocessor, plans, configuration_manager, dataset_json, target_val)
                futures.append(executor.submit(self._cpu_preprocess_worker, args))

            for future in tqdm(futures):
                data, p_out = future.result() 
                with torch.no_grad(): 
                    data_tensor = torch.from_numpy(data).to(self.device_obj, dtype=torch.float32)
                    original_tensor = data_tensor.clone()

                    squeezed_dim = None
                    if data_tensor.ndim == 4:
                        if data_tensor.shape[1] == 1:
                            data_tensor_for_proc = data_tensor.squeeze(1) 
                            squeezed_dim = 1
                        else:
                            data_tensor_for_proc = data_tensor
                    else:
                        data_tensor_for_proc = data_tensor

                    if self.exp_mag_only and not self.exp_mag_one_channel:
                        data_tensor_for_proc[1, :, :] = data_tensor_for_proc[0, :, :]
                    
                    if self.exp_base_no_otsu:
                        processed_tensor = data_tensor_for_proc
                    else:
                        if self.exp_mag_one_channel: 
                            number_of_channels=1
                        else : 
                            number_of_channels=2
                            
                        processed_tensor = preprocess_gpu(
                            data_tensor_for_proc, mag_prepro=self.exp_mag_prepro, phase_prepro=self.exp_phase_prepro, is_3d=False
                        )

                    self.viz_cache.append({'orig': original_tensor.squeeze(1).cpu(), 'proc': processed_tensor.cpu()})

                    if self.exp_mosaic:
                        processed_tensor = _create_mosaic_tensor(processed_tensor)

                    if squeezed_dim is not None:
                        processed_tensor = processed_tensor.unsqueeze(squeezed_dim)
                    elif processed_tensor.ndim == 3:
                        processed_tensor = processed_tensor.unsqueeze(1)

                    yield {'data': processed_tensor.contiguous(), 'data_properties': p_out, 'ofile': None}

    def compute_metrics_and_visualize(self, pred_probs: np.ndarray, gt_mask: np.ndarray, viz_data: Dict, case_id: str, output_dir: str, target_labels: List[int] = None, spacing: tuple = None):
        """
        Modified to return RAW DISTANCES alongside slice-metrics for global aggregation.
        """
        metrics = {'case_id': case_id}
        # Dictionary to store raw distance arrays for this slice
        raw_dists = {}

        orig = viz_data['orig'].numpy()
        proc = viz_data['proc'].numpy()
        
        orig_mag = orig[0]
        orig_phase = orig[1] if orig.shape[0] > 1 else None
        
        pred_argmax = np.argmax(pred_probs, axis=0)

        viz_folder = join(output_dir, "visualizations")
        maybe_mkdir_p(viz_folder)
        
        # --- FIXED VISUALIZATION CALL ---
        try:
            save_path = join(viz_folder, f"{case_id}_viz.png")
            save_2d_slice_viz(
                vol_slice_mag=orig_mag,
                vol_slice_phase=orig_phase,
                gt_2d=gt_mask,
                pred_slice=pred_argmax,
                case_2d_id=case_id,
                output_path=save_path
            )
        except Exception as e:
            print(f"Error plotting {case_id}: {e}")
            plt.close('all')

        if gt_mask is not None:
            if target_labels is not None:
                labels = target_labels
            else:
                labels = self.predictor.label_manager.foreground_labels
            
            for label in labels:
                p_bin = (pred_argmax == label)
                g_bin = (gt_mask == label)
                
                # Use helper for Dice
                dice = compute_binary_dice_2d(p_bin, g_bin)
                metrics[f'dice_class_{label}'] = dice
                
                # Use helper for Raw Distances
                dists = compute_surface_distances_2d(p_bin, g_bin, spacing)
                
                # Use helper for HD95 calculation
                if dists is not None:
                     metrics[f'hd95_slice_class_{label}'] = calculate_hd95(dists)
                     raw_dists[label] = dists
                else:
                     metrics[f'hd95_slice_class_{label}'] = np.nan

            # Combined Logic
            def eval_combo(label_list, name_suffix):
                mask_p = np.isin(pred_argmax, label_list)
                mask_g = np.isin(gt_mask, label_list)
                
                d_val = compute_binary_dice_2d(mask_p, mask_g)
                dst_val = compute_surface_distances_2d(mask_p, mask_g, spacing)
                
                metrics[f'dice_{name_suffix}'] = d_val
                if dst_val is not None:
                    metrics[f'hd95_slice_{name_suffix}'] = calculate_hd95(dst_val)
                    raw_dists[name_suffix] = dst_val
                else:
                    metrics[f'hd95_slice_{name_suffix}'] = np.nan

            if 1 in labels and 2 in labels:
                eval_combo([1, 2], "combined_1_2")
            if 3 in labels and 4 in labels:
                eval_combo([3, 4], "combined_3_4")

        return metrics, raw_dists

    def run(self, mode: str, input_path: str = None, output_root: str = None, rel_path: str = ""):
        exp_suffix = "_TTA" if self.use_tta else ""
        print(f"Running Inference | Mode: {mode} | Experiment: {self.experiment_name}")
        
        if output_root:
            base_output_dir = Path(str(output_root) + exp_suffix)
        else:
            folder_name = f"inference_results_{self.experiment_name}{exp_suffix}"
            base_output_dir = join(self.model_folder, folder_name, mode)
            
        maybe_mkdir_p(base_output_dir)
        io = SimpleITKIO()
        data_pool = self._load_data(mode, input_path, io)
        
        splits = None
        if mode == 'validation':
            if self.splits is None:
                # Try standard path
                path_to_splits = self.model_folder.replace('nnunet_results', 'nnunet_raw')
                if 'nnUNetTrainer' in path_to_splits: path_to_splits = path_to_splits.split('nnUNetTrainer')[0]
                split_file = join(path_to_splits, 'splits_final.json')
                if not isfile(split_file): split_file = join(self.model_folder, 'splits_final.json')
                
                if isfile(split_file): splits = load_json(split_file)
                else: splits = [] 
            else:
                splits = self.splits

        if self.ensemble:
            inference_loops = [("ensemble_predictions", self.folds)]
        else:
            inference_loops = [(f"fold_{f}", (f,)) for f in self.folds]

        for folder_name, folds_to_use in tqdm(inference_loops, desc="Overall Progress"):
            print(f"\n--- Starting Inference Group: {folder_name} (Folds: {folds_to_use}) ---")
            
            if self.exp_stacking:
                passes = [
                    {'model': self.model_folder, 'is_primary': True},
                    {'model': self.model_folder_2, 'is_primary': False}
                ]
            else:
                passes = [{'model': self.model_folder, 'is_primary': False}]

            stacking_cache = {} 
            current_out_dir = join(base_output_dir, folder_name)
            maybe_mkdir_p(current_out_dir)
            
            if mode != 'volume':
                nifti_save_dir = join(current_out_dir, "predictions_nifti")
                maybe_mkdir_p(nifti_save_dir)

            if mode == 'validation' and splits is not None:
                target_keys = set()
                for f in folds_to_use:
                    if f < len(splits): target_keys.update(splits[f]['val'])
                target_keys = sorted([k for k in target_keys if k in data_pool])
            else:
                target_keys = sorted(list(data_pool.keys()))
            
            if not target_keys: continue

            current_images = [data_pool[k]['image'] for k in target_keys]
            current_props = [data_pool[k]['props'] for k in target_keys]
            current_gts = [data_pool[k]['gt'] for k in target_keys]
            current_gts_sec = [data_pool[k].get('gt_secondary', None) for k in target_keys]
            current_ids = target_keys.copy()
            volume_slices_reconstruction = []
            
            # --- AGGREGATORS FOR GLOBAL METRICS ---
            case_distance_accumulator = defaultdict(lambda: defaultdict(list))
            
            # --- INITIALIZE TRACKER ---
            tracker = MetricTracker()

            for pass_info in passes:
                model_path = pass_info['model']
                is_first_stack_pass = pass_info['is_primary'] and self.exp_stacking
                
                self.predictor.initialize_from_trained_model_folder(
                    model_path, use_folds=folds_to_use, checkpoint_name='checkpoint_best.pth'
                )
                self.predictor.verbose = False
                self.predictor.allow_tqdm = False 
                self.preprocessor_class = self.predictor.configuration_manager.preprocessor_class
                
                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    inference_gen = self.predictor.predict_from_data_iterator(
                        self._custom_iterator(current_images, current_props), 
                        save_probabilities=True, 
                        num_processes_segmentation_export=2 
                    )

                for i, prediction_tuple in tqdm(enumerate(inference_gen), total=len(current_images)):
                    if isinstance(prediction_tuple, (list, tuple)):
                        pred_seg, pred_probs = prediction_tuple
                    else:
                        pred_seg, pred_probs = prediction_tuple, None
                    
                    case_id = current_ids[i]
                    if "_slice" in case_id:
                        volume_id = case_id.split("_slice")[0]
                    else:
                        volume_id = case_id 

                    current_labels_for_metrics = None
                    gt_for_metrics = current_gts[i]

                    if self.exp_stacking:
                        if is_first_stack_pass:
                            stacking_cache[case_id] = pred_seg.astype(np.uint8)
                            continue 
                        else:
                            seg_anat = stacking_cache.get(case_id)
                            final_seg = _merge_masks(seg_anat, pred_seg)
                            pred_seg = final_seg
                            
                            num_classes_fused = 5 
                            shape = (num_classes_fused,) + pred_probs.shape[1:]
                            fused_probs = np.zeros(shape, dtype=np.float32)
                            for c in range(num_classes_fused):
                                fused_probs[c][final_seg == c] = 1.0
                            pred_probs = fused_probs
                            current_labels_for_metrics = [1, 2, 3, 4]
                            
                            gt_anat = current_gts[i]
                            gt_path = current_gts_sec[i]
                            if gt_anat is not None and gt_path is not None:                                
                                gt_for_metrics = _merge_masks(gt_anat, gt_path)
                            else:
                                gt_for_metrics = None

                    if self.exp_mosaic:
                        h_orig = pred_seg.shape[-2] // 2
                        w_orig = pred_seg.shape[-1] // 2
                        pred_seg = pred_seg[..., 0:h_orig, 0:w_orig]
                        if pred_probs is not None: pred_probs = pred_probs[..., 0:h_orig, 0:w_orig]

                    if mode == 'volume':
                        volume_slices_reconstruction.append(pred_seg)
                        continue 

                    io.write_seg(pred_seg.astype(np.uint8), join(nifti_save_dir, f"{case_id}.nii.gz"), current_props[i])
                    
                    viz_data = self.viz_cache[i]
                    if pred_probs is not None:
                        if isinstance(pred_probs, torch.Tensor): pred_probs = pred_probs.cpu().numpy()
                        spacing = current_props[i].get('spacing', (1.0, 1.0))
                        
                        # --- Metrics Calculation ---
                        metrics, raw_slice_dists = self.compute_metrics_and_visualize(
                            pred_probs, gt_for_metrics, viz_data, case_id, current_out_dir, 
                            target_labels=current_labels_for_metrics, spacing=spacing
                        )
                        
                        # FIX: Add metrics to tracker
                        tracker.add_case_metric(metrics)
                        
                        if gt_for_metrics is not None:
                            # Accumulate distances for Global HD95
                            for lbl_key, dist_arr in raw_slice_dists.items():
                                case_distance_accumulator[volume_id][lbl_key].append(dist_arr)
                            
                            pred_hard = np.argmax(pred_probs, axis=0)
                            
                            # --- CRITICAL FIX START: DEFINING fg_labels ---
                            if current_labels_for_metrics is not None:
                                fg_labels = current_labels_for_metrics
                            elif hasattr(self.predictor, 'label_manager'):
                                fg_labels = self.predictor.label_manager.foreground_labels
                            else:
                                fg_labels = [1, 2, 3, 4]
                            
                            # Use fg_labels here, NOT current_labels_for_metrics
                            tracker.update_counts(pred_hard, gt_for_metrics, fg_labels)
                            
                            if 3 in fg_labels and 4 in fg_labels:
                                tracker.update_counts_combo(pred_hard, gt_for_metrics, [3, 4], "Combined_3_4")

                            if 1 in fg_labels and 2 in fg_labels:
                                tracker.update_counts_combo(pred_hard, gt_for_metrics, [1, 2], "Combined_1_2")
                            # --- CRITICAL FIX END ---

                gc.collect()
                torch.cuda.empty_cache()

            # --- POST-PROCESSING: AGGREGATE METRICS ---
            if mode != 'volume':
                # 1. Save Casewise Metrics (Renamed to match inference_3d.py)
                tracker.save_casewise_csv(join(current_out_dir, 'metrics_2d_casewise.csv'))
                
                # 2. Compute Extra Stats from case metrics (e.g. mean HD95)
                extra_stats = {}
                if tracker.case_metrics:
                    df_case = pd.DataFrame(tracker.case_metrics)
                    for col in df_case.columns:
                        if 'hd95_slice_' in col:
                            # Extract label and compute mean
                            clean_lbl = col.replace('hd95_slice_', '')
                            # Strip 'class_' if present
                            clean_lbl = clean_lbl.replace('class_', '')
                            extra_stats[clean_lbl] = df_case[col].mean()

                # 3. Save Global Summary (Renamed to match inference_3d.py)
                tracker.save_global_summary_csv(
                    join(current_out_dir, 'metrics_2d_global.csv'),
                    extra_metrics_dict=extra_stats
                )

            if mode == 'volume' and volume_slices_reconstruction:
                print(f"Reconstructing volume...")
                full_volume = np.stack(volume_slices_reconstruction, axis=0)
                if full_volume.ndim == 4 and full_volume.shape[1] == 1:
                    full_volume = full_volume[:, 0, :, :]
                
                first_key = current_ids[0]
                exact_filename = data_pool[first_key]['orig_filename']
                final_save_dir = join(current_out_dir, rel_path)
                maybe_mkdir_p(final_save_dir)
                save_path = join(final_save_dir, exact_filename)
                
                if self.volume_props:
                    io.write_seg(full_volume.astype(np.uint8), save_path, self.volume_props)
                    print(f"✅ Saved full volume: {save_path}")

        # --- FINAL CROSS-FOLD SUMMARY (Matches inference_3d.py format) ---
        if not self.ensemble:
            print("\nComputing Cross-Fold Statistics from saved CSVs on disk...")
            # Look for folders starting with 'fold_' in the output root
            fold_dirs = [join(base_output_dir, d) for d in os.listdir(base_output_dir) if os.path.isdir(join(base_output_dir, d)) and d.startswith('fold_')]
            
            agg_means_casewise = []
            agg_global = []

            for f_dir in fold_dirs:
                fold_name = os.path.basename(f_dir)
                
                # 1. Load Casewise Metrics (metrics_2d_casewise.csv)
                p_case = join(f_dir, "metrics_2d_casewise.csv")
                if isfile(p_case):
                    df = pd.read_csv(p_case)
                    # Calculate mean of all cases in this fold
                    mean_dict = df.mean(numeric_only=True).to_dict()
                    mean_dict['Fold'] = fold_name
                    agg_means_casewise.append(mean_dict)
                
                # 2. Load Global Metrics (metrics_2d_global.csv)
                p_glob = join(f_dir, "metrics_2d_global.csv")
                if isfile(p_glob):
                    df = pd.read_csv(p_glob)
                    glob_dict = {}
                    for _, row in df.iterrows():
                        # Handle Global Dice
                        if 'Global_Dice' in row:
                            glob_dict[f"Dice_{row['Label']}"] = row['Global_Dice']
                        # Handle Extra metrics (like Global HD95)
                        if 'Global_Metric_Extra' in row and pd.notna(row['Global_Metric_Extra']):
                             glob_dict[f"HD95_{row['Label']}"] = row['Global_Metric_Extra']
                             
                    glob_dict['Fold'] = fold_name
                    agg_global.append(glob_dict)

            # --- SAVE SUMMARY ---
            def save_summary(agg_list, suffix):
                if not agg_list: return
                df_agg = pd.DataFrame(agg_list)
                
                # Compute Mean and Std across folds
                means = df_agg.mean(numeric_only=True)
                stds = df_agg.std(numeric_only=True)
                
                summary_data = {}
                for key in means.index:
                    # Parse keys like 'Dice_1', 'HD95_3', 'Dice_combined_1_2'
                    parts = key.split('_')
                    if len(parts) < 2: continue
                    
                    metric_type = parts[0] # Dice or HD95
                    label_part = "_".join(parts[1:]).replace("slice_", "") # '1', 'combined_1_2'
                        
                    combo_key = f"{metric_type}_{label_part}"
                    
                    if label_part not in summary_data: summary_data[label_part] = {'Label': label_part}
                    summary_data[label_part][f'{metric_type}_Mean'] = means[key]
                    summary_data[label_part][f'{metric_type}_Std'] = stds[key]
                
                final_df = pd.DataFrame(list(summary_data.values()))
                if not final_df.empty:
                    # Sort nicely: Integers first, then strings
                    final_df['sort_helper'] = final_df['Label'].apply(lambda x: int(x) if str(x).isdigit() else 999)
                    final_df = final_df.sort_values('sort_helper').drop('sort_helper', axis=1)
                    
                    out_name = join(base_output_dir, f"summary_cross_fold_{suffix}.csv")
                    final_df.to_csv(out_name, index=False)
                    print(f"   Saved Cross-Fold Summary: {os.path.basename(out_name)}")

            save_summary(agg_means_casewise, "casewise_2d")
            save_summary(agg_global, "global_2d")
        
        else:
            # --- ENSEMBLE SUMMARY ---
            print("\nComputing Ensemble Statistics (Across Cases)...")
            ensemble_dir = join(base_output_dir, "ensemble_predictions")
            
            p_case = join(ensemble_dir, "metrics_2d_casewise.csv")
            if isfile(p_case):
                df = pd.read_csv(p_case)
                # Compute Mean and Std across cases directly
                means = df.mean(numeric_only=True)
                stds = df.std(numeric_only=True)
                
                summary_data = {}
                for key in means.index:
                    # Parse keys like 'dice_class_1', 'hd95_slice_class_3'
                    parts = key.split('_') 
                    
                    if len(parts) < 2: continue 
                    
                    metric_type = parts[0] # dice, hd95
                    
                    # Extract label part
                    # If key is 'hd95_slice_class_1': parts=['hd95', 'slice', 'class', '1'] -> join(1:) = slice_class_1 -> replace -> class_1
                    # If key is 'dice_class_1': parts=['dice', 'class', '1'] -> join(1:) = class_1
                    
                    label_part = "_".join(parts[1:])
                    label_part = label_part.replace("slice_", "") # Normalize 'slice_class_1' to 'class_1'
                    
                    if label_part not in summary_data: summary_data[label_part] = {'Label': label_part}
                    summary_data[label_part][f'{metric_type}_Mean'] = means[key]
                    summary_data[label_part][f'{metric_type}_Std'] = stds[key]

                final_df = pd.DataFrame(list(summary_data.values()))
                if not final_df.empty:
                    final_df['sort_helper'] = final_df['Label'].apply(lambda x: int(x) if str(x).isdigit() else 999)
                    final_df = final_df.sort_values('sort_helper').drop('sort_helper', axis=1)
                    
                    # Save INSIDE ensemble_dir
                    out_name = join(ensemble_dir, "summary_cross_fold_casewise_2d.csv")
                    final_df.to_csv(out_name, index=False)
                    print(f"   Saved Ensemble Summary: {os.path.basename(out_name)}")
            
            # --- GLOBAL METRICS (Ensemble) ---
            p_glob = join(ensemble_dir, "metrics_2d_global.csv")
            if isfile(p_glob):
                df = pd.read_csv(p_glob)
                summary = []
                # metrics_2d_global.csv usually has columns: Label, Global_Dice, (maybe Global_Metric_Extra for HD95?)
                # We just rename them to match the "cross-fold" summary format for consistency
                # Cross-Fold format: Label, Dice_Mean, Dice_Std
                # Since we only have 1 run (Ensemble), Std is 0 or NaN, but let's just output the value as Mean.
                
                for _, row in df.iterrows():
                    entry = {'Label': row['Label']}
                    if 'Global_Dice' in row:
                        entry['Dice_Mean'] = row['Global_Dice']
                        entry['Dice_Std'] = 0.0
                    
                    if 'Global_Metric_Extra' in row and pd.notna(row['Global_Metric_Extra']):
                        entry['HD95_Mean'] = row['Global_Metric_Extra']
                        entry['HD95_Std'] = 0.0

                    summary.append(entry)
                
                final_df_glob = pd.DataFrame(summary)
                if not final_df_glob.empty:
                    final_df_glob['sort_helper'] = final_df_glob['Label'].apply(lambda x: int(x) if str(x).isdigit() else 999)
                    final_df_glob = final_df_glob.sort_values('sort_helper').drop('sort_helper', axis=1)
                    
                    # Save INSIDE ensemble_dir
                    out_name_glob = join(ensemble_dir, "summary_cross_fold_global_2d.csv")
                    final_df_glob.to_csv(out_name_glob, index=False)
                    print(f"   Saved Ensemble Global Summary: {os.path.basename(out_name_glob)}")
                
        print("\nAll inference loops completed.")

    def _load_data(self, mode, input_path, io):
        data_pool = {}
        secondary_gt_root = None
        if self.exp_stacking and self.model_folder_2:
            raw_path_2 = _get_raw_data_folder_from_model_folder(self.model_folder_2)
            if raw_path_2:
                secondary_gt_root = join(raw_path_2, 'labelsTr')
                if not os.path.isdir(secondary_gt_root): secondary_gt_root = join(raw_path_2, 'labelsTs')                    
        
        if mode == 'validation':
            # Prefer the explicitly passed -path (an nnUNet_raw dataset dir with imagesTr/labelsTr/splits_final.json);
            # fall back to deriving it from the model folder (legacy nnunet_raw-beside-nnunet_results layout).
            if not (input_path and isfile(join(input_path, 'splits_final.json'))):
                input_path = self.model_folder.replace('nnunet_results', 'nnunet_raw').replace('nnUNet2DCustomTrainer__nnUNetPlans__2d', '')
                if OLD_TRAINER_NAME: input_path = self.model_folder.replace('nnunet_results', 'nnunet_raw').replace('nnUNetTrainerWandb__nnUNetPlans__2d', '')
                if 'nnUNetTrainer' in input_path: input_path = input_path.split('nnUNetTrainer')[0]
            split_file = join(input_path, 'splits_final.json')
            if not isfile(split_file): split_file = join(self.model_folder, 'splits_final.json')
            
            if isfile(split_file):
                self.splits = load_json(split_file)
                val_keys = set()
                for f in self.folds:
                    if f < len(self.splits): val_keys.update(self.splits[f]['val'])
                val_keys = sorted(list(val_keys))
                print(f"Loading {len(val_keys)} cases...")
                for k in val_keys:
                    img_path = join(input_path, 'imagesTr', f"{k}_0000.nii.gz")
                    phase_path = join(input_path, 'imagesTr', f"{k}_0001.nii.gz")
                    gt_path = join(input_path, 'labelsTr', f"{k}.nii.gz")
                    
                    if self.exp_mag_one_channel:
                         if not isfile(img_path): continue
                         img, prop = io.read_images([img_path])
                    else:
                        if not (isfile(img_path) and isfile(phase_path)): continue
                        img, prop = io.read_images([img_path, phase_path])

                    gt, _ = io.read_images([gt_path])
                    entry = {'image': img, 'props': prop, 'gt': gt[0]}
                    if secondary_gt_root:
                        sec_gt_path = join(secondary_gt_root, f"{k}.nii.gz".replace("wm_gm_segmentation", "merge_lesions"))
                        if isfile(sec_gt_path):
                             gt2, _ = io.read_images([sec_gt_path])
                             entry['gt_secondary'] = gt2[0]
                    data_pool[k] = entry
            else: print("Critical: splits_final.json missing.")

        elif mode == 'test':
            files = sorted([f for f in os.listdir(input_path) if f.endswith('.nii.gz') and "_0000" in f])
            for f in files:
                case_id = f.replace('_0000.nii.gz', '')
                c0 = join(input_path, f); c1 = join(input_path, f.replace('_0000', '_0001'))
                channel_files = [c0] if self.exp_mag_one_channel else ([c0, c1] if isfile(c1) else [c0])
                img, prop = io.read_images(channel_files)
                gt = None
                gt_path_ts = join(input_path, '../labelsTs', f"{case_id}.nii.gz")
                gt_path_tr = join(input_path, '../labelsTr', f"{case_id}.nii.gz")
                gt_path = gt_path_ts if isfile(gt_path_ts) else (gt_path_tr if isfile(gt_path_tr) else None)

                if gt_path:
                     gt_arr, _ = io.read_images([gt_path])
                     gt = gt_arr[0]
                entry = {'image': img, 'props': prop, 'gt': gt}
                if secondary_gt_root and gt is not None:
                     sec_gt_path = join(secondary_gt_root, f"{case_id}.nii.gz")
                     if not isfile(sec_gt_path): sec_gt_path = secondary_gt_root.replace('labelsTr', 'labelsTs') + f"/{case_id}.nii.gz".replace("wm_gm_segmentation", "merge_lesions")
                     if isfile(sec_gt_path):
                         gt2, _ = io.read_images([sec_gt_path])
                         entry['gt_secondary'] = gt2[0]
                data_pool[case_id] = entry

        elif mode == 'volume':
            mag_path = input_path
            if os.path.isdir(mag_path):
                candidates = [f for f in os.listdir(mag_path) if "mag" in f and f.endswith(".nii.gz")]
                if candidates: mag_path = join(mag_path, candidates[0])
            phase_path = mag_path.replace("mag", "phase")
            
            if self.exp_mag_one_channel:
                vol, vol_props = io.read_images([mag_path])
            else:
                vol, vol_props = io.read_images([mag_path, phase_path])
            
            self.volume_props = vol_props.copy()
            self.volume_props['spacing'] = list(self.volume_props['spacing']) 
            d = vol.shape[1]
            exact_filename = os.path.basename(mag_path)
            for i in range(d):
                slice_img = vol[:, i:i+1, :, :]
                case_id = f"{exact_filename.replace('.nii.gz', '')}_slice{i:03d}"
                slice_props = vol_props.copy()
                slice_props['shape_after_cropping_and_before_resampling'] = slice_img.shape[1:]
                slice_props['shape_before_cropping'] = slice_img.shape[1:]
                slice_props['spacing'] = list(slice_props['spacing'])
                data_pool[case_id] = {'image': slice_img, 'props': slice_props, 'gt': None, 'orig_filename': exact_filename}

        return data_pool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment', type=str, required=True)
    parser.add_argument('-model_folder', type=str, required=True)
    parser.add_argument('-model_folder_2', type=str, required=False)
    parser.add_argument('-mode', type=str, choices=['validation', 'test', 'volume'], required=True)
    parser.add_argument('-path', type=str, required=True)
    parser.add_argument('-folds', nargs='+', type=int, default=[0])
    parser.add_argument('--use_tta', action='store_true')
    parser.add_argument('-output_root', type=str, required=False)
    parser.add_argument('-rel_path', type=str, default="")
    parser.add_argument('--ensemble', action='store_true')
    
    args = parser.parse_args()
    
    runner = CustomInferenceRunner(
        args.model_folder, args.experiment, tuple(args.folds), 
        device='cuda', use_tta=args.use_tta, ensemble=args.ensemble,
        model_folder_2=args.model_folder_2
    )
    runner.run(args.mode, args.path, output_root=args.output_root, rel_path=args.rel_path)