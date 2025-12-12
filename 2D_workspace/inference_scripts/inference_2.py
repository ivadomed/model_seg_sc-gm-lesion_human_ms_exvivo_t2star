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
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, load_json, isfile
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from concurrent.futures import ThreadPoolExecutor

# =================================================================================================
# HELPER FUNCTIONS
# =================================================================================================

def _preprocess_data_gpu(data_batch: torch.Tensor, mag_prepro: bool = False, phase_prepro: bool = False, number_of_channels: int = 2) -> torch.Tensor:
    """Applies custom preprocessing DIRECTLY ON THE GPU using Kornia and PyTorch."""
    was_3dim = False
    if len(data_batch.size()) == 3: 
        was_3dim = True
        data_batch = data_batch.unsqueeze(0)
    
    # Determine dims
    channel_dim = 0 if data_batch.size()[0] == number_of_channels else 1
    num_channels = data_batch.size()[channel_dim]

    # --- HANDLE CHANNELS ---
    phase_batch = None
    
    if num_channels == 2:
        if channel_dim == 0:
            mag_batch = data_batch[0:1, :, :, :]
            phase_batch = data_batch[1:2, :, :, :]
        else:
            mag_batch = data_batch[:, 0:1, :, :]
            phase_batch = data_batch[:, 1:2, :, :]
    else:
        # Single channel case (Magnitude only)
        if channel_dim == 0:
            mag_batch = data_batch[0:1, :, :, :]
        else:
            mag_batch = data_batch[:, 0:1, :, :]

    # --- PROCESS MAGNITUDE (Otsu) ---
    thresholds_tensor = kornia.filters.otsu_threshold(mag_batch)[1]
    reshaped_thresholds = thresholds_tensor.view(-1, 1, 1, 1)
    mask = (mag_batch >= reshaped_thresholds).float()

    kernel = torch.ones(3, 3, device=mask.device) 
    mask = kornia.morphology.opening(mask, kernel)

    processed_mag = mag_batch * mask
    
    if phase_batch is not None:
        processed_phase = phase_batch * mask
    else:
        processed_phase = None

    if mag_prepro:
        mag_min = processed_mag.amin(dim=(-2, -1), keepdim=True)
        mag_max = processed_mag.amax(dim=(-2, -1), keepdim=True)
        mag_normalized = (processed_mag - mag_min) / (mag_max - mag_min + 1e-6)
        mag_clahe = kornia.enhance.equalize_clahe(mag_normalized)
        processed_mag = mag_clahe * mask

    if phase_prepro and processed_phase is not None:
        safe_mask = mask.bool()
        if safe_mask.any():
            masked_phase_values = torch.masked_select(phase_batch, safe_mask)
            p30 = torch.quantile(masked_phase_values, 0.30)
            p85 = torch.quantile(masked_phase_values, 0.85)
        else:
            p30, p85 = 0.0, 1.0

        phase_rescaled = (phase_batch - p30) / (p85 - p30 + 1e-6)
        phase_rescaled = torch.clamp(phase_rescaled, 0, 1)
        processed_phase = phase_rescaled * mask

    # --- RECOMBINE ---
    if processed_phase is not None:
        if channel_dim == 0:
            processed_batch = torch.cat([processed_mag, processed_phase], dim=0)
        else:
            processed_batch = torch.cat([processed_mag, processed_phase], dim=1)
    else:
        processed_batch = processed_mag

    if was_3dim: 
        processed_batch = processed_batch.squeeze(0)
    return processed_batch

def decode_bitmask_to_7_channels(bitmask: np.ndarray, num_classes: int) -> np.ndarray:
    """Robust decoder that handles various input dimensions."""
    if bitmask is None:
        return None
        
    # Standardize to (H, W)
    if bitmask.ndim == 3 and bitmask.shape[0] == 1:
        bitmask = bitmask[0]
    elif bitmask.ndim == 4 and bitmask.shape[0] == 1 and bitmask.shape[1] == 1:
        bitmask = bitmask[0][0]
    
    if bitmask.ndim != 2:
        print(f"Warning: decode_bitmask got weird shape {bitmask.shape}")
        return np.zeros((num_classes + 1, bitmask.shape[-2], bitmask.shape[-1]), dtype=np.uint8)

    h, w = bitmask.shape
    multi_channel_mask = np.zeros((num_classes + 1, h, w), dtype=np.uint8)
    
    # Fill channels
    for i in range(num_classes + 1):
        multi_channel_mask[i] = (bitmask == i).astype(np.uint8)
        
    return multi_channel_mask

def plot_comparison_and_segmentation(orig_mag, orig_phase, proc_mag, proc_phase, gt_mask, pred_mask, case_id):
    """
    Updated visualization:
    Rows 1-2: Inputs (Mag/Phase)
    Rows 3+: Per Class Breakdown (Left: GT, Right: Pred)
    """
    # Define classes to visualize (assuming 1, 2, 3, 4 based on stacking logic)
    classes_to_plot = [1, 2, 3, 4]
    class_names = {1: "WM/Class 1", 2: "GM/Class 2", 3: "Lesion in WM", 4: "Lesion in GM"}
    
    num_rows = 2 + len(classes_to_plot) # Inputs + one row per class
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, 4 * num_rows))
    fig.suptitle(f'Case: {case_id} | Left: Ground Truth / Right: Prediction', fontsize=16)

    # Helper
    def show_img(ax, img, title, cmap='gray', mask=None, color='red'):
        if img is not None:
            ax.imshow(img.T, cmap=cmap)
        else:
            # Handle missing channel visualization
            ax.text(0.5, 0.5, "Channel Not Available", ha='center', va='center')
        
        if mask is not None and img is not None:
            # Mask overlay
            masked = np.ma.masked_where(mask == 0, mask)
            ax.imshow(masked.T, cmap=color, alpha=0.6, interpolation='none')
        ax.set_title(title)
        ax.axis('off')

    # Row 0: Original Inputs
    show_img(axs[0, 0], orig_mag, "Original Magnitude")
    show_img(axs[0, 1], orig_phase, "Original Phase")
    
    # Row 1: Preprocessed Inputs
    show_img(axs[1, 0], proc_mag, "Preprocessed Magnitude")
    show_img(axs[1, 1], proc_phase, "Preprocessed Phase")

    # Rows 2+: Per Class
    bg = proc_mag if proc_mag is not None else orig_mag

    # Ensure masks are 2D (H, W)
    if gt_mask is not None and gt_mask.ndim == 3: gt_mask = gt_mask[0]
    if pred_mask.ndim == 3: pred_mask = pred_mask[0]

    for i, cls_idx in enumerate(classes_to_plot):
        row = i + 2
        cls_name = class_names.get(cls_idx, f"Class {cls_idx}")
        
        # GT Column
        if gt_mask is not None:
            gt_bin = (gt_mask == cls_idx).astype(np.uint8)
            show_img(axs[row, 0], bg, f"GT: {cls_name}", mask=gt_bin, color='Greens_r')
        else:
            show_img(axs[row, 0], bg, f"GT: {cls_name} (Missing)")

        # Pred Column
        pred_bin = (pred_mask == cls_idx).astype(np.uint8)
        show_img(axs[row, 1], bg, f"Pred: {cls_name}", mask=pred_bin, color='Reds_r')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig

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
    """
    Centralized logic for merging Anatomy and Pathology masks.
    Used for both GT and Predictions to ensure consistency.
    """
    merged = np.zeros_like(anat)
    
    # Default assumptions based on your script
    # Adjust these if your model outputs differ!
    # ANAT: 1=WM, 2=GM
    mask_wm = (anat == 1)
    mask_gm = (anat == 2)
    
    # PATH: Detect label. 
    # If path mask has 2s, we assume 2 is lesion. If only 1s, we assume 1 is lesion.
    lesion_label = 2
    uniques = np.unique(path)
    if 2 not in uniques :
        lesion_label = 999
            
    mask_lesion = (path == lesion_label)
    
    # Merge Logic
    # 1: WM (Healthy)
    # 2: GM (Healthy)
    # 3: Lesion in WM
    # 4: Lesion in GM
    
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
        # New Flag for One-Channel Magnitude Experiment
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

        with ThreadPoolExecutor(max_workers=3) as executor:
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

                    # If experiment is explicitly single channel (new model), DO NOT duplicate.
                    if self.exp_mag_only and not self.exp_mag_one_channel:
                        # Old logic: duplicate mag into phase channel
                        data_tensor_for_proc[1, :, :] = data_tensor_for_proc[0, :, :]
                    
                    if self.exp_base_no_otsu:
                        processed_tensor = data_tensor_for_proc
                    else:
                        # Modified GPU preprocessor now handles 1-channel input gracefully
                        if self.exp_mag_one_channel: 
                            number_of_channels=1
                        else : 
                            number_of_channels=2
                            
                        processed_tensor = _preprocess_data_gpu(
                            data_tensor_for_proc, mag_prepro=self.exp_mag_prepro, phase_prepro=self.exp_phase_prepro, number_of_channels=number_of_channels
                        )


                    self.viz_cache.append({'orig': original_tensor.squeeze(1).cpu(), 'proc': processed_tensor.cpu()})

                    if self.exp_mosaic:
                        processed_tensor = _create_mosaic_tensor(processed_tensor)

                    if squeezed_dim is not None:
                        processed_tensor = processed_tensor.unsqueeze(squeezed_dim)
                    elif processed_tensor.ndim == 3:
                        processed_tensor = processed_tensor.unsqueeze(1)

                    yield {'data': processed_tensor.contiguous(), 'data_properties': p_out, 'ofile': None}

    def compute_metrics_and_visualize(self, pred_probs: np.ndarray, gt_mask: np.ndarray, viz_data: Dict, case_id: str, output_dir: str, target_labels: List[int] = None):
        metrics = {'case_id': case_id}
        
        orig = viz_data['orig'].numpy()
        proc = viz_data['proc'].numpy()
        
        # Safely extract magnitude and phase (handle missing phase for 1-channel exp)
        orig_mag = orig[0]
        orig_phase = orig[1] if orig.shape[0] > 1 else None
        
        proc_mag = proc[0]
        proc_phase = proc[1] if proc.shape[0] > 1 else None
        
        pred_argmax = np.argmax(pred_probs, axis=0)

        viz_folder = join(output_dir, "visualizations")
        maybe_mkdir_p(viz_folder)
        
        try:
            fig = plot_comparison_and_segmentation(
                orig_mag, orig_phase, proc_mag, proc_phase,
                gt_mask, pred_argmax, case_id
            )
            plt.savefig(join(viz_folder, f"{case_id}_viz.png"))
            plt.close(fig)
        except Exception as e:
            print(f"Error plotting {case_id}: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')

        if gt_mask is not None:
            if target_labels is not None:
                labels = target_labels
            else:
                labels = self.predictor.label_manager.foreground_labels
            
            for label in labels:
                p_bin = (pred_argmax == label)
                g_bin = (gt_mask == label)
                gt_exists = g_bin.sum() > 0
                pred_exists = p_bin.sum() > 0
                
                if not gt_exists:
                    dice = np.nan if not pred_exists else 0.0
                else:
                    intersection = np.logical_and(p_bin, g_bin).sum()
                    union = p_bin.sum() + g_bin.sum()
                    dice = (2. * intersection) / (union + 1e-8)
                metrics[f'dice_class_{label}'] = dice

            # Hardcoded combinations based on your script's logic
            if 1 in labels and 2 in labels:
                p_comb_12 = (pred_argmax == 1) | (pred_argmax == 2)
                g_comb_12 = (gt_mask == 1) | (gt_mask == 2)
                gt_comb_12_exists = g_comb_12.sum() > 0
                pred_comb_12_exists = p_comb_12.sum() > 0
                
                if not gt_comb_12_exists:
                    dice_12 = np.nan if not pred_comb_12_exists else 0.0
                else:
                    inter_12 = np.logical_and(p_comb_12, g_comb_12).sum()
                    union_12 = p_comb_12.sum() + g_comb_12.sum()
                    dice_12 = (2. * inter_12) / (union_12 + 1e-8)
                metrics['dice_combined_1_2'] = dice_12

            if 3 in labels and 4 in labels:
                p_comb_34 = (pred_argmax == 3) | (pred_argmax == 4)
                g_comb_34 = (gt_mask == 3) | (gt_mask == 4)
                gt_comb_34_exists = g_comb_34.sum() > 0
                pred_comb_34_exists = p_comb_34.sum() > 0
                
                if not gt_comb_34_exists:
                    dice_34 = np.nan if not pred_comb_34_exists else 0.0
                else:
                    inter_34 = np.logical_and(p_comb_34, g_comb_34).sum()
                    union_34 = p_comb_34.sum() + g_comb_34.sum()
                    dice_34 = (2. * inter_34) / (union_34 + 1e-8)
                metrics['dice_combined_3_4'] = dice_34

        return metrics

    def run(self, mode: str, input_path: str = None, output_root: str = None, rel_path: str = ""):
        exp_suffix = "_TTA" if self.use_tta else ""
        
        print(f"Running Inference | Mode: {mode} | Experiment: {self.experiment_name} | TTA: {self.use_tta} | Ensemble: {self.ensemble}")
        
        if output_root:
            base_output_dir = Path(str(output_root) + exp_suffix)
        else:
            folder_name = f"inference_results_{self.experiment_name}{exp_suffix}"
            base_output_dir = join(self.model_folder, folder_name, mode)
            
        maybe_mkdir_p(base_output_dir)
        print(f"Outputs will be saved to: {base_output_dir}")

        io = SimpleITKIO()
        data_pool = self._load_data(mode, input_path, io)
        
        splits = None
        if mode == 'validation':
            if self.splits is None:
                path_to_splits = self.model_folder.replace('nnunet_results', 'nnunet_raw').replace('nnUNetTrainerWandb__nnUNetPlans__2d', '')
                if 'nnUNetTrainer' in path_to_splits: path_to_splits = path_to_splits.split('nnUNetTrainer')[0]
                # Try standard path
                split_file = join(path_to_splits, 'splits_final.json')
                if not isfile(split_file):
                    # Fallback try to find splits inside model folder (sometimes saved there)
                    split_file = join(self.model_folder, 'splits_final.json')
                
                if isfile(split_file):
                    splits = load_json(split_file)
                else:
                    print("Warning: splits_final.json not found. Validation mode might fail or run on all data.")
                    splits = [] # Handle gracefully
            else:
                splits = self.splits

        metrics_history = defaultdict(lambda: defaultdict(list))
        fold_aggregates_history = [] 
        global_dice_history = defaultdict(list)

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
            current_global_counts = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0})
            group_slice_metrics = []
            
            current_out_dir = join(base_output_dir, folder_name)
            maybe_mkdir_p(current_out_dir)
            
            if mode != 'volume':
                nifti_save_dir = join(current_out_dir, "predictions_nifti")
                maybe_mkdir_p(nifti_save_dir)

            if mode == 'validation' and splits is not None:
                target_keys = set()
                for f in folds_to_use:
                    if f < len(splits):
                        target_keys.update(splits[f]['val'])
                target_keys = sorted(list(target_keys))
                target_keys = [k for k in target_keys if k in data_pool]
                print(f"Group {folder_name}: Processing {len(target_keys)} validation cases.")
            else:
                target_keys = sorted(list(data_pool.keys()))
                print(f"Group {folder_name}: Processing all {len(target_keys)} cases.")
            
            if not target_keys:
                print(f"No cases found. Skipping.")
                continue

            current_images = [data_pool[k]['image'] for k in target_keys]
            current_props = [data_pool[k]['props'] for k in target_keys]
            current_gts = [data_pool[k]['gt'] for k in target_keys]
            current_gts_sec = [data_pool[k].get('gt_secondary', None) for k in target_keys]
            current_ids = target_keys.copy()
            volume_slices_reconstruction = []

            for pass_info in passes:
                model_path = pass_info['model']
                is_first_stack_pass = pass_info['is_primary'] and self.exp_stacking
                
                print(f"  > Loading Model: {os.path.basename(model_path.rstrip('/'))} ...")
                
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

                desc = f"Inferring Model 1 (Stacking)" if is_first_stack_pass else "Inferring Final"
                
                for i, prediction_tuple in tqdm(enumerate(inference_gen), total=len(current_images), desc=desc):
                    if isinstance(prediction_tuple, (list, tuple)):
                        pred_seg, pred_probs = prediction_tuple
                    else:
                        pred_seg, pred_probs = prediction_tuple, None
                    
                    case_id = current_ids[i]
                    current_labels_for_metrics = None
                    gt_for_metrics = current_gts[i]

                    # --- STACKING LOGIC ---
                    if self.exp_stacking:
                        if is_first_stack_pass:
                            stacking_cache[case_id] = pred_seg.astype(np.uint8)
                            continue 
                        else:
                            seg_anat = stacking_cache.get(case_id)
                            seg_path = pred_seg 
                            
                            # Use centralized merger for prediction
                            final_seg = _merge_masks(seg_anat, seg_path)
                            pred_seg = final_seg
                            
                            # Fused Probabilities for Visualization
                            num_classes_fused = 5 
                            shape = (num_classes_fused,) + pred_probs.shape[1:]
                            fused_probs = np.zeros(shape, dtype=np.float32)
                            for c in range(num_classes_fused):
                                fused_probs[c][final_seg == c] = 1.0
                            pred_probs = fused_probs
                            current_labels_for_metrics = [1, 2, 3, 4]
                            
                            # -- GT FUSION --
                            gt_anat = current_gts[i]
                            gt_path = current_gts_sec[i]
                            
                            if gt_anat is not None and gt_path is not None:                                
                                # Use SAME centralized merger for GT
                                # This fixes the potential mismatch where logic differed!
                                gt_for_metrics = _merge_masks(gt_anat, gt_path)
                            else:
                                gt_for_metrics = None

                    if self.exp_mosaic:
                        h_full = pred_seg.shape[-2]
                        w_full = pred_seg.shape[-1]
                        h_orig = h_full // 2
                        w_orig = w_full // 2
                        if pred_seg.ndim == 2:
                            pred_seg = pred_seg[0:h_orig, 0:w_orig]
                        elif pred_seg.ndim == 3:
                            pred_seg = pred_seg[:, 0:h_orig, 0:w_orig]
                        if pred_probs is not None:
                            pred_probs = pred_probs[:, :, 0:h_orig, 0:w_orig]

                    if mode == 'volume':
                        volume_slices_reconstruction.append(pred_seg)
                        continue 

                    # SAFE SAVE: Cast to uint8
                    io.write_seg(pred_seg.astype(np.uint8), join(nifti_save_dir, f"{case_id}.nii.gz"), current_props[i])
                    
                    viz_data = self.viz_cache[i]
                    if pred_probs is not None:
                        if isinstance(pred_probs, torch.Tensor): pred_probs = pred_probs.cpu().numpy()
                        
                        metrics = self.compute_metrics_and_visualize(
                            pred_probs, gt_for_metrics, viz_data, case_id, current_out_dir, 
                            target_labels=current_labels_for_metrics
                        )
                        
                        if gt_for_metrics is not None:
                            group_slice_metrics.append(metrics)
                            for m_key, m_val in metrics.items():
                                if m_key != 'case_id':
                                    metrics_history[case_id][m_key].append(m_val)
                            
                            pred_hard = np.argmax(pred_probs, axis=0)
                            if current_labels_for_metrics is not None:
                                foreground_labels = current_labels_for_metrics
                            else:
                                foreground_labels = self.predictor.label_manager.foreground_labels
                            
                            for label in foreground_labels:
                                p_mask = (pred_hard == label)
                                g_mask = (gt_for_metrics == label)
                                tp = np.logical_and(p_mask, g_mask).sum()
                                fp = np.logical_and(p_mask, ~g_mask).sum()
                                fn = np.logical_and(~p_mask, g_mask).sum()
                                current_global_counts[label]['TP'] += tp
                                current_global_counts[label]['FP'] += fp
                                current_global_counts[label]['FN'] += fn

                            if 3 in foreground_labels and 4 in foreground_labels:
                                p_mask_comb = (pred_hard == 3) | (pred_hard == 4)
                                g_mask_comb = (gt_for_metrics == 3) | (gt_for_metrics == 4)
                                tp_c = np.logical_and(p_mask_comb, g_mask_comb).sum()
                                fp_c = np.logical_and(p_mask_comb, ~g_mask_comb).sum()
                                fn_c = np.logical_and(~p_mask_comb, g_mask_comb).sum()
                                current_global_counts["Combined_3_4"]['TP'] += tp_c
                                current_global_counts["Combined_3_4"]['FP'] += fp_c
                                current_global_counts["Combined_3_4"]['FN'] += fn_c

                            if 1 in foreground_labels and 2 in foreground_labels and len(foreground_labels) > 2:
                                p_mask_comb = (pred_hard == 1) | (pred_hard == 2)
                                g_mask_comb = (gt_for_metrics == 1) | (gt_for_metrics == 2)
                                tp_c = np.logical_and(p_mask_comb, g_mask_comb).sum()
                                fp_c = np.logical_and(p_mask_comb, ~g_mask_comb).sum()
                                fn_c = np.logical_and(~p_mask_comb, g_mask_comb).sum()
                                current_global_counts["Combined_1_2"]['TP'] += tp_c
                                current_global_counts["Combined_1_2"]['FP'] += fp_c
                                current_global_counts["Combined_1_2"]['FN'] += fn_c

                gc.collect()
                torch.cuda.empty_cache()

            if mode == 'volume' and volume_slices_reconstruction:
                print(f"Reconstructing volume from {len(volume_slices_reconstruction)} slices...")
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
                else:
                    print("Error: Volume properties not found. Cannot save 3D volume.")

            if mode != 'volume' and group_slice_metrics:
                df_group = pd.DataFrame(group_slice_metrics)
                df_group.to_csv(join(current_out_dir, 'metrics_per_case.csv'), index=False)
                group_mean_summary = df_group.mean(numeric_only=True)
                group_mean_summary.to_csv(join(current_out_dir, 'summary_mean_dice.csv'))
                
                global_dice_data = []
                if self.exp_stacking:
                     items_to_report = [1, 2, 3, 4, "Combined_3_4", "Combined_1_2"]
                else:
                    items_to_report = list(current_global_counts.keys())

                for key in items_to_report:
                    if key not in current_global_counts: continue
                    counts = current_global_counts[key]
                    denom = 2 * counts['TP'] + counts['FP'] + counts['FN']
                    dice_glob = (2 * counts['TP']) / denom if denom > 0 else np.nan
                    
                    if not self.ensemble:
                        global_dice_history[key].append(dice_glob)

                    global_dice_data.append({
                        'Label': key,
                        'Global_Dice': dice_glob,
                        'TP_Total': counts['TP'],
                        'FP_Total': counts['FP'],
                        'FN_Total': counts['FN']
                    })
                
                df_global = pd.DataFrame(global_dice_data)
                df_global.to_csv(join(current_out_dir, 'summary_global_dice.csv'), index=False)
                print(f"Group {folder_name} Global Dice Results:")
                print(df_global[['Label', 'Global_Dice']])
                
                group_mean_dict = group_mean_summary.to_dict()
                group_mean_dict['group'] = folder_name
                fold_aggregates_history.append(group_mean_dict)

        if mode != 'volume' and not self.ensemble and fold_aggregates_history:
            print("\n--- Computing Final Cross-Fold Statistics ---")
            
            df_history = pd.DataFrame(fold_aggregates_history)
            global_mean = df_history.mean(numeric_only=True)
            global_std = df_history.std(numeric_only=True)
            summary_df = pd.DataFrame({'Mean': global_mean, 'Std (Fold)': global_std})
            
            out_path = join(base_output_dir, 'final_experiment_summary.csv')
            summary_df.to_csv(out_path)
            
            if global_dice_history:
                final_global_stats = []
                for label, scores in global_dice_history.items():
                    valid_scores = [s for s in scores if not np.isnan(s)]
                    mean_glob = np.mean(valid_scores) if valid_scores else np.nan
                    std_glob = np.std(valid_scores, ddof=1) if len(valid_scores) > 1 else 0.0
                    
                    final_global_stats.append({
                        'Class_Label': label,
                        'Global_Dice_Mean': mean_glob,
                        'Global_Dice_Std': std_glob,
                        'Num_Folds': len(valid_scores)
                    })
                out_path_glob = join(base_output_dir, 'final_experiment_global_dice.csv')
                pd.DataFrame(final_global_stats).to_csv(out_path_glob, index=False)

        print("Done.")

    def _load_data(self, mode, input_path, io):
        """Returns Dictionary: {case_id: {'image': ..., 'props': ..., 'gt': ..., 'gt_secondary': ...}}"""
        data_pool = {}
        
        # Determine path for Secondary GT if stacking
        secondary_gt_root = None
        if self.exp_stacking and self.model_folder_2:
            raw_path_2 = _get_raw_data_folder_from_model_folder(self.model_folder_2)
            if raw_path_2:
                secondary_gt_root = join(raw_path_2, 'labelsTr')
                if not os.path.isdir(secondary_gt_root):
                    secondary_gt_root = join(raw_path_2, 'labelsTs')                    
        
        if mode == 'validation':
            input_path = self.model_folder.replace('nnunet_results', 'nnunet_raw').replace('nnUNetTrainerWandb__nnUNetPlans__2d', '')
            if 'nnUNetTrainer' in input_path: input_path = input_path.split('nnUNetTrainer')[0]
            
            # Load splits to know what keys to look for
            split_file = join(input_path, 'splits_final.json')
            if not isfile(split_file): split_file = join(self.model_folder, 'splits_final.json')
            
            if isfile(split_file):
                self.splits = load_json(split_file)
                val_keys = set()
                for f in self.folds:
                    if f < len(self.splits):
                        val_keys.update(self.splits[f]['val'])
                val_keys = sorted(list(val_keys))
                
                print(f"Loading {len(val_keys)} cases (union of validation splits)...")
                for k in val_keys:
                    img_path = join(input_path, 'imagesTr', f"{k}_0000.nii.gz")
                    phase_path = join(input_path, 'imagesTr', f"{k}_0001.nii.gz")
                    gt_path = join(input_path, 'labelsTr', f"{k}.nii.gz")
                    
                    # Logic Change for mag-one-channel
                    if self.exp_mag_one_channel:
                         if not isfile(img_path):
                             print(f"Skipping {k}, missing magnitude file.")
                             continue
                         img, prop = io.read_images([img_path])
                    else:
                        if not (isfile(img_path) and isfile(phase_path)):
                            print(f"Skipping {k}, missing files.")
                            continue
                        img, prop = io.read_images([img_path, phase_path])

                    gt, _ = io.read_images([gt_path])
                    
                    entry = {'image': img, 'props': prop, 'gt': gt[0]}
                    if secondary_gt_root:
                        sec_gt_path = join(secondary_gt_root, f"{k}.nii.gz".replace("wm_gm_segmentation", "merge_lesions"))
                        if isfile(sec_gt_path):
                             gt2, _ = io.read_images([sec_gt_path])
                             entry['gt_secondary'] = gt2[0]
                        else : 
                            print(f"Couldn't find {sec_gt_path}")
                    
                    data_pool[k] = entry
            else:
                 print("Critical: splits_final.json missing for validation mode.")

        elif mode == 'test':
            if not input_path: raise ValueError("Provide test directory.")
            files = sorted([f for f in os.listdir(input_path) if f.endswith('.nii.gz') and "_0000" in f])
            print(f"Found {len(files)} test cases...")
            for f in files:
                case_id = f.replace('_0000.nii.gz', '')
                c0 = join(input_path, f); c1 = join(input_path, f.replace('_0000', '_0001'))
                
                # Logic Change for mag-one-channel
                if self.exp_mag_one_channel:
                    channel_files = [c0]
                else:
                    channel_files = [c0, c1] if isfile(c1) else [c0]
                
                img, prop = io.read_images(channel_files)
                gt = None
                
                # Try finding GT in labelsTs or labelsTr
                gt_path_ts = join(input_path, '../labelsTs', f"{case_id}.nii.gz")
                gt_path_tr = join(input_path, '../labelsTr', f"{case_id}.nii.gz")
                
                gt_path = gt_path_ts if isfile(gt_path_ts) else (gt_path_tr if isfile(gt_path_tr) else None)

                if gt_path:
                     gt_arr, _ = io.read_images([gt_path])
                     gt = gt_arr[0]
                
                entry = {'image': img, 'props': prop, 'gt': gt}
                
                if secondary_gt_root and gt is not None:
                     sec_gt_path = join(secondary_gt_root, f"{case_id}.nii.gz")
                     if not isfile(sec_gt_path):
                         sec_gt_path = secondary_gt_root.replace('labelsTr', 'labelsTs') + f"/{case_id}.nii.gz".replace("wm_gm_segmentation", "merge_lesions")
                     if isfile(sec_gt_path):
                         gt2, _ = io.read_images([sec_gt_path])
                         entry['gt_secondary'] = gt2[0]

                data_pool[case_id] = entry

        elif mode == 'volume':
            if not input_path: raise ValueError("Provide volume path.")
            
            mag_path = input_path
            
            if os.path.isdir(mag_path):
                candidates = [f for f in os.listdir(mag_path) if "mag" in f and f.endswith(".nii.gz")]
                if not candidates:
                    raise ValueError(f"No file containing 'mag' and ending in .nii.gz found in folder: {mag_path}")
                mag_path = join(mag_path, candidates[0])

            if "mag" not in os.path.basename(mag_path):
                raise ValueError(f"Volume mode expects input path to contain 'mag' to infer phase. Got: {mag_path}")
            
            phase_path = mag_path.replace("mag", "phase")
            
            # Logic Change for mag-one-channel
            if self.exp_mag_one_channel:
                print(f"Loading Volume (Magnitude only): {mag_path}")
                vol, vol_props = io.read_images([mag_path])
            else:
                if not isfile(phase_path):
                    raise ValueError(f"Inferred phase file not found: {phase_path}")
                print(f"Loading Volume Pair:\n  Magnitude: {mag_path}\n  Phase:     {phase_path}")
                vol, vol_props = io.read_images([mag_path, phase_path])
            
            self.volume_props = vol_props.copy()
            self.volume_props['spacing'] = list(self.volume_props['spacing']) 
            
            c, d, h, w = vol.shape
            print(f"Slicing volume with depth {d} into individual slices...")
            
            exact_filename = os.path.basename(mag_path)
            
            for i in range(d):
                slice_img = vol[:, i:i+1, :, :]
                base_name = exact_filename.replace(".nii.gz", "")
                case_id = f"{base_name}_slice{i:03d}"

                slice_props = vol_props.copy()
                current_shape = slice_img.shape[1:] 
                slice_props['shape_after_cropping_and_before_resampling'] = current_shape
                slice_props['shape_before_cropping'] = current_shape
                slice_props['spacing'] = list(slice_props['spacing'])
                
                data_pool[case_id] = {'image': slice_img, 'props': slice_props, 'gt': None, 'orig_filename': exact_filename}

        return data_pool

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-experiment', type=str, required=True, help="Experiment name")
    parser.add_argument('-model_folder', type=str, required=True, help="Path to nnUNet model folder")
    parser.add_argument('-model_folder_2', type=str, required=False, help="Path to second nnUNet model folder for stacking")
    parser.add_argument('-mode', type=str, choices=['validation', 'test', 'volume'], required=True)
    parser.add_argument('-path', type=str, required=True, help="Path to input data (Mag path for volumes)")
    parser.add_argument('-folds', nargs='+', type=int, default=[0], help="Folds to use")
    parser.add_argument('--use_tta', action='store_true', help="Enable Test Time Augmentation")
    parser.add_argument('-output_root', type=str, required=False, help="Root folder for outputs")
    parser.add_argument('-rel_path', type=str, default="", help="Relative path for output structure")
    parser.add_argument('--ensemble', action='store_true', help="Run inference using ensemble of all specified folds")
    
    args = parser.parse_args()
    
    runner = CustomInferenceRunner(
        args.model_folder, 
        args.experiment, 
        tuple(args.folds), 
        device='cuda', 
        use_tta=args.use_tta, 
        ensemble=args.ensemble,
        model_folder_2=args.model_folder_2
    )
    runner.run(args.mode, args.path, output_root=args.output_root, rel_path=args.rel_path)