import os
import pandas as pd
import glob
import numpy as np

def get_experiment_metrics(base_path, ds_id, metric_config):
    """
    Helper function to retrieve mean/std metrics for a specific dataset ID.
    Handles 'wide' (standard smoothness) and 'long' (new roughness) formats.
    """
    # 1. Find the dataset directory
    search_pattern = os.path.join(base_path, f"Dataset{ds_id}_*")
    candidates = glob.glob(search_pattern)
    
    if not candidates:
        print(f"[WARNING] No dataset folder found for ID {ds_id} in {base_path}")
        return None, None
    
    dataset_dir = candidates[0]
    folder_name = os.path.basename(dataset_dir)
    parts = folder_name.split('_', 1)
    exp_name_short = parts[1] if len(parts) > 1 else folder_name

    # 2. Find the validation folder
    trainer_pattern = os.path.join(dataset_dir, "nnUnet3DCustom*Trainer*")
    trainer_dirs = glob.glob(trainer_pattern)
    if not trainer_dirs: return None, exp_name_short
    
    inference_dir = os.path.join(trainer_dirs[0], "inference_results")
    eval_pattern = os.path.join(inference_dir, "*_validation_EVAL")
    eval_dirs = glob.glob(eval_pattern)
    if not eval_dirs: return None, exp_name_short
    
    eval_dir = eval_dirs[0]
    
    # 3. Determine File and Format
    csv_filename = metric_config.get("filename", "summary_cross_fold_smoothness.csv")
    csv_path = os.path.join(eval_dir, csv_filename)
    
    if not os.path.exists(csv_path):
        return None, exp_name_short

    # 4. Process CSV
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        results = {}
        target_classes = ['1', '2', '3', '4']

        # --- LONG FORMAT (New Roughness Metric) ---
        if metric_config.get("format") == "long":
            # Expects: Metric, CrossFold_Mean, CrossFold_Std
            mean_col = metric_config["mean_col"]
            std_col = metric_config["std_col"]
            lookup_fmt = metric_config["lookup_pattern"] 
            
            # Classes 1-4
            for cls in target_classes:
                lookup_key = lookup_fmt.format(cls)

                row = df[df['Label'] == lookup_key]
                if not row.empty:
                    results[f"class_{cls}"] = {
                        'mean': row.iloc[0].get(mean_col, np.nan),
                        'std': row.iloc[0].get(std_col, np.nan)
                    }
                else:
                    results[f"class_{cls}"] = {'mean': np.nan, 'std': np.nan}
            
            # Average (Read 'Volume_Roughness' directly)
            avg_row = df[df['Label'] == 'Volume_Roughness']
            if not avg_row.empty:
                results["average"] = {
                    'mean': avg_row.iloc[0].get(mean_col, np.nan),
                    'std': avg_row.iloc[0].get(std_col, np.nan)
                }
            else:
                # Fallback: compute manual mean if missing
                means = [results[f"class_{c}"]['mean'] for c in target_classes if not np.isnan(results[f"class_{c}"]['mean'])]
                if means:
                    results["average"] = {'mean': np.mean(means), 'std': np.std(means)}
                else:
                    results["average"] = {'mean': np.nan, 'std': np.nan}

        # --- WIDE FORMAT (Old Metrics) ---
        else:
            if 'Source' not in df.columns or 'Class' not in df.columns: return None, exp_name_short
            pred_df = df[df['Source'] == 'Prediction'].copy()
            pred_df['Class'] = pred_df['Class'].astype(str)
            col_mean, col_std = metric_config["mean_col"], metric_config["std_col"]

            for cls in target_classes:
                class_row = pred_df[pred_df['Class'] == cls]
                if not class_row.empty:
                    results[f"class_{cls}"] = {
                        'mean': class_row.iloc[0].get(col_mean, np.nan),
                        'std': class_row.iloc[0].get(col_std, np.nan)
                    }
            
            avg_row = pred_df[pred_df['Class'] == 'Average']
            if not avg_row.empty:
                results["average"] = {
                    'mean': avg_row.iloc[0].get(col_mean, np.nan),
                    'std': avg_row.iloc[0].get(col_std, np.nan)
                }

        return results, exp_name_short

    except Exception as e:
        print(f"[ERROR] parsing CSV for {ds_id}: {e}")
        return None, exp_name_short


def aggregate_smoothness_results(base_path, dataset_ids, output_filename="smoothness_results.csv", metric="interslice_dice", ref_ids=None):
    
    # --- CONFIGURATION ---
    is_relative_metric = False
    
    if metric == "z_roughness":
        config = {
            "filename": "summary_cross_fold_roughness.csv",
            "format": "long",
            "mean_col": "CrossFold_Mean",
            "std_col": "CrossFold_Std",
            "lookup_pattern": "Class_{}_Roughness"
        }
        out_prefix = "Z_Roughness"
        subfolder = "Z_Roughness"
        is_relative_metric = True 
        
    elif metric == "dice":
        config = {
            "filename": "summary_cross_fold_global_2d.csv",
            "format": "long",
            "mean_col": "Dice_Mean",
            "std_col": "Dice_Std",
            "lookup_pattern": "{}"
        }
        out_prefix = "Dice"
        subfolder = "Dice"
        is_relative_metric = False 
        
    elif metric == "interslice_dice":
        config = {"filename": "summary_cross_fold_smoothness.csv", "format": "wide", "mean_col": "InterSlice_Dice_mean", "std_col": "InterSlice_Dice_std"}
        out_prefix = "inter_slice_dice"
        subfolder = "interslice_dice"
        
    elif metric == "sphericity":
        config = {"filename": "summary_cross_fold_smoothness.csv", "format": "wide", "mean_col": "Sphericity_mean", "std_col": "Sphericity_std"}
        out_prefix = "sphericity"
        subfolder = "sphericity"
        
    elif metric == "tv_z":
        config = {"filename": "summary_cross_fold_smoothness.csv", "format": "wide", "mean_col": "TotalVariation_Z_mean", "std_col": "TotalVariation_Z_std"}
        out_prefix = "TV_Z"
        subfolder = "TV_Z"
        is_relative_metric = True
        
    elif metric == "tv_y":
        config = {"filename": "summary_cross_fold_smoothness.csv", "format": "wide", "mean_col": "TotalVariation_Y_mean", "std_col": "TotalVariation_Y_std"}
        out_prefix = "TV_Y"
        subfolder = "TV_Y"
        is_relative_metric = True
        
    elif metric == "tv_x":
        config = {"filename": "summary_cross_fold_smoothness.csv", "format": "wide", "mean_col": "TotalVariation_X_mean", "std_col": "TotalVariation_X_std"}
        out_prefix = "TV_X"
        subfolder = "TV_X"
        is_relative_metric = True
    else:
        print(f"[ERROR] Unknown metric: {metric}")
        return

    # Fetch References
    ref_data = {}
    if is_relative_metric and ref_ids:
        print(f"   > Fetching reference data for relative {out_prefix}...")
        pseudo_metrics, _ = get_experiment_metrics(base_path, ref_ids['pseudo'], config)
        if pseudo_metrics: ref_data['pseudo'] = pseudo_metrics
        pred_metrics, _ = get_experiment_metrics(base_path, ref_ids['pred'], config)
        if pred_metrics: ref_data['pred'] = pred_metrics

    ordered_suffixes = ['class_1', 'class_2', 'class_3', 'class_4', 'average']
    aggregated_rows = []
    
    print(f"Starting {metric} aggregation...")

    for ds_id in dataset_ids:
        metrics, exp_name = get_experiment_metrics(base_path, ds_id, config)
        if not metrics:
            print(f"[WARNING] Skipping {ds_id}")
            continue

        row_data = {'dataset_id': ds_id, 'experiment_name': exp_name}
        
        # Absolute
        for suffix, values in metrics.items():
            row_data[f"{out_prefix}_mean_{suffix}"] = values['mean']
            row_data[f"{out_prefix}_std_{suffix}"] = values['std']

        # Relative
        if is_relative_metric:
            reference_source = None
            if str(ds_id).startswith("2"): reference_source = ref_data.get('pred')
            elif str(ds_id).startswith("1"): reference_source = ref_data.get('pseudo')
            
            if reference_source:
                for suffix in ordered_suffixes:
                    curr_mean = row_data.get(f"{out_prefix}_mean_{suffix}")
                    ref_vals = reference_source.get(suffix)
                    if curr_mean is not None and ref_vals is not None:
                        row_data[f"Relative_{out_prefix}_mean_{suffix}"] = curr_mean - ref_vals['mean']

        aggregated_rows.append(row_data)

    if not aggregated_rows: return

    df_all = pd.DataFrame(aggregated_rows)

    # Save Absolute
    abs_cols = ['dataset_id', 'experiment_name'] + [f"{out_prefix}_{s}_{suffix}" for suffix in ordered_suffixes for s in ['mean', 'std']]
    df_abs = df_all.reindex(columns=abs_cols)
    out_dir = os.path.join("noise_tables", subfolder)
    os.makedirs(out_dir, exist_ok=True)
    df_abs.to_csv(os.path.join(out_dir, output_filename), index=False)
    print(f"Saved Absolute: {output_filename}")

    # Save Relative
    if is_relative_metric:
        rel_cols = ['dataset_id', 'experiment_name'] + [f"Relative_{out_prefix}_mean_{suffix}" for suffix in ordered_suffixes]
        # Only save if data exists
        if any(c in df_all.columns for c in rel_cols[2:]):
            df_rel = df_all.reindex(columns=rel_cols)
            out_rel_dir = os.path.join("noise_tables", f"relative_{subfolder}")
            os.makedirs(out_rel_dir, exist_ok=True)
            df_rel.to_csv(os.path.join(out_rel_dir, output_filename), index=False)
            print(f"Saved Relative: {output_filename}")
            
            
if __name__ == "__main__":
    
    # Path to nnUNet_results where the Dataset folders are located
    BASE_PATH = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/nnunet_dataset/nnUNet_results"
    
    # Define references
    REF_IDS = {
        "pseudo": "1718",
        "pred": "2000"
    }

    # Iterate through modes
    modes = ["pred_in_plane", "pseudo_in_plane", "pred_through_plane", "pseudo_through_plane"]
    
    for mode in modes:
        if mode == "pred_in_plane":
            EXPERIMENTS_IDS = [
                "2001", 
                "2004", 
                "2005", 
                "2006", 
                "2010", 
                "2011"
                ]
            FILENAME = "in_plane_pred_smoothness_results.csv"
        
        elif mode == "pseudo_in_plane":
            EXPERIMENTS_IDS = [
                "1734", 
                "1737", 
                "1738", 
                "1739", 
                "1740", 
                "1744"
                ]
            FILENAME = "in_plane_pseudo_smoothness_results.csv"
            
        elif mode == "pred_through_plane":
            EXPERIMENTS_IDS = [
                "2012", 
                "2002", 
                "2003", 
                "2007", 
                "2008", 
                "2009",
                
                ]
            FILENAME = "through_plane_pred_smoothness_results.csv"
            
        elif mode == "pseudo_through_plane":
            EXPERIMENTS_IDS = [
                "1733", 
                "1735", 
                "1736", 
                "1741", 
                "1742", 
                "1743"
                ]
            FILENAME = "through_plane_pseudo_smoothness_results.csv"

        # Run for metrics
        # Added "z_roughness" to the list
        for metric in ["interslice_dice", "sphericity", "tv_z", "tv_y", "tv_x", "z_roughness", "dice"]:
            aggregate_smoothness_results(
                BASE_PATH, 
                EXPERIMENTS_IDS, 
                output_filename=FILENAME, 
                metric=metric, 
                ref_ids=REF_IDS
            )