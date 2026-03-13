import os
import pandas as pd
import glob
import numpy as np

def generate_latex_table(df, output_filename):
    """
    Generates a LaTeX table from the aggregated DataFrame.
    Calculates averages for Total columns and formats with Mean ± Std.
    Bolds the best results (Highest Dice, Lowest HD95).
    """
    # 1. Calculate Averages (Total)
    # Mapping assumed: Class 1=WM, 2=GM, 3=WML, 4=GML
    classes = [1, 2, 3, 4]
    
    # Calculate Total Mean/Std for Dice
    # We take the mean across the 4 classes for each row
    dice_cols = [f'dice_class_{c}' for c in classes]
    std_dice_cols = [f'std_dice_class_{c}' for c in classes]
    
    # Note: For strict correctness, propagating stds would be sqrt(sum(sigma^2))/N or similar,
    # but often simple averaging of stds is displayed or mean of stds. 
    # Here we simply compute the mean of the values in the columns for the "Total" column.
    df['dice_total'] = df[dice_cols].mean(axis=1)
    df['std_dice_total'] = df[std_dice_cols].mean(axis=1)

    # Calculate Total Mean/Std for HD95
    hd95_cols = [f'hd95_class_{c}' for c in classes]
    std_hd95_cols = [f'std_hd95_class_{c}' for c in classes]
    
    df['hd95_total'] = df[hd95_cols].mean(axis=1)
    df['std_hd95_total'] = df[std_hd95_cols].mean(axis=1)

    # 2. Determine Best Values for Bolding
    # Structure: (column_name, 'max' or 'min')
    metrics_config = [
        ('dice_class_1', 'max'), ('hd95_class_1', 'min'), # WM
        ('dice_class_2', 'max'), ('hd95_class_2', 'min'), # GM
        ('dice_class_3', 'max'), ('hd95_class_3', 'min'), # WML
        ('dice_class_4', 'max'), ('hd95_class_4', 'min'), # GML
        ('dice_total', 'max'),   ('hd95_total', 'min'),   # Total
    ]

    best_values = {}
    for col, mode in metrics_config:
        if col in df.columns:
            # We must ignore NaNs
            valid_vals = df[col].dropna()
            if not valid_vals.empty:
                if mode == 'max':
                    best_values[col] = valid_vals.max()
                else:
                    best_values[col] = valid_vals.min()

    # 3. Generate LaTeX
    latex_lines = []
    
    # Header matches the user's example style
    latex_lines.append(r"\begin{table*}[t]")
    latex_lines.append(r"    \centering")
    latex_lines.append(r"    \caption{Comparison of Dice and HD95 metrics (3D Results). Best results are highlighted in bold.}")
    latex_lines.append(r"    \label{tab:results_3d}")
    latex_lines.append(r"    \resizebox{\textwidth}{!}{%")
    latex_lines.append(r"        \begin{tabular}{lcccccccccc}")
    latex_lines.append(r"            \toprule")
    latex_lines.append(r"            & \multicolumn{2}{c}{\textbf{WM}} & \multicolumn{2}{c}{\textbf{GM}} & \multicolumn{2}{c}{\textbf{WM Lesions}} & \multicolumn{2}{c}{\textbf{GM Lesions}} & \multicolumn{2}{c}{\textbf{Total Average}} \\")
    latex_lines.append(r"            \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}")
    # Using HG95 in header to match user example, though variable is HD95
    latex_lines.append(r"            \textbf{Config} & \textbf{Dice} & \textbf{HG95} & \textbf{Dice} & \textbf{HG95} & \textbf{Dice} & \textbf{HG95} & \textbf{Dice} & \textbf{HG95} & \textbf{Dice} & \textbf{HG95} \\")
    latex_lines.append(r"            \midrule")

    def format_val(row, mean_col, std_col, decimals):
        mean = row.get(mean_col, np.nan)
        std = row.get(std_col, np.nan)
        
        # Check for NaN
        if pd.isna(mean) or str(mean).strip() == "":
            return "--"
        
        try:
            mean_float = float(mean)
            mean_str = f"{mean_float:.{decimals}f}"
            
            std_str = ""
            if not pd.isna(std) and str(std).strip() != "":
                std_float = float(std)
                std_str = f"{std_float:.{decimals}f}"

            # Construct display string
            if std_str:
                cell_str = f"{mean_str} \\pm {std_str}"
            else:
                cell_str = f"{mean_str}"
                
            # Check for Bolding
            is_best = False
            # Find the mode for this column
            target_mode = None
            for c, m in metrics_config:
                if c == mean_col:
                    target_mode = m
                    break
            
            if target_mode and mean_col in best_values:
                best = best_values[mean_col]
                # Compare with tolerance
                if abs(mean_float - best) < 1e-9:
                    is_best = True
            
            if is_best:
                return f"$\\mathbf{{{cell_str}}}$"
            else:
                return f"${cell_str}$"
        except ValueError:
            return str(mean)

    for idx, row in df.iterrows():
        # Escape underscores in experiment name for latex
        name = str(row.get('experiment_name', '')).replace('_', r'\_')
        
        # Format columns: Dice (3 decimals), HD95 (2 decimals) as per user example
        
        # WM (Class 1)
        wm_dice = format_val(row, 'dice_class_1', 'std_dice_class_1', 3)
        wm_hd95 = format_val(row, 'hd95_class_1', 'std_hd95_class_1', 2)
        
        # GM (Class 2)
        gm_dice = format_val(row, 'dice_class_2', 'std_dice_class_2', 3)
        gm_hd95 = format_val(row, 'hd95_class_2', 'std_hd95_class_2', 2)
        
        # WML (Class 3)
        wml_dice = format_val(row, 'dice_class_3', 'std_dice_class_3', 3)
        wml_hd95 = format_val(row, 'hd95_class_3', 'std_hd95_class_3', 2)
        
        # GML (Class 4)
        gml_dice = format_val(row, 'dice_class_4', 'std_dice_class_4', 3)
        gml_hd95 = format_val(row, 'hd95_class_4', 'std_hd95_class_4', 2)
        
        # Total
        tot_dice = format_val(row, 'dice_total', 'std_dice_total', 3)
        tot_hd95 = format_val(row, 'hd95_total', 'std_hd95_total', 2)
        
        row_line = f"            {name} & {wm_dice} & {wm_hd95} & {gm_dice} & {gm_hd95} & {wml_dice} & {wml_hd95} & {gml_dice} & {gml_hd95} & {tot_dice} & {tot_hd95} \\\\"
        latex_lines.append(row_line)

    latex_lines.append(r"            \bottomrule")
    latex_lines.append(r"        \end{tabular}%")
    latex_lines.append(r"    }")
    latex_lines.append(r"\end{table*}")

    with open(output_filename, 'w') as f:
        f.write('\n'.join(latex_lines))
    print(f"\nLaTeX table saved to: {os.path.abspath(output_filename)}")

def aggregate_nnunet_results(base_path, experiments_config, output_filename="aggregated_results.csv", mode="3d"):
    """
    Aggregates metrics from multiple nnU-Net experiments into a single CSV.
    Fetches Dice metrics from the Global summary file and HD95 metrics from the Casewise summary file.
    """
    
    # --- 1. Define Metric Sources ---
    # Map metrics to their specific source CSVs and column names
    
    # For summary_cross_fold_global_2d.csv
    dice_map = {
        'dice_mean': 'dice',
        'dice_std':  'std_dice'
    }
    
    # For summary_cross_fold_casewise_2d.csv
    hd95_map = {
        'hd95_mean': 'hd95',
        'hd95_std':  'std_hd95'
    }
    
    target_classes = [1, 2, 3, 4]
    
    aggregated_rows = []

    print(f"Starting aggregation for mode {mode} with {len(experiments_config)} experiments...\n")

    for exp in experiments_config:
        if len(exp) == 2:
            exp_id, exp_name = exp
            set_type = "validation" 
        elif len(exp) == 3:
            exp_id, exp_name, inference_style = exp 
            if 'test' in inference_style.lower():
                set_type = "test"
            else:
                set_type = "validation"

        row_data = {'experiment_name': exp_name}
        
        # Construct the Base Directory Path for this experiment
        dataset_dir = f"Dataset{exp_id}_MagPhase_{exp_name}" if mode == "3d" else f"Dataset{exp_id}_MagPhaseExp_simple_training_{exp_name}"
        trainer_dir = "nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"
        inference_dir = "inference_results"
        validation_dir = f"{exp_name}_{set_type}_EVAL" if mode == "3d" else f"{exp_name}_FULL_EVAL"
        
        base_exp_path = os.path.join(
            base_path, 
            dataset_dir, 
            trainer_dir if mode == "3d" else "nnUNetTrainerWandb__nnUNetPlans__2d",
            inference_dir, 
            validation_dir
        )
        
        # Define the two file paths
        path_dice = os.path.join(base_exp_path, "summary_cross_fold_global_2d.csv")
        path_hd95 = os.path.join(base_exp_path, "summary_cross_fold_casewise_2d.csv")

        # Helper function to process a single file and update row_data
        def process_metrics_file(csv_path, metric_map, file_label):
            if not os.path.exists(csv_path):
                print(f"[WARNING] {file_label} file not found: {csv_path}")
                return

            try:
                df = pd.read_csv(csv_path)
                # lowercase the columns keys to ensure consistency
                
                df.columns = [col.lower() for col in df.columns]
                
                # Ensure Label column is integer for matching and remove "class_" from it
                if 'label' in df.columns:
                    df['label'] = df['label'].astype(str).str.replace('class_', '', regex=False)
                    df['label'] = pd.to_numeric(df['label'], errors='coerce')
                
                for cls in target_classes:
                    # Filter for the specific class
                    class_row = df[df['label'] == cls]

                    if class_row.empty:
                        # Optional: Print verbose only if needed to avoid clutter
                        print(f"  [INFO] Class {cls} missing in {file_label} for {exp_name}")
                        continue
                    
                    # Extract specific metrics defined in the map
                    for src_col, prefix in metric_map.items():
                        new_col_name = f"{prefix}_class_{cls}"
                        
                        if src_col in class_row.columns:
                            # Get the value (iloc[0] because we expect 1 row per class)
                            val = class_row.iloc[0][src_col]
                            row_data[new_col_name] = val
                        else:
                            print(f"  [WARN] Column '{src_col}' missing in {file_label} for {exp_name}")
                            
            except Exception as e:
                print(f"[ERROR] Failed processing {file_label} for {exp_name}: {e}")

        # --- Process Dice (Global File) ---
        process_metrics_file(path_dice, dice_map, "Global/Dice")

        # --- Process HD95 (Casewise File) ---
        process_metrics_file(path_hd95, hd95_map, "Casewise/HD95")

        aggregated_rows.append(row_data)
        print(f"[SUCCESS] Processed: {exp_id} - {exp_name}")

    # 2. Create DataFrame from aggregated rows
    result_df = pd.DataFrame(aggregated_rows)

    # 3. Reorder Columns
    final_columns = ['experiment_name']
    for cls in target_classes:
        # The order requested: dice, std_dice, hd95, std_hd95
        final_columns.append(f"dice_class_{cls}")
        final_columns.append(f"std_dice_class_{cls}")
        final_columns.append(f"hd95_class_{cls}")
        final_columns.append(f"std_hd95_class_{cls}")

    # Reindex to enforce column order (adds NaNs if columns are missing)
    result_df = result_df.reindex(columns=final_columns)

    # 4. Save to CSV
    save_dir = "tables_3d" if mode == "3d" else "tables_2d"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, output_filename)
    result_df.to_csv(output_path, index=False)
    print(f"\nAggregation complete. Saved to: {os.path.abspath(output_path)}")
    
    # 5. Generate LaTeX Table
    # Assumes output_filename ends in .csv
    latex_filename = os.path.join(save_dir, output_filename.rsplit('.', 1)[0] + ".tex")
    generate_latex_table(result_df, latex_filename)


# ["1708"]="patchsize_"

# ["1709"]="patchsize_192,64,168"

# ["1710"]="patchsize_192,96,160"
# ["1711"]="patchsize_4"

# ["1712"]="patchsize_5"

# ["1713"]="patchsize_5_spatial_aug_2"
# ["1714"]="patchsize_5_spatial_aug_1"


# # ["1715"]="patchsize_5_otsu_masking"
# ["1716"]="patchsize_5_soft_loss_fixed"
# ["1717"]="patchsize_5_soft_loss_2_fixed"


# # ["1718"]="patchsize_5_adamw"



# # ["1719"]="patchsize_5_adamw_otsu"
# ["1720"]="patchsize_5_adamw_phase_prepro"
# ["1721"]="patchsize_5_adamw_mag_prepro"
# ["1722"]="patch_5_adamw_soft_loss_3"


# # ["1723"]="patch_5_adamw_spatial_aug_3"

# ["1724"]="patch_5_adamw_mag_one_channel"

# # ["1725"]="patch_5_adamw_aug_no_rot_z"

# # ["1726"]="patch_5_adamw_aug_5"

# # ["1727"]="patch_5_adamw_aug_6"
# # ["1728"]="patch_5_adamw_aug_7"
# # ["1729"]="patch_5_adamw_aug_8"

# ["1730"]="patch_5_adamw_aug_9"
# ["1731"]="patch_5_adamw_aug_10"
# ["1732"]="patch_5_adamw_aug_11"

# # ["1733"]="patch_5_adamw_aug_12"

# # ["1734"]="patch_5_adamw_aug_13"
# # ["1735"]="patch_5_adamw_aug_14"
# # ["1736"]="patch_5_adamw_aug_15"

# # ["1737"]="patch_5_adamw_aug_16"
# # ["1738"]="patch_5_adamw_aug_17"
# # ["1739"]="patch_5_adamw_aug_18"
# # ["1740"]="patch_5_adamw_aug_19"

# # ["1741"]="patch_5_adamw_aug_20"
# # ["1742"]="patch_5_adamw_aug_21"
# # ["1743"]="patch_5_adamw_aug_22"
# # ["1744"]="patch_5_adamw_aug_23"

if __name__ == "__main__":
    
    MODE = "3d"
    
    if MODE == "3d":

        BASE_PATH = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/nnunet_dataset/nnUNet_results" 

        # Define your experiments here.
        EXPERIMENTS = [
            # base
            ("1718", "patchsize_5_adamw"),
            
            ## Patch sizes patchsize.csv
            # ("1708", "patchsize_"),
            # ("1709", "patchsize_192,64,168"),
            # ("1710", "patchsize_192,96,160"),
            # ("1711", "patchsize_4"),
            # ("1712", "patchsize_5"),
            
            # Soft Segmentation soft_seg.csv
            # ("1718", "patchsize_5_adamw"),
            # ("1716", "patchsize_5_soft_loss_fixed"),
            # ("1717", "patchsize_5_soft_loss_2_fixed"),
            # ("1722", "patch_5_adamw_soft_loss_3"),
            
            # Preprocessing prepro.csv
            # ("1718", "patchsize_5_adamw"),
            # ("1720", "patchsize_5_adamw_phase_prepro"),
            # ("1721", "patchsize_5_adamw_mag_prepro"),
            
            # Phase/Mag #mag_phase.csv
            # ("1718", "patchsize_5_adamw"), #Mag + Phase
            # ("1724", "patch_5_adamw_mag_one_channel"),
            
            # Spatial Augmentation augmentation.csv
            # ("1718", "patchsize_5_adamw"),
            # ("1730", "patch_5_adamw_aug_9"),
            # ("1731", "patch_5_adamw_aug_10"),
            # ("1732", "patch_5_adamw_aug_11"),
            
            # Winning Combination winning_combination.csv
            ("1718", "patchsize_5_adamw"),
            ("1718", "patchsize_5_adamw", "test_TTA_Ensemble"),
        ]

        aggregate_nnunet_results(BASE_PATH, EXPERIMENTS, output_filename="augmentation.csv", mode="3d")
        
    elif MODE == "2d":
        
        BASE_PATH = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_results" 

        
        EXPERIMENTS = [
            # Spatial Augmentation augmentation.csv
            # ("706", "no_spatial_aug"),
            # ("700", "base"),
            # ("714", "spatial_aug_2"),
            # ("715", "spatial_aug_3"),
            
            # Preprocessing prepro.csv
            # ("700", "base"),
            # ("704", "phase_prepro"),
            # ("705", "mag_prepro"),
            
            # Phase/Mag #mag_phase.csv
            # ("700", "base"),
            # ("710", "mag-one-channel"),
            
            # Soft Segmentation soft_seg.csv
            # ("700", "base"),
            # ("711", "soft_loss_fixed"),
            # ("712", "soft_loss_2_fixed"),
            # ("713", "soft_loss_3_fixed"),
        
            # Winning Combination winning_combination.csv
            # ("717", "winning_combination"),
            # ("717", "winning_combination_tta"),
        ]

        aggregate_nnunet_results(BASE_PATH, EXPERIMENTS, output_filename="winning_combination.csv", mode="2d")