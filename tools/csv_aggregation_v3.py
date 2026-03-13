import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# =================USER CONFIGURATION=================
# Base paths for 3D and 2D
# Note: Ensure these paths are correct and accessible
BASE_PATH_3D = Path("/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/nnunet_dataset/nnUNet_results")
BASE_PATH_2D = Path("/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_results")

# Class Mapping
CLASS_MAP = {
    1: "White Matter",
    2: "Gray Matter",
    3: "Lesion WM",
    4: "Lesion GM"
}

# Experiment Groups
EXPERIMENT_GROUPS = [
    # ---------------- 3D Experiments ----------------
    {
        "type": "3d",
        "output_filename": "patchsize.csv",
        "experiments": [
            {"id": "1708", "name": "patchsize_"},
            {"id": "1709", "name": "patchsize_192,64,168"},
            {"id": "1710", "name": "patchsize_192,96,160"},
            {"id": "1711", "name": "patchsize_4"},
            {"id": "1712", "name": "patchsize_5"},
            {"id": "1718", "name": "patchsize_5_adamw"}, # Base for comparison
        ]
    },
    {
        "type": "3d",
        "output_filename": "soft_seg.csv",
        "experiments": [
            {"id": "1718", "name": "patchsize_5_adamw"}, # Base
            {"id": "1716", "name": "patchsize_5_soft_loss_fixed"},
            {"id": "1717", "name": "patchsize_5_soft_loss_2_fixed"},
            {"id": "1722", "name": "patch_5_adamw_soft_loss_3"},
        ]
    },
    {
        "type": "3d",
        "output_filename": "prepro.csv",
        "experiments": [
            {"id": "1718", "name": "patchsize_5_adamw"}, # Base
            {"id": "1720", "name": "patchsize_5_adamw_phase_prepro"},
            {"id": "1721", "name": "patchsize_5_adamw_mag_prepro"},
        ]
    },
    {
        "type": "3d",
        "output_filename": "mag_phase.csv",
        "experiments": [
            {"id": "1718", "name": "patchsize_5_adamw"}, # Mag + Phase
            {"id": "1724", "name": "patch_5_adamw_mag_one_channel"},
        ]
    },
    {
        "type": "3d",
        "output_filename": "augmentation.csv",
        "experiments": [
            {"id": "1718", "name": "patchsize_5_adamw"},
            {"id": "1730", "name": "patch_5_adamw_aug_9"},
            {"id": "1731", "name": "patch_5_adamw_aug_10"},
            {"id": "1732", "name": "patch_5_adamw_aug_11"},
        ]
    },
        {
        "type": "3d",
        "output_filename": "ostu_masking.csv",
        "experiments": [
            {"id": "1718", "name": "patchsize_5_adamw"},
            {"id": "1719", "name": "patchsize_5_adamw_otsu"},
        ]
    },
    {
        "type": "3d",
        "output_filename": "optimizer.csv",
        "experiments": [
            {"id": "1712", "name": "patchsize_5"},
            {"id": "1718", "name": "patchsize_5_adamw"},

        ]
    },
    {
        "type": "3d",
        "output_filename": "winning_combination.csv",
        "experiments": [
            {"id": "1718", "name": "patchsize_5_adamw", "modes": ["single"]},
            {"id": "1718", "name": "patchsize_5_adamw", "modes": ["TTA"]},
        ]
    },
    
    {
        "type": "3d",
        "output_filename": "test_results.csv",
        "experiments": [
            {"id": "1718", "name": "patchsize_5_adamw", "modes": ["TTA"], "dataset_split": "test"},
            {"id": "1718", "name": "patchsize_5_adamw", "modes": ["TTA + Ensemble"], "dataset_split": "test"},
        ]
    },
    
    # ---------------- 2D Experiments ----------------
    {
        "type": "2d",
        "output_filename": "augmentation.csv",
        "experiments": [
            {"id": "700", "name": "base"},
            {"id": "706", "name": "no_spatial_aug"},
            {"id": "714", "name": "spatial_aug_2"},
            {"id": "715", "name": "spatial_aug_3"},
        ]
    },
    {
        "type": "2d",
        "output_filename": "prepro.csv",
        "experiments": [
            {"id": "700", "name": "base"},
            {"id": "704", "name": "phase_prepro"},
            {"id": "705", "name": "mag_prepro"},
        ]
    },
    {
        "type": "2d",
        "output_filename": "mag_phase.csv",
        "experiments": [
            {"id": "700", "name": "base"},
            {"id": "710", "name": "mag-one-channel"},
        ]
    },
    {
        "type": "2d",
        "output_filename": "soft_seg.csv",
        "experiments": [
            {"id": "700", "name": "base"},
            {"id": "711", "name": "soft_loss_fixed"},
            {"id": "712", "name": "soft_loss_2_fixed"},
            {"id": "713", "name": "soft_loss_3_fixed"},
        ]
    },
    {
        "type": "2d",
        "output_filename": "ostu_masking.csv",
        "experiments": [
            {"id": "708", "name": "base_no_otsu"}, 
            {"id": "700", "name": "base"},
        ]
    },
    {
        "type": "2d",
        "output_filename": "optimizer.csv",
        "experiments": [
            {"id": "716", "name": "sgd_optimizer"},
            {"id": "700", "name": "base"},
        ]
    },
    {
        "type": "2d",
        "output_filename": "decoupled.csv",
        "experiments": [
            {"id": "700", "name": "base"},
            {"id": "stacking", "name": "stacking"},
        ]
    },
    {
        "type": "2d",
        "output_filename": "supp_comparison.csv",
        "experiments": [
            {"id": "700", "name": "base"},
            {"id": "717", "name": "winning_combination"}, # soft 2 & aug 2
            {"id": "714", "name": "spatial_aug_2"}, # soft 2 & aug 2
            {"id": "712", "name": "soft_loss_2_fixed"}, # soft 2 and aug 1
        ]
    },
    {
        "type": "2d",
        "output_filename": "winning_combination.csv", #aug 2 soft 2
        "experiments": [
            {"id": "712", "name": "soft_loss_2_fixed", "modes": ["single"]},
            {"id": "712", "name": "soft_loss_2_fixed", "modes": ["TTA"]},
        ]
    },
    {
        "type": "2d",
        "output_filename": "test_results.csv",
        "experiments": [
            {"id": "712", "name": "soft_loss_2_fixed", "modes": ["TTA"], "dataset_split": "test"},
            {"id": "712", "name": "soft_loss_2_fixed", "modes": ["TTA + Ensemble"], "dataset_split": "test"},
        ]
    }
]

# ====================================================
def log_error(msg: str):
    """Prints a loud error message in red."""
    print(f"\033[91m[ERROR] {msg}\033[0m")

def _find_dataset_dir(base_path: Path, exp_id: str, exp_name: str, mode_type: str) -> Path | None:
    prefixes = [f"Dataset{exp_id}_MagPhase_{exp_name}", f"Dataset{exp_id}_{exp_name}"]
    if mode_type == "2d":
        prefixes[0] = f"Dataset{exp_id}_MagPhaseExp_simple_training_{exp_name}"

    for prefix in prefixes:
        target = base_path / prefix
        if target.exists():
            return target
    return None

def _get_target_directories(inference_root: Path, exp_name: str, mode: str, split: str, mode_type: str) -> tuple[list[Path], bool]:
    """
    Returns a tuple of (List of fold/ensemble directories, is_ensemble).
    Strictly follows the hardcoded paths provided.
    """
    dirs = []
    is_ensemble = "ensemble" in mode.lower()

    if mode_type == "3d":
        if split == "validation":
            base = inference_root / f"{exp_name}_validation_EVAL"
            if mode == "single":
                dirs = [d for d in base.glob("fold_*") if d.is_dir() and "TTA" not in d.name and "Ensemble" not in d.name]
            elif mode == "TTA":
                dirs = [d for d in base.glob("fold_*_TTA") if d.is_dir()]
            elif mode == "Ensemble":
                dirs = [base / "ensemble_predictions"]
            elif mode in ["TTA + Ensemble", "TTA+Ensemble"]:
                dirs = [base / "ensemble_predictions_TTA"] 

        elif split == "test":
            base = inference_root / f"{exp_name}_test_EVAL"
            if mode == "single":
                dirs = [d for d in (base / "inference_results_test").glob("fold_*") if d.is_dir()]
            elif mode == "TTA":
                dirs = [d for d in (base / "inference_results_test_TTA").glob("fold_*") if d.is_dir()]
            elif mode == "Ensemble":
                raise ValueError("3D Test Ensemble without TTA requested but should not exist.")
            elif mode in ["TTA + Ensemble", "TTA+Ensemble"]:
                dirs = [base / "inference_results_test_TTA_Ensemble"]
                
    elif mode_type == "2d":
        if split == "validation":
            if mode == "single":
                dirs = [d for d in (inference_root / f"{exp_name}_FULL_EVAL").glob("fold_*") if d.is_dir()]
            elif mode == "TTA":
                dirs = [d for d in (inference_root / f"{exp_name}_FULL_EVAL_TTA").glob("fold_*") if d.is_dir()]
            elif mode == "Ensemble":
                dirs = [inference_root / f"{exp_name}_FULL_EVAL" / "ensemble_predictions"]
            elif mode in ["TTA + Ensemble", "TTA+Ensemble"]:
                dirs = [inference_root / f"{exp_name}_FULL_EVAL_TTA" / "ensemble_predictions"]

        elif split == "test":
            if mode == "single":
                raise ValueError("2D Test Single requested but should not exist (leave blank).")
            elif mode == "TTA":
                dirs = [d for d in (inference_root / f"{exp_name}_TEST_SET_TTA").glob("fold_*") if d.is_dir()]
            elif mode in ["TTA + Ensemble", "TTA+Ensemble"]:
                dirs = [inference_root / f"{exp_name}_TEST_SET_TTA" / "ensemble_predictions"]

    valid_dirs = [d for d in dirs if d.exists()]
    if dirs and not valid_dirs:
        log_error(f"Target directory path formed, but does not exist on disk: {dirs[0]}")
        
    return valid_dirs, is_ensemble


def parse_fold_metrics(fold_dir: Path, mode_type: str) -> dict:
    """Extracts Dice and HD95 exactly from the global and casewise CSV files."""
    fold_metrics = {}
    
    # 1. Parse Dice from Global file
    dice_csv = fold_dir / "metrics_2d_global.csv"
    if not dice_csv.exists():
        log_error(f"Missing Global CSV: {dice_csv}")
    else:
        df_dice = pd.read_csv(dice_csv)
        df_dice.columns = df_dice.columns.str.strip()
        
        label_col = next((c for c in df_dice.columns if c.lower() in ['label', 'class', 'structure']), None)
        if label_col:
            for cls_id, cls_name in CLASS_MAP.items():
                row = df_dice[df_dice[label_col].astype(str).str.contains(rf'^{cls_id}\.?0?$|^class_{cls_id}$', regex=True, case=False)]
                if not row.empty:
                    dice_col = next((c for c in row.columns if c.lower() in ['global_dice','dice_mean', 'dice', 'mean_dice']), None)
                    if dice_col and not pd.isna(row.iloc[0][dice_col]):
                        fold_metrics[f'Dice_{cls_name}'] = float(row.iloc[0][dice_col])

    # 2. Parse HD95 from Casewise file
    hd95_csv = fold_dir / "metrics_2d_casewise.csv"
    if not hd95_csv.exists():
        log_error(f"Missing Casewise CSV: {hd95_csv}")
    else:
        df_hd95 = pd.read_csv(hd95_csv)
        df_hd95.columns = df_hd95.columns.str.strip()
        
        for cls_id, cls_name in CLASS_MAP.items():
            hd95_col = next((c for c in df_hd95.columns if c.lower() in [f'hd95_{cls_id}', f'hd95_slice_class_{cls_id}']), None)
            if hd95_col:
                vals = pd.to_numeric(df_hd95[hd95_col], errors='coerce').dropna()
                if not vals.empty:
                    val_mean = float(vals.mean())
                    # --- CRITICAL FIX: Convert 2D HD95 from voxels to mm ---
                    if mode_type == "2d":
                        val_mean *= 0.075
                    fold_metrics[f'HD95_{cls_name}'] = val_mean

    # 3. Compute mathematically valid total averages for this specific fold
    dice_vals = [v for k, v in fold_metrics.items() if 'Dice_' in k]
    if dice_vals:
        fold_metrics['Dice_Total'] = float(np.mean(dice_vals))
        
    hd95_vals = [v for k, v in fold_metrics.items() if 'HD95_' in k]
    if hd95_vals:
        fold_metrics['HD95_Total'] = float(np.mean(hd95_vals))

    return fold_metrics


def aggregate_experiment_metrics(base_path: Path, exp: dict, mode_type: str) -> list:
    """Finds folds, parses them, and computes cross-fold mean and std."""
    
    # --- SPECIAL CASE FOR STACKING ---
    if exp["id"] == "stacking":
        # Ensure we look for the correct EVAL folder name manually since the standard logic might fail
        inference_root = Path("/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_results/Dataset709_MagPhaseExp_simple_training_wm_gm_segmentation/nnUNetTrainerWandb__nnUNetPlans__2d/inference_results")
    else:
        ds_path = _find_dataset_dir(base_path, exp["id"], exp["name"], mode_type)
        if not ds_path:
            log_error(f"Dataset directory not found for {exp['name']} ({exp['id']})")
            return []

        trainer_paths = list(ds_path.glob("*nnUNetPlans__2d*")) if mode_type == "2d" else list(ds_path.glob("nnUnet3DCustomTrainer*3d_fullres*"))
        if not trainer_paths:
            trainer_paths = list(ds_path.glob("nnUnetTrainer*")) + list(ds_path.glob("nnUnet3DCustomTrainer*"))
            
        if not trainer_paths:
            log_error(f"Trainer directory not found in {ds_path}")
            return []

        trainer_path = trainer_paths[0]
        inference_root = trainer_path / "inference_results"

    if not inference_root.exists():
        log_error(f"inference_results missing in {trainer_path}")
        return []

    exp_results = []
    splits = [exp.get("dataset_split", "validation")] if exp.get("dataset_split") in ["validation", "test"] else ["validation"]

    for split in splits:
        for mode in exp.get("modes", ["single"]):
            try:
                target_dirs, is_ensemble = _get_target_directories(inference_root, exp["name"], mode, split, mode_type)
            except ValueError as e:
                log_error(str(e))
                continue
                
            if not target_dirs:
                log_error(f"No valid result directories found for {exp['name']} | Split: {split} | Mode: {mode}")
                continue

            all_classes = list(CLASS_MAP.values()) + ['Total']
            fold_data = {cls_name: {'Dice': [], 'HD95': []} for cls_name in all_classes}
            
            for f_dir in target_dirs:
                # --- Pass mode_type into parse_fold_metrics here ---
                metrics = parse_fold_metrics(f_dir, mode_type)
                for cls_name in all_classes:
                    if f'Dice_{cls_name}' in metrics: fold_data[cls_name]['Dice'].append(metrics[f'Dice_{cls_name}'])
                    if f'HD95_{cls_name}' in metrics: fold_data[cls_name]['HD95'].append(metrics[f'HD95_{cls_name}'])

            config_label = f"Cross-Fold ({mode})" if mode != "single" else "Cross-Fold"
            if split == "test": config_label = f"Test ({mode})"

            result_row = {
                "Experiment Name": exp['name'],
                "Mode": split,
                "Config": config_label,
            }

            for cls_name in all_classes:
                dices = fold_data[cls_name]['Dice']
                hd95s = fold_data[cls_name]['HD95']

                if dices:
                    result_row[f'Dice_{cls_name}_Mean'] = np.mean(dices)
                    result_row[f'Dice_{cls_name}_Std'] = None if is_ensemble else np.std(dices)
                if hd95s:
                    result_row[f'HD95_{cls_name}_Mean'] = np.mean(hd95s)
                    result_row[f'HD95_{cls_name}_Std'] = None if is_ensemble else np.std(hd95s)

            if len(result_row) > 3: 
                exp_results.append(result_row)

    return exp_results


def generate_latex_table(df: pd.DataFrame, output_filename: Path, title: str = "Metrics"):
    """Generates LaTeX table. Bolds best values, properly formats `Mean \\pm Std` or `Mean`."""
    
    best_values = {}
    for col in df.columns:
        if 'Mean' in col:
            mode = 'max' if 'Dice' in col else 'min'
            best_values[col] = df[col].max() if mode == 'max' else df[col].min()

    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        f"    \\caption{{Comparison of Dice and HD95 metrics ({title}). Best results are highlighted in bold.}}",
        r"    \label{tab:results}",
        r"    \resizebox{\textwidth}{!}{%",
        r"        \begin{tabular}{lcccccccccc}",
        r"            \toprule",
        r"            & \multicolumn{2}{c}{\textbf{White Matter}} & \multicolumn{2}{c}{\textbf{Gray Matter}} & \multicolumn{2}{c}{\textbf{Lesion WM}} & \multicolumn{2}{c}{\textbf{Lesion GM}} & \multicolumn{2}{c}{\textbf{Total Average}} \\",
        r"            \cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} \cmidrule(lr){10-11}",
        r"            \textbf{Config} & \textbf{Dice} & \textbf{HD95} & \textbf{Dice} & \textbf{HD95} & \textbf{Dice} & \textbf{HD95} & \textbf{Dice} & \textbf{HD95} & \textbf{Dice} & \textbf{HD95} \\",
        r"            \midrule"
    ]

    for _, row in df.iterrows():
        config_name = str(row.get('Experiment Name', ''))
        if row.get('Config'):
             config_name += f" ({row['Config']})"
             
        cells = [config_name.replace('_', r'\_')]
        
        def format_cell(metric_type, target_class):
            mean_val = row.get(f"{metric_type}_{target_class}_Mean", np.nan)
            std_val = row.get(f"{metric_type}_{target_class}_Std", np.nan)
            
            if pd.isna(mean_val):
                return "--"
                
            decimals = 3 if metric_type == "Dice" else 2
            mean_str = f"{float(mean_val):.{decimals}f}"
            
            if pd.isna(std_val) or std_val is None:
                txt = f"{mean_str}"
            else:
                txt = f"{mean_str} \\pm {float(std_val):.{decimals}f}"

            is_best = abs(float(mean_val) - best_values.get(f"{metric_type}_{target_class}_Mean", -1)) < 1e-9
            return f"$\\mathbf{{{txt}}}$" if is_best else f"${txt}$"

        all_classes = list(CLASS_MAP.values()) + ["Total"]
        for base_col in all_classes:
            cells.append(format_cell("Dice", base_col))
            cells.append(format_cell("HD95", base_col))
        
        lines.append("            " + " & ".join(cells) + r" \\")

    lines.extend([
        r"            \bottomrule",
        r"        \end{tabular}%",
        r"    }",
        r"\end{table*}"
    ])

    output_filename.write_text('\n'.join(lines))
    print(f"  -> Latex table saved to {output_filename}")


def main():
    parser = argparse.ArgumentParser(description="Strictly Aggregate nnUNet results.")
    args = parser.parse_args()
    
    for group in EXPERIMENT_GROUPS:
        mode_type = group.get("type", "3d")
        output_filename = group["output_filename"]
        base_path = BASE_PATH_2D if mode_type == "2d" else BASE_PATH_3D
        
        out_dir = Path(f"tables_{mode_type}")
        out_dir.mkdir(exist_ok=True)
        full_output_path = out_dir / output_filename
        
        print(f"\n[{mode_type.upper()}] Processing group '{output_filename}'...")

        all_results = []
        for exp in group["experiments"]:
            res = aggregate_experiment_metrics(base_path, exp, mode_type)
            all_results.extend(res)

        if all_results:
            df_results = pd.DataFrame(all_results)
            print(f"  -> Saving CSV to {full_output_path}")
            df_results.to_csv(full_output_path, index=False)
            
            tex_filename = full_output_path.with_suffix(".tex")
            generate_latex_table(df_results, tex_filename, title=f"{mode_type.upper()} Results - {output_filename}")
        else:
            print("  -> No results found for this group.")

if __name__ == "__main__":
    main()