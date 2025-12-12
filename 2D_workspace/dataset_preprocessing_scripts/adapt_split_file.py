import json
import argparse
from pathlib import Path

def adapt_split_file(source_split_path, target_dataset_dir, old_task_name, new_task_name):
    """
    Adapts a splits_final.json file from a previous experiment for a new one
    by replacing the task name in all case identifiers.
    """
    print(f"Loading source split file from: {source_split_path}")
    try:
        with open(source_split_path, 'r') as f:
            splits_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: The source split file was not found at {source_split_path}")
        print("   Please ensure the path is correct and the file exists.")
        exit(1)
    except json.JSONDecodeError:
        print(f"❌ CRITICAL ERROR: The source split file at {source_split_path} is not a valid JSON file.")
        exit(1)

    adapted_splits = []
    
    # The case identifiers look like 'TASKNAME_XXXX'
    old_prefix = f"{old_task_name}_"
    new_prefix = f"{new_task_name}_"

    print(f"Replacing prefix '{old_prefix}' with '{new_prefix}'...")

    # Iterate through each fold's dictionary in the list
    for fold in splits_data:
        adapted_fold = {
            "train": [case.replace(old_prefix, new_prefix) for case in fold.get("train", [])],
            "val": [case.replace(old_prefix, new_prefix) for case in fold.get("val", [])]
        }
        adapted_splits.append(adapted_fold)

    target_path = Path(target_dataset_dir) / "splits_final.json"
    target_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists

    print(f"Saving adapted split file to: {target_path}")
    with open(target_path, 'w') as f:
        json.dump(adapted_splits, f, indent=4)
        
    print("✅ Adaptation complete.")
    print(f"   Original train cases in fold 0: {len(splits_data[0].get('train', []))}")
    print(f"   Adapted train cases in fold 0:  {len(adapted_splits[0].get('train', []))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt an nnU-Net splits_final.json for a new experiment.")
    parser.add_argument("--source-split-json", required=True, type=Path, help="Path to the source (golden) splits_final.json file.")
    parser.add_argument("--target-dataset-dir", required=True, type=Path, help="Path to the target nnU-Net raw dataset directory.")
    parser.add_argument("--old-task-name", required=True, type=str, help="The task name prefix from the source file (e.g., MagPhaseExp_old).")
    parser.add_argument("--new-task-name", required=True, type=str, help="The new task name prefix for the target file (e.g., MagPhaseExp_new).")
    
    args = parser.parse_args()
    
    adapt_split_file(args.source_split_json, args.target_dataset_dir, args.old_task_name, args.new_task_name)
