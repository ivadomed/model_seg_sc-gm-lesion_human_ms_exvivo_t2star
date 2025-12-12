import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np

def create_subject_splits(dataset_path: Path, manifest_path: Path, num_folds: int = 4):
    """
    Generates a subject-aware 5-fold cross-validation split file for nnU-Net.

    Args:
        dataset_path: Path to the nnU-Net dataset directory (e.g., .../nnunet_raw/Dataset508_...).
        manifest_path: Path to the inference_manifest.json created by the conversion script.
        num_folds: The number of folds for cross-validation.
    """
    print("Loading inference manifest...")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Group sample IDs by subject
    subject_to_samples = defaultdict(list)
    for sample_id, subject_info in manifest.items():
        # Assumes subject is the first part of the manifest value, e.g., "sub-01/..."
        subject_id = subject_info.split('/')[0]
        subject_to_samples[subject_id].append(sample_id)
    
    # Get a unique, sorted list of subjects to ensure reproducibility
    unique_subjects = sorted(list(subject_to_samples.keys()))
    print(f"Found {len(unique_subjects)} unique subjects.")

    # Shuffle subjects randomly for fold assignment
    np.random.seed(42) # Use a fixed seed for reproducibility
    # np.random.shuffle(unique_subjects)

    # Split subjects into k folds
    subject_folds = np.array_split(unique_subjects, num_folds)
    
    all_splits = []
    
    # Create the split dictionary for each fold
    for i in range(num_folds):
        val_subjects = subject_folds[i]
        train_subjects = [s for s in unique_subjects if s not in val_subjects]

        # Get all sample IDs for train and val subjects
        train_cases = [sample for sub in train_subjects for sample in subject_to_samples[sub]]
        val_cases = [sample for sub in val_subjects for sample in subject_to_samples[sub]]
        
        all_splits.append({
            "train": sorted(train_cases),
            "val": sorted(val_cases)
        })
        print(f"Fold {i}: {len(train_cases)} train cases ({len(train_subjects)} subjects), "
              f"{len(val_cases)} val cases ({len(val_subjects)} subjects)")

    # Save the splits file
    output_file = dataset_path / "splits_final.json"
    with open(output_file, 'w') as f:
        json.dump(all_splits, f, indent=4)
        
    print(f"\n✅ Successfully saved subject-aware splits to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create subject-aware splits for nnU-Net.")
    parser.add_argument("--dataset-path", required=True, type=Path, help="Path to the nnU-Net raw dataset directory (e.g., .../nnunet_raw/DatasetXXX_TaskName).")
    parser.add_argument("--manifest-path", required=True, type=Path, help="Path to the inference_manifest.json file.")
    args = parser.parse_args()
    
    create_subject_splits(args.dataset_path, args.manifest_path)