import os
import glob
import shutil
import json
import argparse
import numpy as np
from sklearn.model_selection import KFold

def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def preprocess_and_split(bids_dir, nnunet_raw_dir, dataset_id, experiment_name, use_smooth_labels=False):
    task_name = f"Dataset{dataset_id:03d}_MagPhase_{experiment_name}"
    out_base = os.path.join(nnunet_raw_dir, task_name)
    
    # --- LOGIC FLAG ---
    # Check if we are in single-channel mode (Magnitude only)
    use_single_channel = 'mag-one-channel' in experiment_name
    
    if use_single_channel:
        print(f"!! Experiment '{experiment_name}' detected. Running in SINGLE CHANNEL (Magnitude only) mode. !!")
    else:
        print(f"Experiment '{experiment_name}' detected. Running in DUAL CHANNEL (Magnitude + Phase) mode.")

    imagesTr = os.path.join(out_base, "imagesTr")
    if use_smooth_labels:
        labelsTr = os.path.join(nnunet_raw_dir, "smooth_labelsTr")
    else:
        labelsTr = os.path.join(out_base, "labelsTr")
        
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)
    
    # Dictionary to track which Subject owns which nnUNet Case Identifier
    subject_to_case_map = {}
    
    # List to store manifest data (Mapping ID -> Original Files)
    manifest = []
    
    dataset_name = "MagPhase"
    case_counter = 0
    
    # Path to derivatives based on your tree
    deriv_path = os.path.join(bids_dir, "derivatives", "labels")
    subjects = sorted(glob.glob(os.path.join(deriv_path, "sub-*")))
    
    print(f"Found {len(subjects)} subjects with derivatives.")
    
    for subj_path in subjects:
        subj_name = os.path.basename(subj_path) # e.g., sub-PML014
        
        # Files are in sub-X/anat/
        seg_files = sorted(glob.glob(os.path.join(subj_path, "anat", "*.nii.gz")))
        
        if not seg_files:
            continue
            
        # Initialize list for this subject
        if subj_name not in subject_to_case_map:
            subject_to_case_map[subj_name] = []
            
        for seg_file in seg_files:
            filename = os.path.basename(seg_file)
            
            # Construct Raw paths
            raw_mag_path = os.path.join(bids_dir, subj_name, "anat", filename)
            
            # Construct Phase filename
            phase_filename = filename.replace("part-mag", "part-phase")
            raw_phase_path = os.path.join(bids_dir, subj_name, "anat", phase_filename)
            
            # --- VALIDATION ---
            # Always check Mag exists. 
            # Only check Phase exists if we are NOT in single channel mode.
            if not os.path.exists(raw_mag_path):
                print(f"Skipping {filename}, missing magnitude file.")
                continue
            
            if not use_single_channel and not os.path.exists(raw_phase_path):
                print(f"Skipping {filename}, missing phase file (required for dual-channel).")
                continue
                
            # Create new nnUNet ID
            case_id = f"{dataset_name}_{case_counter:04d}"
            
            # --- COPYING ---
            # Destination Mapping
            dest_mag = os.path.join(imagesTr, f"{case_id}_0000.nii.gz")
            dest_seg = os.path.join(labelsTr, f"{case_id}.nii.gz")
            
            shutil.copy2(raw_mag_path, dest_mag)
            shutil.copy2(seg_file, dest_seg)

            # Only copy phase if NOT single channel mode
            if not use_single_channel:
                dest_phase = os.path.join(imagesTr, f"{case_id}_0001.nii.gz")
                shutil.copy2(raw_phase_path, dest_phase)
            
            # Record that this case belongs to this subject
            subject_to_case_map[subj_name].append(case_id)
            
            # --- MANIFEST ENTRY ---
            # Create a record mapping the new ID to the original files
            manifest_entry = {
                "nnunet_id": case_id,
                "subject": subj_name,
                "original_filename": filename,
                "original_path_seg": seg_file,
                "original_path_mag": raw_mag_path,
                "original_path_phase": raw_phase_path if not use_single_channel else None
            }
            manifest.append(manifest_entry)
            
            case_counter += 1
            
    # --- Generate dataset.json ---
    
    if use_single_channel:
        channel_conf = { "0": "Magnitude" }
    else:
        channel_conf = { "0": "Magnitude", "1": "Phase" }

    json_dict = {
        "channel_names": channel_conf,
        "labels": {
            "background": 0,
            "White Matter": 1,
            "Gray Matter": 2,
            "Lesion WM": 3,
            "Lesion GM": 4
        },
        "numTraining": case_counter,
        "file_ending": ".nii.gz",
        "name": task_name
    }
    save_json(json_dict, os.path.join(out_base, "dataset.json"))
    
    # --- SAVE MANIFEST ---
    manifest_path = os.path.join(out_base, "manifest.json")
    save_json(manifest, manifest_path)
    print(f"Manifest saved to: {manifest_path}")
    
    # --- Generate Custom Splits (Subject-based) ---
    print("Generating subject-based splits...")
    unique_subjects = list(subject_to_case_map.keys())
    kf = KFold(n_splits=4, shuffle=True, random_state=12345)
    
    splits = []
    
    for train_idx, val_idx in kf.split(unique_subjects):
        train_subjects = [unique_subjects[i] for i in train_idx]
        val_subjects = [unique_subjects[i] for i in val_idx]
        
        # Expand subjects to actual case IDs
        train_cases = []
        for s in train_subjects:
            train_cases.extend(subject_to_case_map[s])
            
        val_cases = []
        for s in val_subjects:
            val_cases.extend(subject_to_case_map[s])
            
        splits.append({
            "train": train_cases,
            "val": val_cases
        })
        
    # Ideally save splits in the same output folder, but keeping your original local save too
    with open("splits_final.json", 'w') as f:
        json.dump(splits, f, indent=4)
        
    print(f"Done. Processed {case_counter} cases from {len(subjects)} subjects.")
    print("splits_final.json generated locally.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids", required=True)
    parser.add_argument("--nnunet_raw", required=True)
    parser.add_argument("--id", type=int, default=501)
    parser.add_argument("--experiment_name", type=str, default="")
    args = parser.parse_args()
    parser.add_argument("--use_smooth_labels", type=bool, default=False)
    
    preprocess_and_split(args.bids, args.nnunet_raw, args.id, args.experiment_name, args.use_smooth_labels)