import os
import glob
import json
import argparse

def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def generate_manifest_only(bids_dir, experiment_name):
    
    # --- LOGIC FLAG ---
    # Must match original execution to ensure ID alignment
    use_single_channel = 'mag-one-channel' in experiment_name
    
    if use_single_channel:
        print(f"!! Experiment '{experiment_name}' detected. Logic: SINGLE CHANNEL (Magnitude only).")
    else:
        print(f"Experiment '{experiment_name}' detected. Logic: DUAL CHANNEL (Magnitude + Phase).")

    dataset_name = "MagPhase"
    case_counter = 0
    manifest = []
    
    # Path to derivatives (Must match original script logic)
    deriv_path = os.path.join(bids_dir, "derivatives", "labels")
    subjects = sorted(glob.glob(os.path.join(deriv_path, "sub-*")))
    
    print(f"Found {len(subjects)} subjects. scanning for mapping...")
    
    for subj_path in subjects:
        subj_name = os.path.basename(subj_path)
        
        # Files are in sub-X/anat/
        seg_files = sorted(glob.glob(os.path.join(subj_path, "anat", "*.nii.gz")))
        
        if not seg_files:
            continue
            
        for seg_file in seg_files:
            filename = os.path.basename(seg_file)
            
            # Construct Raw paths
            raw_mag_path = os.path.join(bids_dir, subj_name, "anat", filename)
            
            # Construct Phase filename
            phase_filename = filename.replace("part-mag", "part-phase")
            raw_phase_path = os.path.join(bids_dir, subj_name, "anat", phase_filename)
            
            # --- VALIDATION (Must match original script) ---
            # If the original script skipped a file, we must skip it here too
            # to keep the numbering consistent.
            
            if not os.path.exists(raw_mag_path):
                # skipped in original
                continue
            
            if not use_single_channel and not os.path.exists(raw_phase_path):
                # skipped in original
                continue
                
            # Create the ID
            case_id = f"{dataset_name}_{case_counter:04d}"
            
            # Add to Manifest
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

    # --- SAVE ---
    output_file = "manifest.json"
    save_json(manifest, output_file)
    print(f"Success! Generated manifest for {case_counter} cases.")
    print(f"Saved to current directory: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids", default="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset", help="Path to BIDS root directory")
    parser.add_argument("--experiment_name", default="mag-one-channel", type=str, help="Use 'mag-one-channel' if original dataset was single channel")
    args = parser.parse_args()
    
    generate_manifest_only(args.bids, args.experiment_name)