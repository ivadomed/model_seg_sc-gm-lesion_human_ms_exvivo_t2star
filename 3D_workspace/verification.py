import os
import glob
import nibabel as nib
import numpy as np
import argparse

def check_integrity(bids_root):
    print(f"Scanning BIDS root: {bids_root}")
    
    # Define paths based on your tree structure
    deriv_path = os.path.join(bids_root, "derivatives", "labels")
    
    # Find all potential segmentation files in derivatives
    # Pattern: derivatives/labels/sub-*/anat/*.nii.gz
    seg_files = glob.glob(os.path.join(deriv_path, "sub-*", "anat", "*.nii.gz"))
    
    if not seg_files:
        print("ERROR: No segmentation files found. Check your path or pattern.")
        return

    print(f"Found {len(seg_files)} potential segmentation files.")
    
    valid_cases = 0
    label_values_set = set()
    
    for seg_path in seg_files:
        # Reconstruct expected raw paths
        # Logic: derivatives/labels/sub-X/anat/FILE -> sub-X/anat/FILE
        # NOTE: Your tree showed the seg file has the same name as mag '...part-mag.nii.gz'
        # We need to handle this carefully.
        
        rel_path = os.path.relpath(seg_path, deriv_path) # sub-PML014/anat/file.nii.gz
        
        # 1. Identify Subject
        subj_name = rel_path.split(os.sep)[0] # sub-PML014
        
        # 2. Construct Raw Paths
        # Your raw data is in: root/sub-PML014/anat/
        raw_anat_dir = os.path.join(bids_root, subj_name, "anat")
        
        filename = os.path.basename(seg_path)
        
        # Assumption based on your tree: 
        # The file in derivatives is named "...part-mag.nii.gz" (which implies it might be a copy or just bad naming)
        # OR it is the segmentation mask. 
        
        # We assume the raw MAG file has the EXACT same name as the file in derivatives/labels
        mag_path = os.path.join(raw_anat_dir, filename)
        
        # We assume the PHASE file swaps 'part-mag' to 'part-phase'
        phase_filename = filename.replace("part-mag", "part-phase")
        phase_path = os.path.join(raw_anat_dir, phase_filename)
        
        # --- CHECKS ---
        missing = []
        if not os.path.exists(mag_path): missing.append("Magnitude")
        if not os.path.exists(phase_path): missing.append("Phase")
        
        if missing:
            print(f"[MISSING] {subj_name} / {filename}: {', '.join(missing)}")
            continue
            
        # Load Headers to check geometry
        try:
            img_seg = nib.load(seg_path)
            img_mag = nib.load(mag_path)
            img_phase = nib.load(phase_path)
            
            # Check 1: Shapes
            if not (img_seg.shape == img_mag.shape == img_phase.shape):
                print(f"[SHAPE MISMATCH] {filename}")
                print(f"  Seg: {img_seg.shape}, Mag: {img_mag.shape}, Phase: {img_phase.shape}")
                continue

            # Check 2: Affine (Orientation)
            # We use a loose tolerance because float errors happen
            if not np.allclose(img_seg.affine, img_mag.affine, atol=1e-3) or \
               not np.allclose(img_mag.affine, img_phase.affine, atol=1e-3):
                print(f"[AFFINE MISMATCH] {filename} - Images might not be registered!")
                continue

            # Check 3: Label Values (Crucial for nnU-Net)
            seg_data = img_seg.get_fdata()
            unique_labels = np.unique(seg_data).astype(int)
            label_values_set.update(unique_labels)
            
            # Heuristic check: If it has huge values (like MRI intensity > 100), it's not a mask
            if np.max(unique_labels) > 20:
                print(f"[WARNING] {filename} has max value {np.max(unique_labels)}. Is this actually a segmentation mask?")
            else:
                valid_cases += 1
                
        except Exception as e:
            print(f"[ERROR] Loading {filename}: {e}")

    print("\n" + "="*30)
    print(f"Summary:")
    print(f"Valid paired triplets found: {valid_cases}")
    print(f"Unique Label Values found across all files: {sorted(list(label_values_set))}")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bids", required=True, help="Path to BIDS root")
    args = parser.parse_args()
    check_integrity(args.bids)