"""
Extract annotated slices from MRI image (magnitude and phase) and segmentation masks.

This python script finds annotated slices in a 3D volume and extracts the
corresponding 2D slices from both the magnitude and phase images.
It creates a new folder with the BIDS dataset format and carries forward the
label definition JSON file for the next step.
"""

import os
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Extract annotated slices from dataset.')
    parser.add_argument('--path-data', required=True, type=Path, help='Path to BIDS structured dataset.')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='List of label types to extract (e.g., SC GM lesion)')
    parser.add_argument('--path-out', required=True, type=Path, help='Path to output directory.')
    parser.add_argument('--label-folder', required=True, type=Path,
                        help='Path to the folder containing the label files (can be absolute or relative to path-data).')
    return parser.parse_args()

def find_subject_folders(data_path):
    """Finds all sub-* folders in the root directory."""
    return [d for d in Path(data_path).iterdir() if d.is_dir() and d.name.startswith('sub-')]

def save_nifti_slice(data, affine, header, out_path):
    """Saves a 2D numpy array as a NIfTI file."""
    if data.ndim < 2: return
    img = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(img, str(out_path))

def process_subject(subject_dir, label_base_path, output_dir, expected_labels):
    """Processes a single subject, finding mag/phase pairs, and extracting labeled slices."""
    anat_path = subject_dir / 'anat'
    if not anat_path.exists(): return 0

    mag_files = list(anat_path.glob('*_part-mag_*.nii.gz'))
    slice_count = 0

    for mag_path in tqdm(mag_files, desc=f"Processing {subject_dir.name}", leave=False):
        phase_filename = mag_path.name.replace('_part-mag_', '_part-phase_')
        phase_path = mag_path.with_name(phase_filename)
        if not phase_path.exists(): continue

        try:
            mag_img = nib.load(mag_path)
            mag_data = np.asarray(mag_img.dataobj)
            phase_img = nib.load(phase_path)
            phase_data = np.asarray(phase_img.dataobj)
        except Exception: continue

        base_name_for_label = mag_path.name.replace('.nii.gz', '')
        label_folder = label_base_path / subject_dir.name / 'anat'
        main_label_path = label_folder / f"{base_name_for_label}_label-{expected_labels[0]}_seg.nii.gz"
        if not main_label_path.exists(): continue

        main_label_data = np.asarray(nib.load(main_label_path).dataobj)
        annotated_slice_indices = [i for i in range(main_label_data.shape[1]) if np.any(main_label_data[:, i, :])]
        if not annotated_slice_indices: continue
        
        output_images_anat = output_dir / subject_dir.name / 'anat'
        output_labels_anat = output_dir / 'derivatives' / 'labels' / subject_dir.name / 'anat'
        output_images_anat.mkdir(parents=True, exist_ok=True)
        output_labels_anat.mkdir(parents=True, exist_ok=True)

        for slice_idx in annotated_slice_indices:
            mag_slice_fname = f"{base_name_for_label}_slice-{slice_idx}.nii.gz"
            phase_slice_fname = mag_slice_fname.replace('_part-mag_', '_part-phase_')
            save_nifti_slice(mag_data[:, slice_idx, :], mag_img.affine, mag_img.header, output_images_anat / mag_slice_fname)
            save_nifti_slice(phase_data[:, slice_idx, :], phase_img.affine, phase_img.header, output_images_anat / phase_slice_fname)

            for label_type in expected_labels:
                label_path = label_folder / f"{base_name_for_label}_label-{label_type}_seg.nii.gz"
                if label_path.exists():
                    label_img = nib.load(label_path)
                    out_label_fname = f"{base_name_for_label}_slice-{slice_idx}_label-{label_type}_seg.nii.gz"
                    save_nifti_slice(np.asarray(label_img.dataobj)[:, slice_idx, :], label_img.affine, label_img.header, output_labels_anat / out_label_fname)
            slice_count += 1
    return slice_count

def main():
    args = parse_args()
    path_data = args.path_data
    path_out = args.path_out
    
    path_labels_in = args.label_folder if args.label_folder.is_absolute() else path_data / args.label_folder

    path_out.mkdir(parents=True, exist_ok=True)
    
    subject_folders = find_subject_folders(path_data)
    total_slices = 0
    print(f"Found {len(subject_folders)} subjects. Starting slice extraction...")

    for subject_dir in tqdm(subject_folders, desc="Overall Progress"):
        count = process_subject(subject_dir, path_labels_in, path_out, args.labels)
        total_slices += count

    print("\nSearching for label definition file...")
    json_path = next(path_labels_in.rglob("*_classes.json"), None)
    if json_path:
        dest_dir = path_out / 'derivatives' / 'labels'
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(json_path, dest_dir)
        print(f"✅ Copied {json_path.name} to {dest_dir}")
    else:
        print(f"⚠️ Warning: Could not find a _classes.json file in {path_labels_in}. The next step might fail.")

    print(f'\nDone! Extracted a total of {total_slices} slices.')

if __name__ == '__main__':
    main()
