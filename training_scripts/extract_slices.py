"""
Extract annotated slices from MRI image and segmentation masks

This python script extracts the segmentations masks slices which are annotated as well as the corresponding mri image slice.
It creates a new folder with the BIDS dataset format.

How to run:

    $ python extract_slices.py --path-data /path/to/data --path-out /path/to/project/folder
    
Author: Julien Cohen-Adad
"""

import os
import argparse
from pathlib import Path
import numpy as np
from nibabel import load, Nifti1Image, save
from tqdm import tqdm
from time import time


def parse_args():
    parser = argparse.ArgumentParser(description='Extract annotated slices from dataset.')
    # Mandatory arguments
    parser.add_argument('--path-data', required=True, help='Path to BIDS structured dataset.')
    parser.add_argument('--labels', nargs='+', required=True,
                        help='List of label types to extract (e.g., SC GM lesion)')
    parser.add_argument('--path-out', required=True, help='Path to output directory.')
    # Optional arguments
    parser.add_argument('--label-folder', default='derivatives/labels',
                        help='Relative path to the label folder inside the dataset.')
    parser.add_argument('--adjacent-slices', type=int, default=0, 
                        help='Number of adjacent MRI slices to include before and after each labeled slice (labels not included). Default: 0')
    return parser.parse_args()


def find_subject_folders(data_path):
    return [f.path for f in os.scandir(data_path) if f.is_dir() and 'derivatives' not in f.path]


def find_files(folder, extension='.nii.gz'):
    anat_path = Path(folder) / 'anat'
    if anat_path.exists():
        return [f.path for f in os.scandir(anat_path) if f.name.endswith(extension)]
    return []


def get_available_labels(label_folder, base_name, expected_labels):
    available = {}
    for f in os.scandir(label_folder):
        if f.name.endswith('.nii.gz') and base_name in f.name:
            for label in expected_labels:
                if f'label-{label}' in f.name or f'_{label}_' in f.name:
                    available[label] = f.path
    return available


def extract_modality_from_filename(filename):
    parts = filename.split('_')
    for part in parts:
        if '-' in part:
            _, val = part.split('-', 1)
            return val
    return 'modality'  # fallback


def save_nifti_slice(data, affine, out_path):
    dtype = np.uint8 if np.array_equal(data, data.astype(int)) else np.float32
    img = Nifti1Image(data.astype(dtype), affine)
    save(img, str(out_path))


def process_subject(mri_path, label_base_path, output_images_dir, output_labels_dir, expected_labels, adjacent_slices):
    folder_name = Path(mri_path).name
    files = find_files(mri_path)
    count = 0
    slice_log = []

    for file_path in files:
        file_name = Path(file_path).name
        base_name = file_name.replace('.nii.gz', '')
        subject = base_name.split('_')[0]  # e.g., sub-01
        mri_img = load(file_path)
        mri_data = np.asarray(mri_img.dataobj)

        out_anat_path = output_images_dir / folder_name / 'anat'
        out_anat_path.mkdir(parents=True, exist_ok=True)

        label_folder = label_base_path / folder_name / 'anat'
        label_paths = get_available_labels(label_folder, base_name, expected_labels)

        if not all(label in label_paths for label in expected_labels):
            return count, []

        label_data = {
            label: (load(label_paths[label]), np.asarray(load(label_paths[label]).dataobj))
            for label in expected_labels
        }

        nb_slices = label_data[expected_labels[0]][1].shape[1]
        labeled_slices = [i for i in range(nb_slices) if np.sum(label_data[expected_labels[0]][1][:, i, :]) > 0]
        extracted_slices = set()

        for slice_i in labeled_slices:
            slice_out_path = output_labels_dir / folder_name / 'anat'
            slice_out_path.mkdir(parents=True, exist_ok=True)

            for offset in range(-adjacent_slices, adjacent_slices + 1):
                idx = slice_i + offset
                if idx < 0 or idx >= nb_slices or idx in extracted_slices:
                    continue

                extracted_slices.add(idx)
                mri_slice = mri_data[:, idx, :]
                mri_out_name = f"{base_name}_slice-{idx}.nii.gz"
                save_nifti_slice(mri_slice, mri_img.affine, out_anat_path / mri_out_name)

            # Save labels only at the center slice (slice_i)
            count += 1
            slice_log.append(slice_i)
            mri_center_slice = mri_data[:, slice_i, :]
            for label in expected_labels:
                label_slice = label_data[label][1][:, slice_i, :]
                label_filename = f"{base_name}_slice-{slice_i}_label-{label}_seg.nii.gz"
                save_nifti_slice(label_slice, label_data[label][0].affine, slice_out_path / label_filename)

            print(f"\r- {subject}: {', '.join(map(str, slice_log))}", end='', flush=True)

    print()  # Newline after subject
    return count, slice_log


def main():
    args = parse_args()

    path_data = Path(args.path_data)
    path_labels = path_data / args.label_folder
    path_out_images = Path(args.path_out)
    path_out_labels = path_out_images / 'derivatives' / 'labels'

    path_out_images.mkdir(parents=True, exist_ok=True)
    path_out_labels.mkdir(parents=True, exist_ok=True)

    total_count = 0
    print("Extracting slices:")
    for folder_path in find_subject_folders(path_data):
        t0 = time()
        count, slice_indices = process_subject(
            folder_path, path_labels, path_out_images, path_out_labels, args.labels, args.adjacent_slices
        )
        total_count += count

    print(f'\nDone! Number of extracted slices: {total_count}')


if __name__ == '__main__':
    main()
