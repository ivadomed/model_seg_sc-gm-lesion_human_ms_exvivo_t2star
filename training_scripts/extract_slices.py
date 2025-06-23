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
    img = Nifti1Image(data, affine)
    save(img, str(out_path))


def process_subject(mri_path, label_base_path, output_images_dir, output_labels_dir, expected_labels):
    folder_name = Path(mri_path).name
    print(f"Processing {folder_name}...")
    files = find_files(mri_path)
    count = 0

    for file_path in files:
        file_name = Path(file_path).name
        base_name = file_name.replace('.nii.gz', '')
        subject, chunk = base_name.split('_')[:2]
        modality = extract_modality_from_filename(file_name)

        mri_img = load(file_path)
        mri_data = np.asarray(mri_img.dataobj)

        out_anat_path = output_images_dir / folder_name / 'anat'
        out_anat_path.mkdir(parents=True, exist_ok=True)
        save_nifti_slice(mri_data, mri_img.affine, out_anat_path / file_name)

        label_folder = label_base_path / folder_name / 'anat'
        label_paths = get_available_labels(label_folder, base_name, expected_labels)

        if not all(label in label_paths for label in expected_labels):
            return count  # Skip if any expected label is missing

        label_data = {
            label: (load(label_paths[label]), np.asarray(load(label_paths[label]).dataobj))
            for label in expected_labels
        }

        nb_slices = label_data[expected_labels[0]][1].shape[1]
        for slice_i in range(nb_slices):
            if np.sum(label_data[expected_labels[0]][1][:, slice_i, :]) == 0:
                continue

            count += 1
            slice_out_path = output_labels_dir / folder_name / 'anat'
            slice_out_path.mkdir(parents=True, exist_ok=True)

            mri_slice = mri_data[:, slice_i, :]
            mri_out_name = f'{subject}_{chunk}-slice-{slice_i}_{modality}.nii.gz'
            save_nifti_slice(mri_slice, mri_img.affine, out_anat_path / mri_out_name)

            for label in expected_labels:
                label_slice = label_data[label][1][:, slice_i, :]
                label_filename = f'{subject}_{chunk}-slice-{slice_i}_{modality}_label-{label}_seg.nii.gz'
                save_nifti_slice(label_slice, label_data[label][0].affine, slice_out_path / label_filename)

    return count



def main():
    args = parse_args()

    path_data = Path(args.path_data)
    path_labels = path_data / args.label_folder
    path_out_images = Path(args.path_out) / 'data_extracted'
    path_out_labels = path_out_images / 'derivatives' / 'labels'

    path_out_images.mkdir(parents=True, exist_ok=True)
    path_out_labels.mkdir(parents=True, exist_ok=True)

    total_count = 0
    for folder_path in find_subject_folders(path_data):
        t0 = time()
        total_count += process_subject(folder_path, path_labels, path_out_images, path_out_labels, args.labels)
        print(f"Done {Path(folder_path).name} in {time() - t0:.1f}s")

    print(f'--- Finished: extracted {total_count} slices ---')


if __name__ == '__main__':
    main()
