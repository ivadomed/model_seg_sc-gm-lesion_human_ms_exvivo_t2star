

"""
Converts a BIDS-structured dataset into the nnUNetv2 format, correctly handling
multi-channel phase images, enforcing isotropic headers, and ensuring full
geometric consistency.
(Version 6 - Final Definitive Fix)
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from tqdm import tqdm
from collections import defaultdict

def load_and_preprocess_image(image_path: Path) -> np.ndarray:
    """
    Loads a NIfTI image and handles conversion of 3-channel RGB images 
    to single-channel grayscale by averaging the channels.
    """
    img = nib.load(image_path)
    data = np.asarray(img.dataobj).astype(np.float32)
    if data.ndim == 3 and data.shape[-1] in [3, 4]:
        # Check if it's a grayscale image saved in RGB format
        if np.array_equal(data[..., 0], data[..., 1]) and np.array_equal(data[..., 0], data[..., 2]):
            data = data[..., 0]
        else: 
            data = data.mean(axis=-1)
            
    return data

def save_nifti(data: np.ndarray, affine: np.ndarray, header, output_path: Path):
    """
    Saves a numpy array as a NIfTI file using a provided affine and header.
    """
    if data.ndim == 2:
        data = data[..., np.newaxis]
    
    new_nifti = nib.Nifti1Image(data.astype(np.float32), affine, header)
    nib.save(new_nifti, output_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert BIDS slices into nnUNet format with geometric consistency.")
    parser.add_argument("--path-data", required=True, type=Path, help="Path to root of the sliced BIDS dataset.")
    parser.add_argument("--path-out", required=True, type=Path, help="Output path for nnUNet raw data.")
    parser.add_argument("--taskname", default="Segmentation", type=str, help="Task name for nnU-Net.")
    parser.add_argument("--tasknumber", type=int, default=502, help="Task number for nnU-Net.")
    parser.add_argument("--label-suffixes", nargs="+", required=True, help="List of label suffixes in order (e.g., SC GM lesion).")
    return parser.parse_args()

def main():
    args = parse_args()
    path_data = args.path_data
    path_out = args.path_out
    
    dataset_name_full = f'Dataset{args.tasknumber:03d}_{args.taskname}'
    output_base = path_out / dataset_name_full
    imagesTr = output_base / "imagesTr"
    labelsTr = output_base / "labelsTr"

    if output_base.exists():
        shutil.rmtree(output_base)
    
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    images_root = path_data
    labels_root = path_data / "derivatives" / "labels"
    
    file_groups = defaultdict(lambda: {'labels': {}})

    for mag_path in images_root.rglob("*_part-mag_*.nii.gz"):
        key = mag_path.name.replace('_part-mag_', '_').removesuffix('.nii.gz')
        file_groups[key]['mag'] = mag_path

    for phase_path in images_root.rglob("*_part-phase_*.nii.gz"):
        key = phase_path.name.replace('_part-phase_', '_').removesuffix('.nii.gz')
        file_groups[key]['phase'] = phase_path
    
    for label_path in labels_root.rglob("*_seg.nii.gz"):
        for suffix in args.label_suffixes:
            if f"_label-{suffix}_" in label_path.name:
                key_part = label_path.name.split(f"_label-{suffix}_")[0]
                key = key_part.replace('_part-mag_', '_')
                file_groups[key]['labels'][suffix] = label_path
                break
    
    sample_count = 0
    for key, files in tqdm(file_groups.items(), desc="Converting to nnUNet format"):
        mag_path = files.get('mag')
        phase_path = files.get('phase')
        
        if not all([mag_path, phase_path, files['labels']]):
            continue
            
        sample_id = f'{args.taskname}_{sample_count:04d}'
        
        mag_nifti_ref = nib.load(mag_path) # Use original mag NIfTI for geo reference
        mag_data = load_and_preprocess_image(mag_path)
        phase_data = load_and_preprocess_image(phase_path)
        
        if mag_data.shape != phase_data.shape:
             print(f"⚠️ Skipping {key}: Shape mismatch after preprocessing. Mag: {mag_data.shape}, Phase: {phase_data.shape}")
             continue

        reference_affine = mag_nifti_ref.affine
        iso_affine = np.copy(reference_affine)
        np.fill_diagonal(iso_affine, [1.0, 1.0, 1.0, 1.0])
        iso_affine[:3, 3] = reference_affine[:3, 3]

        combined_label_map = np.zeros_like(mag_data, dtype=np.uint8)
        for i, suffix in enumerate(args.label_suffixes):
            label_path = files['labels'].get(suffix)
            if label_path and label_path.exists():
                bit_value = 2**i
                label_data = load_and_preprocess_image(label_path)
                if label_data.shape == combined_label_map.shape:
                    combined_label_map += label_data.astype(np.uint8) * bit_value

        save_nifti(mag_data, iso_affine, mag_nifti_ref.header, imagesTr / f'{sample_id}_0000.nii.gz')
        save_nifti(phase_data, iso_affine, mag_nifti_ref.header, imagesTr / f'{sample_id}_0001.nii.gz')
        save_nifti(combined_label_map, iso_affine, mag_nifti_ref.header, labelsTr / f'{sample_id}.nii.gz')
        
        sample_count += 1

    print(f"\nGenerating dataset.json for {sample_count} cases...")
    if sample_count == 0:
        print("❌ CRITICAL: No valid training cases were found.")
        return

    num_classes = len(args.label_suffixes)
    labels = {"background": 0}
    for i in range(1, 2**num_classes):
        combo_name = "_".join([args.label_suffixes[j] for j in range(num_classes) if (i >> j) & 1])
        labels[combo_name] = i
        
    regions = {}
    for i, suffix in enumerate(args.label_suffixes):
        region_values = [value for key, value in labels.items() if suffix in key]
        regions[suffix] = region_values

    generate_dataset_json(
        str(output_base),
        channel_names={"0": "magnitude", "1": "phase"},
        labels=labels,
        num_training_cases=sample_count,
        file_ending=".nii.gz",
        dataset_name=args.taskname,
        regions=regions,
        overwrite_image_reader_writer="NibabelIOWithReorient"
    )
    print(f"✅ Finished. Created {sample_count} samples in {output_base}")

if __name__ == "__main__":
    main()