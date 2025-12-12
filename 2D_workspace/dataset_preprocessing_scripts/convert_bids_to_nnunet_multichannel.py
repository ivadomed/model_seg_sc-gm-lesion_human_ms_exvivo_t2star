"""
Converts a BIDS-structured dataset into the nnUNetv2 format, correctly handling
multi-channel phase images, enforcing isotropic headers, and ensuring full
geometric consistency.
(Version 10 - Added 'mag-one-channel' channel support)
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

def save_nifti(data: np.ndarray, affine: np.ndarray, header, output_path: Path, is_label: bool = False):
    """
    Saves a numpy array as a NIfTI file using a provided affine and header.
    If is_label is True, saves data as uint8 to prevent floating point errors.
    """
    if data.ndim == 2:
        data = data[..., np.newaxis]
    
    # Use uint8 for labels and float32 for images
    dtype = np.uint8 if is_label else np.float32
    new_nifti = nib.Nifti1Image(data.astype(dtype), affine, header)
    nib.save(new_nifti, output_path)

def parse_args():
    parser = argparse.ArgumentParser(description="Convert BIDS slices into nnUNet format with geometric consistency.")
    parser.add_argument("--path-data", required=True, type=Path, help="Path to root of the sliced BIDS dataset.")
    parser.add_argument("--path-out", required=True, type=Path, help="Output path for nnUNet raw data.")
    parser.add_argument("--taskname", default="Segmentation", type=str, help="Task name for nnU-Net.")
    parser.add_argument("--tasknumber", type=int, default=502, help="Task number for nnU-Net.")
    parser.add_argument("--label-suffixes", nargs="+", required=True, help="List of label suffixes in order (e.g., SC GM lesion).")
    
    parser.add_argument(
        "--label-mode",
        type=str,
        default="all",
        choices=["all", "tissues", "lesions", "merged_lesion", "sc_and_lesion"],
        help="Defines which set of labels to generate."
    )
    
    # --- MODIFIED ARGUMENT: Channel Configuration ---
    parser.add_argument(
        "--channel-config",
        type=str,
        default="mag_phase",
        choices=["mag_phase", "mag-one-channel"],
        help="Defines the input channels. 'mag_phase' uses both Magnitude (0) and Phase (1). 'mag-one-channel' uses only Magnitude (0)."
    )
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
        print(f"⚠️ Warning: Deleting existing directory {output_base}")
        shutil.rmtree(output_base)
    
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    images_root = path_data
    labels_root = path_data / "derivatives" / "labels"
    
    file_groups = defaultdict(lambda: {'labels': {}})

    for mag_path in images_root.rglob("*_part-mag_*.nii.gz"):
        key = mag_path.name.replace('_part-mag_', '_').removesuffix('.nii.gz')
        file_groups[key]['mag'] = mag_path
        file_groups[key]['subject'] = str(mag_path).split('/')[-3]

    # Only look for phase if we need it, though collecting it doesn't hurt.
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
    
    manifest = {}
    
    sample_count = 0
    for key, files in tqdm(file_groups.items(), desc=f"Converting (mode: {args.label_mode}, channels: {args.channel_config})"):
        mag_path = files.get('mag')
        phase_path = files.get('phase')
        subject = files.get('subject')
        
        # --- MODIFIED LOGIC: Check availability based on channel config ---
        if args.channel_config == "mag_phase":
            if not all([mag_path, phase_path, files['labels']]):
                continue
        elif args.channel_config == "mag-one-channel":
            if not all([mag_path, files['labels']]):
                continue
            phase_path = None # Ensure we don't use it even if it exists
            
        sample_id = f'{args.taskname}_{sample_count:04d}'
        
        mag_nifti_ref = nib.load(mag_path) # Use original mag NIfTI for geo reference
        mag_data = load_and_preprocess_image(mag_path)
        
        if args.channel_config == "mag_phase":
            phase_data = load_and_preprocess_image(phase_path)
            if mag_data.shape != phase_data.shape:
                 print(f"⚠️ Skipping {key}: Shape mismatch after preprocessing. Mag: {mag_data.shape}, Phase: {phase_data.shape}")
                 continue

        reference_affine = mag_nifti_ref.affine
        iso_affine = np.copy(reference_affine)
        np.fill_diagonal(iso_affine, [1.0, 1.0, 1.0, 1.0])
        iso_affine[:3, 3] = reference_affine[:3, 3]

        # Get paths for the required base labels (SC, GM)
        sc_path = files['labels'].get('SC')
        gm_path = files['labels'].get('GM')

        # Check that the required base labels (SC, GM) exist for this slice
        if not all([sc_path, gm_path]):
            print(f"⚠️ Skipping {key}: Missing required SC or GM label file.")
            continue

        # Get the lesion path (this one is OPTIONAL)
        lesion_path = files['labels'].get('lesion')

        # Load required label data into numpy arrays
        sc_data = load_and_preprocess_image(sc_path)
        gm_data = load_and_preprocess_image(gm_path)
        
        # If a lesion file exists, load it. Otherwise, create an empty (all zeros) array.
        if lesion_path:
            lesion_data = load_and_preprocess_image(lesion_path)
        else:
            # Create a blank lesion mask for healthy/control cases
            lesion_data = np.zeros_like(mag_data, dtype=np.uint8) 

        # Convert to boolean masks (True where the label is present)
        if mag_data.shape != sc_data.shape or mag_data.shape != gm_data.shape or mag_data.shape != lesion_data.shape:
            print(f"⚠️ Skipping {key}: Shape Mismatch!")
            continue # Skip this problematic case

        # --- Create base boolean masks ---
        sc_mask = sc_data.astype(bool)
        gm_mask = gm_data.astype(bool)
        lesion_mask = lesion_data.astype(bool)
        
        # --- Perform the logical subtraction to define WM ---
        wm_mask = sc_mask & ~gm_mask
        
        # Initialize the final combined label map
        combined_label_map = np.zeros_like(mag_data, dtype=np.uint8)
        
        # --- Apply labels based on selected mode ---
        if args.label_mode == "all":
            combined_label_map[wm_mask & ~lesion_mask] = 1
            combined_label_map[gm_mask & ~lesion_mask] = 2
            combined_label_map[wm_mask & lesion_mask] = 3
            combined_label_map[gm_mask & lesion_mask] = 4
        elif args.label_mode == "tissues":
            combined_label_map[wm_mask] = 1
            combined_label_map[gm_mask] = 2
        elif args.label_mode == "lesions":
            combined_label_map[wm_mask & lesion_mask] = 1
            combined_label_map[gm_mask & lesion_mask] = 2
        elif args.label_mode == "merged_lesion":
            combined_label_map[wm_mask & ~lesion_mask] = 1
            combined_label_map[gm_mask & ~lesion_mask] = 2
            combined_label_map[sc_mask & lesion_mask] = 3
        elif args.label_mode == "sc_and_lesion":
            combined_label_map[sc_mask & ~lesion_mask] = 1
            combined_label_map[sc_mask & lesion_mask] = 2

        # --- Save Images ---
        save_nifti(mag_data, iso_affine, mag_nifti_ref.header, imagesTr / f'{sample_id}_0000.nii.gz')
        
        if args.channel_config == "mag_phase":
            save_nifti(phase_data, iso_affine, mag_nifti_ref.header, imagesTr / f'{sample_id}_0001.nii.gz')
        
        save_nifti(combined_label_map, iso_affine, mag_nifti_ref.header, labelsTr / f'{sample_id}.nii.gz', is_label=True)
        
        manifest[sample_id] = subject + "/" + key.removesuffix('.nii.gz')
        sample_count += 1

    print(f"\nGenerating dataset.json for {sample_count} cases (mode: {args.label_mode})...")
    if sample_count == 0:
        print("❌ CRITICAL: No valid training cases were found.")
        return

    # Define labels and regions based on the selected mode
    if args.label_mode == "all":
        labels = {"background": 0, "WM": 1, "GM": 2, "lesion_WM": 3, "lesion_GM": 4}
        regions = {"WM": [1, 3], "GM": [2, 4], "lesion": [3, 4]}
    elif args.label_mode == "tissues":
        labels = {"background": 0, "WM": 1, "GM": 2}
        regions = {"WM": 1, "GM": 2}
    elif args.label_mode == "lesions":
        labels = {"background": 0, "lesion_WM": 1, "lesion_GM": 2}
        regions = {"lesion_WM": 1, "lesion_GM": 2, "lesion_any": [1, 2]}
    elif args.label_mode == "sc_and_lesion":
        labels = {"background": 0, "SC_healthy": 1, "lesion": 2}
        regions = {"SC": [1, 2], "lesion": 2}

    # --- MODIFIED LOGIC: Channel Names ---
    if args.channel_config == "mag_phase":
        channel_names = {"0": "magnitude", "1": "phase"}
    else: # mag-one-channel
        channel_names = {"0": "magnitude"}

    generate_dataset_json(
        str(output_base),
        channel_names=channel_names,
        labels=labels,
        num_training_cases=sample_count,
        file_ending=".nii.gz",
        dataset_name=args.taskname,
        regions=regions,
        overwrite_image_reader_writer="NibabelIOWithReorient"
    )
    print(f"✅ Finished. Created {sample_count} samples in {output_base}")
    
    manifest_path = output_base / "inference_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    print(f"✅ Saved inference manifest to {manifest_path}")

if __name__ == "__main__":
    main()