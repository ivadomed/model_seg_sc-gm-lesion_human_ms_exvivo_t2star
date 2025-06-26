"""
Converts BIDS-structured dataset to the nnUNetv2 dataset format. Full details about
the format can be found here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md

Example of the input tree structure:

├── derivatives
│   └── labels
│       └── sub-01
│           └── anat
│               ├── sub-01_part-mag_chunk-02_T2starw_slice-123_label-combined_seg.nii.gz
│               ├── sub-01_part-mag_chunk-02_T2starw_slice-412_label-combined_seg.nii.gz
│               ├── sub-01_part-mag_chunk-02_T2starw_slice-605_label-combined_seg.nii.gz
│               └── sub-01_part-mag_chunk-02_T2starw_slice-656_label-combined_seg.nii.gz
└── sub-01
    └── anat
        ├── sub-01_part-mag_chunk-02_T2starw_slice-123.nii.gz
        ├── sub-01_part-mag_chunk-02_T2starw_slice-412.nii.gz
        ├── sub-01_part-mag_chunk-02_T2starw_slice-605.nii.gz
        ├── sub-01_part-mag_chunk-02_T2starw_slice-656.nii.gz
        ├── sub-01_part-mag_chunk-02_T2starw.nii.gz
        └── sub-01_part-phase_chunk-02_T2starw.nii.gz


With the corresponding nnUnetv2 dataset structure:

└── Dataset502_PostmortemSpineSlices
    ├── dataset.json
    ├── imagesTr
    │   ├── PostmortemSpineSlices_0000_0000.nii.gz
    │   ├── PostmortemSpineSlices_0001_0000.nii.gz
    │   ├── PostmortemSpineSlices_0002_0000.nii.gz
    │   └── PostmortemSpineSlices_0003_0000.nii.gz
    └── labelsTr
        ├── PostmortemSpineSlices_0000.nii.gz
        ├── PostmortemSpineSlices_0001.nii.gz
        ├── PostmortemSpineSlices_0002.nii.gz
        └── PostmortemSpineSlices_0003.nii.gz


"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


def ensure_3d_and_save(src_path: Path, dst_path: Path, force_dtype=None):
    img = nib.load(str(src_path))
    data = np.asarray(img.dataobj)

    # Force data to 3D
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    elif data.ndim == 4 and data.shape[-1] == 1:
        data = data[:, :, :, 0]
    if data.shape[1] == 1:
        data = np.transpose(data, (2, 1, 0))
    elif data.shape[2] == 1:
        data = np.transpose(data, (1, 2, 0))
    elif data.shape[0] == 1:
        data = np.transpose(data, (1, 0, 2))
    elif data.shape[0] < data.shape[1] and data.shape[0] < data.shape[2]:
        data = np.transpose(data, (1, 2, 0))

    dtype = force_dtype if force_dtype else data.dtype
    nib.save(nib.Nifti1Image(data.astype(dtype), img.affine, img.header), str(dst_path))


def parse_args():
    parser = argparse.ArgumentParser(description="Convert BIDS-style dataset into nnUNet v2 format (slice-based).")
    parser.add_argument("--path-data", required=True, help="Path to root of the BIDS dataset")
    parser.add_argument("--label-json", required=True, help="Path to JSON file specifying label names and values (e.g., '{\"background\": 0, ...}')")
    parser.add_argument("--path-out", default=None, help="(Optional) Output path (default: path-data + '_nnunet_raw')")
    parser.add_argument("--modality", default="T2starw", help="Modality name (e.g., T2starw, T1w). Default: T2starw")
    parser.add_argument("--taskname", default="Segmentation", help="Task name. Default: Segmentation")
    parser.add_argument("--tasknumber", type=int, default=502, help="Task number. Default: 502")
    return parser.parse_args()


def main():
    args = parse_args()

    path_data = Path(args.path_data)
    path_out = Path(args.path_out) if args.path_out else path_data.parent / "nnunet_raw"
    labels_root = path_data / "derivatives" / "labels"
    images_root = path_data

    target_dataset_name = f'Dataset{args.tasknumber:03d}_{args.taskname}'
    output_base = path_out / target_dataset_name
    imagesTr = output_base / "imagesTr"
    labelsTr = output_base / "labelsTr"

    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    # Load label definitions
    with open(args.label_json, 'r') as f:
        # Validate label dictionary
        label_dict = json.load(f)
        assert isinstance(label_dict, dict), "Label JSON must define a dictionary"
        assert all(isinstance(k, str) for k in label_dict.keys()), "All label keys must be strings"
        assert all(isinstance(v, int) for v in label_dict.values()), "All label values must be integers"
        assert len(set(label_dict.values())) == len(label_dict), "Label values must be unique"
        assert 0 in label_dict.values(), "Label values must include 0 (for background)"
        print(f"✅ Loaded label definitions: {label_dict}")

    label_files = sorted(labels_root.rglob("*_label-combined_seg.nii.gz"))
    sample_count = 0

    for label_path in label_files:
        fname = label_path.name.replace("_label-combined_seg.nii.gz", "")
        subject = label_path.parts[-3]
        image_path = images_root / subject / "anat" / f"{fname}.nii.gz"

        if not image_path.exists():
            print(f"❌ Image not found for {fname}, skipping.")
            continue

        sample_id = f'{args.taskname}_{sample_count:04d}'
        img_out = imagesTr / f'{sample_id}_0000.nii.gz'
        lbl_out = labelsTr / f'{sample_id}.nii.gz'

        ensure_3d_and_save(image_path, img_out, force_dtype=np.float32)
        ensure_3d_and_save(label_path, lbl_out, force_dtype=np.uint8)
        print(f"✅ Copied {image_path.name} and label to {sample_id}")
        sample_count += 1

    generate_dataset_json(
        output_folder=output_base,
        channel_names={0: args.modality},
        labels=label_dict,
        num_training_cases=sample_count,
        file_ending=".nii.gz",
        dataset_name=args.taskname,
        overwrite_image_reader_writer="NibabelIOWithReorient",
        release="1.0.0",
        reference="Julien Cohen-Adad, Postmortem SC MRI",
        license="CC-BY 4.0"
    )

    print(f"\n📁 Finished structuring {sample_count} samples into: {output_base}")


if __name__ == "__main__":
    main()