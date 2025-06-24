import os
import nibabel as nib
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

LABEL_VALUES = {
    "background": 0,
    "wm_wo_lesion": 1,
    "wm_with_lesion": 2,
    "gm_wo_lesion": 3,
    "gm_with_lesion": 4,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Combine SC, GM, and lesion labels into one multi-class label file.")
    parser.add_argument("--path-label-in", required=True,
                        help="Path to the root of the label derivatives folder (e.g. derivatives/labels)")
    parser.add_argument("--path-label-out", required=True,
                        help="Path to save the combined label masks (e.g. derivatives/combined_labels)")
    return parser.parse_args()


def find_label_sets(label_dir):
    label_sets = defaultdict(dict)
    for path in Path(label_dir).rglob("*_label-*_seg.nii.gz"):
        base_prefix = path.name.split('_label-')[0]
        label_type = path.name.split('_label-')[1].split('_')[0]
        label_sets[base_prefix][label_type] = path
    return label_sets


def combine_labels(label_dict, output_path, reference_image_path=None):
    required = {"SC", "GM", "lesion"}
    if not required.issubset(label_dict.keys()):
        print(f"❌ Missing one or more required labels in: {label_dict}")
        return

    data = {k: (nib.load(p).get_fdata() > 0).astype(np.uint8) for k, p in label_dict.items()}
    ref_path = reference_image_path or list(label_dict.values())[0]
    ref_nib = nib.load(str(ref_path))
    shape = ref_nib.shape
    affine = ref_nib.affine
    header = ref_nib.header

    combined = np.zeros(shape, dtype=np.uint8)

    lesion_mask = data["lesion"]
    gm_mask = data["GM"]
    wm_mask = np.logical_and(data["SC"], np.logical_not(gm_mask))

    combined[np.logical_and(wm_mask, np.logical_not(lesion_mask))] = LABEL_VALUES["wm_wo_lesion"]
    combined[np.logical_and(wm_mask, lesion_mask)] = LABEL_VALUES["wm_with_lesion"]
    combined[np.logical_and(gm_mask, np.logical_not(lesion_mask))] = LABEL_VALUES["gm_wo_lesion"]
    combined[np.logical_and(gm_mask, lesion_mask)] = LABEL_VALUES["gm_with_lesion"]

    nib.save(nib.Nifti1Image(combined, affine, header), output_path)
    print(f"✅ Saved: {output_path}")


def main():
    args = parse_args()
    input_root = Path(args.path_label_in)
    output_root = Path(args.path_label_out)

    for subject_dir in input_root.glob("sub-*"):
        input_anat = subject_dir / "anat"
        output_anat = output_root / subject_dir.name / "anat"
        output_anat.mkdir(parents=True, exist_ok=True)

        label_sets = find_label_sets(input_anat)

        for base_name, label_dict in label_sets.items():
            if not {"GM", "SC", "lesion"}.issubset(label_dict.keys()):
                print(f"⚠️ Skipping {base_name} (missing one or more required labels)")
                continue

            output_file = output_anat / f"{base_name}_label-combined_seg.nii.gz"

            # Try to find the original image
            ref_image_path = (
                input_root.parent.parent / subject_dir.name / "anat" / f"{base_name}.nii.gz"
            )
            ref_image_path = ref_image_path if ref_image_path.exists() else None

            combine_labels(label_dict, output_file, ref_image_path)


if __name__ == "__main__":
    main()
