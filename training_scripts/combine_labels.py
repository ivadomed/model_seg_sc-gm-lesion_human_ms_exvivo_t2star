"""
Combine segmentation labels for SC, GM and lesions from multiple files into a single NIfTI file.
This script processes label files in a BIDS structured dataset, combining them based on the presence of
lesions and the type of tissue (SC or GM). It saves the combined labels in a new directory structure.
"""

import os
import nibabel as nib
import numpy as np
from pathlib import Path
from collections import defaultdict

LABEL_VALUES = {
    "background": 0,
    "wm_wo_lesion": 1,
    "wm_with_lesion": 2,
    "gm_wo_lesion": 3,
    "gm_with_lesion": 4,
}

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
        print(f"❌ Missing one or more required labels in {label_dict}")
        return

    data = {k: (nib.load(p).get_fdata() > 0).astype(np.uint8) for k, p in label_dict.items()}
    ref_path = reference_image_path or list(label_dict.values())[0]
    ref_nib = nib.load(str(ref_path))
    shape = ref_nib.shape
    affine = ref_nib.affine
    header = ref_nib.header

    combined = np.zeros(shape, dtype=np.uint8)

    lesion_mask = data["lesion"]
    # check if lesion mask has non-zero values
    if np.sum(lesion_mask) == 0:
        print(f"⚠️ No lesions found in {output_path}, skipping lesion labels.")
        lesion_mask = np.zeros(shape, dtype=np.uint8)
    else:
        print(f"✅ Found lesions in {output_path}, combining lesion labels.")
        # display coordinates of lesion mask
        lesion_coords = np.argwhere(lesion_mask)
        if lesion_coords.size > 0:
            print(f"Lesion coordinates: {lesion_coords[:5]}... (showing first 5)")  
    
    gm_mask = data["GM"]
    wm_mask = np.logical_and(data["SC"], np.logical_not(gm_mask))

    combined[np.logical_and(wm_mask, np.logical_not(lesion_mask))] = LABEL_VALUES["wm_wo_lesion"]
    combined[np.logical_and(wm_mask, lesion_mask)] = LABEL_VALUES["wm_with_lesion"]
    combined[np.logical_and(gm_mask, np.logical_not(lesion_mask))] = LABEL_VALUES["gm_wo_lesion"]
    combined[np.logical_and(gm_mask, lesion_mask)] = LABEL_VALUES["gm_with_lesion"]

    nib.save(nib.Nifti1Image(combined, affine, header), output_path)
    print(f"✅ Saved combined label: {output_path}")


def main():
    input_root = Path("/Users/julien/data/Postmortem_SC_MRI/Processed_BIDS/derivatives/labels")
    output_root = input_root.parent / "combined_labels"

    for subject_dir in input_root.glob("sub-*"):
        input_anat = subject_dir / "anat"
        output_anat = output_root / subject_dir.name / "anat"
        output_anat.mkdir(parents=True, exist_ok=True)

        label_sets = find_label_sets(input_anat)

        for base_name, label_dict in label_sets.items():
            if not {"GM", "SC"}.issubset(label_dict):
                print(f"⚠️ Skipping {base_name} (missing GM or SC)")
                continue
            output_file = output_anat / f"{base_name}_label-combined_seg.nii.gz"

            # Try to find the original MRI to use for affine/header
            ref_image = (input_root.parent / subject_dir.name / "anat" / f"{base_name}.nii.gz")
            if not ref_image.exists():
                ref_image = None
            combine_labels(label_dict, output_file, ref_image)

if __name__ == "__main__":
    main()
