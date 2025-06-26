"""
combine_labels_modular.py

A general-purpose tool for combining multiple overlapping segmentation labels
(e.g., spinal cord, gray matter, lesion) into a single non-overlapping multi-class label map,
with user-defined logic for label relationships (such as label inclusion/subset constraints).

- Accepts any set of label suffixes and their prior inclusion logic.
- Automatically generates all valid, mutually exclusive label combinations.
- Produces a combined multi-class NIfTI file and a JSON mapping class names to integer labels.

Usage example:
    python combine_labels_modular.py \
        --input derivatives/labels/sub-01/anat/ \
        --output combined_labels.nii.gz \
        --suffixes SC GM lesion \
        --priors GM:SC lesion:SC \
        --json LABEL_VALUES.json
"""

import os
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import json
import itertools

def parse_args():
    parser = argparse.ArgumentParser(description="Combine labels into a multi-class label file, with flexible logic.")
    parser.add_argument("--input", required=True, help="Path to folder containing label NIfTI files.")
    parser.add_argument("--output", required=True, help="Output path for combined label NIfTI file.")
    parser.add_argument("--suffixes", nargs="+", required=True, help="List of label suffixes (e.g., SC GM lesion).")
    parser.add_argument("--priors", nargs="*", default=[], help="List of 'CHILD:PARENT' (e.g., GM:SC lesion:SC).")
    parser.add_argument("--json", required=True, help="Output path for JSON mapping.")
    return parser.parse_args()

def load_labels(label_dir, suffixes):
    # Assumes files named *_label-<suffix>_seg.nii.gz
    label_arrays = {}
    for suffix in suffixes:
        matches = list(Path(label_dir).glob(f"*label-{suffix}_seg.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"Could not find label file for suffix '{suffix}' in {label_dir}")
        arr = (nib.load(str(matches[0])).get_fdata() > 0).astype(np.uint8)
        label_arrays[suffix] = arr
    return label_arrays

def generate_combinations(label_names, priors):
    # priors is a list of (child, parent)
    all_combos = list(itertools.product([0, 1], repeat=len(label_names)))
    valid_combos = []
    for combo in all_combos:
        state = dict(zip(label_names, combo))
        valid = True
        for child, parent in priors:
            if state[child] and not state[parent]:
                valid = False
                break
        # Optionally skip pure background (all 0s)
        valid_combos.append(combo) if valid else None
    return valid_combos

def human_readable_class(combo, label_names):
    present = [name for i, name in enumerate(label_names) if combo[i]]
    absent = [name for i, name in enumerate(label_names) if not combo[i]]
    if not present:
        return "background"
    return "_".join(present) + ("_without_" + "_".join(absent) if absent else "")

def build_label_masks(label_arrays, combinations, label_names):
    ref_shape = next(iter(label_arrays.values())).shape
    final = np.zeros(ref_shape, dtype=np.uint8)
    class_dict = {}
    class_idx = 1  # 0 is background
    for combo in combinations:
        mask = np.ones(ref_shape, dtype=bool)
        class_str = human_readable_class(combo, label_names)
        for idx, present in enumerate(combo):
            name = label_names[idx]
            mask &= (label_arrays[name] > 0) if present else (label_arrays[name] == 0)
        # Background stays zero
        if class_str == "background":
            class_dict[class_str] = 0
            continue
        class_dict[class_str] = class_idx
        final[mask] = class_idx
        class_idx += 1
    return final, class_dict

def main():
    args = parse_args()
    suffixes = args.suffixes
    priors = [tuple(prior.split(":")) for prior in args.priors]
    # Load label files
    label_arrays = load_labels(args.input, suffixes)
    combinations = generate_combinations(suffixes, priors)
    combined, label_dict = build_label_masks(label_arrays, combinations, suffixes)
    # Save NIfTI
    first_label = list(label_arrays.values())[0]
    first_img = next(Path(args.input).glob(f"*label-{suffixes[0]}_seg.nii.gz"))
    nii = nib.load(str(first_img))
    nib.save(nib.Nifti1Image(combined, nii.affine, nii.header), args.output)
    # Save JSON
    with open(args.json, "w") as f:
        json.dump(label_dict, f, indent=2)
    print(f"✅ Saved {args.output}")
    print(f"✅ Saved {args.json}")

if __name__ == "__main__":
    main()
