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
        --path-label-in derivatives/labels \
        --path-label-out derivatives/labels_combined \
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
    """
    Parse command-line arguments for modular label combination in a folder structure.
    """
    parser = argparse.ArgumentParser(description="Combine labels for all subjects into multi-class files, with flexible logic.")
    parser.add_argument("--path-label-in", required=True, help="Root folder containing per-subject label NIfTI files.")
    parser.add_argument("--path-label-out", required=True, help="Root folder for saving combined labels (mirrors input structure).")
    parser.add_argument("--suffixes", nargs="+", required=True, help="List of label suffixes (e.g., SC GM lesion).")
    parser.add_argument("--priors", nargs="*", default=[], help="List of 'CHILD:PARENT' (e.g., GM:SC lesion:SC).")
    return parser.parse_args()

def load_labels(label_dir, suffixes):
    """
    Load binary label masks from NIfTI files for each specified label suffix.
    """
    label_arrays = {}
    for suffix in suffixes:
        matches = list(Path(label_dir).glob(f"*label-{suffix}_seg.nii.gz"))
        if not matches:
            return None  # Some labels missing for this subject/anat; skip it
        arr = (nib.load(str(matches[0])).get_fdata() > 0).astype(np.uint8)
        label_arrays[suffix] = arr
    return label_arrays

def generate_combinations(label_names, priors):
    """
    Generate all valid label presence/absence combinations, filtering out combinations
    that violate the specified inclusion (prior) logic.
    """
    all_combos = list(itertools.product([0, 1], repeat=len(label_names)))
    valid_combos = []
    for combo in all_combos:
        state = dict(zip(label_names, combo))
        valid = True
        for child, parent in priors:
            if state[child] and not state[parent]:
                valid = False
                break
        valid_combos.append(combo) if valid else None
    return valid_combos

def human_readable_class(combo, label_names):
    """
    Generate a human-readable class name for a given label combination.
    """
    present = [name for i, name in enumerate(label_names) if combo[i]]
    absent = [name for i, name in enumerate(label_names) if not combo[i]]
    if not present:
        return "background"
    return "_".join(present) + ("_without_" + "_".join(absent) if absent else "")

def build_label_masks(label_arrays, combinations, label_names):
    """
    Construct a non-overlapping multi-class label array according to valid label combinations,
    and generate the class mapping dictionary.
    """
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

def find_label_sets(label_dir, suffixes):
    """
    Returns a dict mapping base_prefix -> dict(suffix -> path) for all unique prefixes in the directory.
    """
    label_sets = {}
    for path in Path(label_dir).glob(f"*label-{suffixes[0]}_seg.nii.gz"):
        base_prefix = path.name.split(f"_label-{suffixes[0]}")[0]
        label_dict = {}
        for suffix in suffixes:
            label_file = path.parent / f"{base_prefix}_label-{suffix}_seg.nii.gz"
            if label_file.exists():
                label_dict[suffix] = label_file
        if len(label_dict) == len(suffixes):
            label_sets[base_prefix] = label_dict
    return label_sets

def main():
    """
    Recursively processes all subjects under the input directory, combining labels
    for each subject/session/anat folder and writing results to a mirrored output structure.
    """
    args = parse_args()
    input_root = Path(args.path_label_in)
    output_root = Path(args.path_label_out)
    suffixes = args.suffixes
    priors = [tuple(p.split(":")) for p in args.priors]

    # Recursively find all subject/anat folders containing labels
    for subject_dir in input_root.glob("sub-*"):
        input_anat = subject_dir / "anat"
        if not input_anat.exists():
            continue
        output_anat = output_root / subject_dir.name / "anat"
        output_anat.mkdir(parents=True, exist_ok=True)

        # Find all unique label sets (by prefix) in this folder
        label_sets = find_label_sets(input_anat, suffixes)
        for base_prefix, label_dict in label_sets.items():
            # Load all required label arrays
            label_arrays = {}
            for suffix in suffixes:
                arr = (nib.load(str(label_dict[suffix])).get_fdata() > 0).astype(np.uint8)
                label_arrays[suffix] = arr
            combinations = generate_combinations(suffixes, priors)
            combined, label_map = build_label_masks(label_arrays, combinations, suffixes)
            # Reference for affine/header
            ref_img = nib.load(str(label_dict[suffixes[0]]))
            output_file = output_anat / f"{base_prefix}_label-combined_seg.nii.gz"
            nib.save(nib.Nifti1Image(combined, ref_img.affine, ref_img.header), output_file)
            # Save per-file JSON
            json_file = output_anat / f"{base_prefix}_label-combined_classes.json"
            with open(json_file, "w") as f:
                json.dump(label_map, f, indent=2)
            print(f"✅ Saved: {output_file}")
            print(f"✅ Saved: {json_file}")

if __name__ == "__main__":
    main()
