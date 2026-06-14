#!/usr/bin/env python3
"""
Create a plans VARIANT that only changes the training patch_size, reusing the SAME preprocessed
data (keeps data_identifier), so a patch sweep needs no re-preprocessing. Prints the new plans
identifier to stdout.

  python make_patch_plans.py <preprocessed_dataset_dir> <base_plans> <configuration> <patch_csv>
e.g. ... nnUNet_data/nnUNet_preprocessed/Dataset011_3D_MagPhase nnUNetPlans 3d_fullres 64,288,112
"""
import json, os, sys

pp_dir, base_plans, config, patch_csv = sys.argv[1:5]
patch = [int(x) for x in patch_csv.split(",")]
base = json.load(open(os.path.join(pp_dir, f"{base_plans}.json")))
tag = "p" + "x".join(str(x) for x in patch)
new_name = f"{base_plans}_{tag}"
base["plans_name"] = new_name
cfg = base["configurations"][config]
cfg["patch_size"] = patch
cfg["data_identifier"] = f"{base_plans}_{config}"      # reuse the SAME preprocessed arrays
out = os.path.join(pp_dir, f"{new_name}.json")
json.dump(base, open(out, "w"), indent=4, sort_keys=False)
print(new_name)
