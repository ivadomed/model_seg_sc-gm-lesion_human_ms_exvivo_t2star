#!/usr/bin/env python
"""Build a *smaller* 3D nnU-Net plans variant for the distillation student.

Starts from the teacher's patch-variant plans and shrinks ONLY the network width
(`features_per_stage`), keeping patch size, spacing, normalization and `data_identifier`
identical -- so the student reuses the same preprocessed arrays AND the cached teacher logits
align voxel-for-voxel. Optionally drops the deepest stage(s) too.

Writes `<plans_name>.json` into the preprocessed dataset folder; train with `-p <plans_name>`.
"""
import argparse
import json
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, check=True).stdout.strip()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(REPO_DIR))
DEFAULT_DATASET_DIR = f"{PROJECT_ROOT}/nnUNet_data/nnUNet_preprocessed/Dataset011_3D_MagPhase"


def round_to_multiple_of_8(x):
    return max(8, int(round(x / 8.0)) * 8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    ap.add_argument("--src-plans", default="nnUNetPlans_p192x64x208.json")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--plans-name", default="nnUNetPlansDistillS", help="output plans identifier (also the JSON filename stem)")
    ap.add_argument("--width-scale", type=float, default=0.5, help="multiply features_per_stage by this (rounded to a multiple of 8, min 8)")
    ap.add_argument("--max-features", type=int, default=None, help="cap each stage's feature count after scaling (e.g. 160)")
    ap.add_argument("--drop-last-stages", type=int, default=0, help="remove this many of the deepest encoder stages")
    args = ap.parse_args()

    plans = json.load(open(os.path.join(args.dataset_dir, args.src_plans)))
    config = plans["configurations"][args.config]
    arch = config["architecture"]["arch_kwargs"]

    old_features = list(arch["features_per_stage"])
    n_stages = len(old_features) - args.drop_last_stages
    features = [round_to_multiple_of_8(f * args.width_scale) for f in old_features[:n_stages]]
    if args.max_features is not None:
        features = [min(f, args.max_features) for f in features]

    arch["features_per_stage"] = features
    arch["n_conv_per_stage"] = arch["n_conv_per_stage"][:n_stages]
    arch["n_conv_per_stage_decoder"] = arch["n_conv_per_stage_decoder"][:n_stages - 1]
    arch["kernel_sizes"] = arch["kernel_sizes"][:n_stages]
    arch["strides"] = arch["strides"][:n_stages]
    if "n_stages" in arch:
        arch["n_stages"] = n_stages

    plans["plans_name"] = args.plans_name
    plans["configurations"] = {args.config: config}  # keep ONLY the student config

    out_path = os.path.join(args.dataset_dir, args.plans_name + ".json")
    json.dump(plans, open(out_path, "w"), indent=4)

    print(f"[student plans] {args.src_plans} :: {args.config}")
    print(f"  features_per_stage: {old_features}  ->  {features}")
    print(f"  n_stages: {len(old_features)} -> {n_stages}   data_identifier: {config['data_identifier']}")
    print(f"  patch_size: {config['patch_size']} (unchanged)")
    print(f"  wrote: {out_path}")
    print(f"  train with:  -p {args.plans_name} -c {args.config} -tr nnUnet3DDistillTrainer")


if __name__ == "__main__":
    main()
