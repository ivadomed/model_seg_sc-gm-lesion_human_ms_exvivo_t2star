#!/usr/bin/env python3
"""
Read an experiment config JSON and emit shell-eval-able exports for the runners.
Also writes the trainer_config sub-dict to a temp file and prints its path (for
$NNUNET_EXP_CONFIG, consumed by the custom trainers).

Usage (inside a runner):
    eval "$(.venv/bin/python experiments/load_config.py experiments/3D/adamw_baseline.json)"
"""
import json, os, sys, tempfile

cfg_path = sys.argv[1]
cfg = json.load(open(cfg_path))
exp_name = os.path.splitext(os.path.basename(cfg_path))[0]

d = cfg["dataset"]
out = {
    "EXP_NAME": exp_name,
    "EXP_DIM": cfg["dim"],                       # "2D" | "3D"
    "EXP_FAMILY": cfg.get("family", "misc"),
    "DATASET_ID": d["id"],
    "DATASET_NAME": d["name"],
    "DATASET_CHANNELS": d.get("channels", "mag_phase"),
    "DATASET_LABEL_MODE": d.get("label_mode", "all"),
    "CONFIGURATION": cfg.get("configuration", "3d_fullres" if cfg["dim"] == "3D" else "2d"),
    "TRAINER": cfg["trainer"],
    "PLANS": cfg.get("plans", "nnUNetPlans"),
    # optional plans-level patch override (collapsed patch sweep); empty = use planner default
    "PATCH_SIZE": ",".join(str(x) for x in cfg["patch_size"]) if cfg.get("patch_size") else "",
}

# trainer_config -> temp json for $NNUNET_EXP_CONFIG
tc = cfg.get("trainer_config", {})
fd, tc_path = tempfile.mkstemp(prefix=f"expcfg_{exp_name}_", suffix=".json")
with os.fdopen(fd, "w") as f:
    json.dump(tc, f)
out["TRAINER_CONFIG_PATH"] = tc_path

for k, v in out.items():
    print(f'export {k}="{v}"')
