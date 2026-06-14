#!/usr/bin/env python3
"""
Generate the experiment config JSONs (documents the experiment matrix in one place).
3D experiments share Dataset011 (mag+phase) and vary only trainer_config, EXCEPT channel
variants which use their own base dataset. The 3D trainer enforces exactly ONE active EXP_*
flag, so each non-baseline config turns the baseline's spatial-aug off and its own flag on.
Idempotent: rewrites experiments/{2D,3D}/*.json (hand-authored baseline/sgd/winning kept).
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))

def w(dim, name, cfg):
    cfg.setdefault("dim", dim)
    p = os.path.join(HERE, dim, f"{name}.json")
    json.dump(cfg, open(p, "w"), indent=2)
    print("  wrote", os.path.relpath(p, os.path.dirname(HERE)))

DS11 = {"id": 11, "name": "3D_MagPhase", "channels": "mag_phase"}
def t3(**flags):
    base = {"num_epochs": 200, "EXP_SPATIAL_AUGMENTATION": False, "EXP_SPATIAL_AUGMENTATION_ID": 12,
            "EXP_SGD_OPTIMIZER": False}
    base.update(flags)
    return base

# --- 3D: AdamW prepro/loss/channel variants on Dataset011 (one EXP flag each) ---
v3 = {
    "otsu":            ("adamw_prepro", t3(EXP_OTSU=True),                            "Dataset1719"),
    "phase_prepro":    ("adamw_prepro", t3(EXP_PHASE_PREPRO=True),                    "Dataset1720"),
    "mag_prepro":      ("adamw_prepro", t3(EXP_MAG_PREPRO=True),                      "Dataset1721"),
    "soft_loss_3":     ("adamw_loss",   t3(EXP_SOFT_EDGE_LOSS_3=True),                "Dataset1722"),
    "spatial_aug_3":   ("adamw_aug",    t3(EXP_SPATIAL_AUGMENTATION=True, EXP_SPATIAL_AUGMENTATION_ID=3), "Dataset1723"),
    "mag_one_channel": ("channels",     t3(EXP_MAG_ONE_CHANNEL=True),                "Dataset1701/1724"),
}
for name, (fam, tc, hist) in v3.items():
    w("3D", name, {"family": fam, "comment": f"historical {hist}", "dataset": DS11,
                   "configuration": "3d_fullres", "trainer": "nnUnet3DCustomTrainer",
                   "plans": "nnUNetPlans", "trainer_config": tc})

# --- 3D: channel variant needing its own (single-channel) dataset ---
w("3D", "mag_only", {"family": "channels", "comment": "historical Dataset1700 (single-channel magnitude)",
                     "dataset": {"id": 12, "name": "3D_Mag", "channels": "mag"},
                     "configuration": "3d_fullres", "trainer": "nnUnet3DCustomTrainer",
                     "plans": "nnUNetPlans", "trainer_config": t3(EXP_SPATIAL_AUGMENTATION=True)})

# --- 3D: augmentation sweep aug_5..aug_26 (one config each, one-click) ---
for aug_id in list(range(5, 27)):
    w("3D", f"aug_{aug_id}", {"family": "aug_sweep",
        "comment": f"augmentation config {aug_id} (historical Dataset{1721+aug_id})",
        "dataset": DS11, "configuration": "3d_fullres", "trainer": "nnUnet3DCustomTrainer",
        "plans": "nnUNetPlans",
        "trainer_config": t3(EXP_SPATIAL_AUGMENTATION=True, EXP_SPATIAL_AUGMENTATION_ID=aug_id)})
w("3D", "aug_no_rot_z", {"family": "aug_sweep", "comment": "historical Dataset1725 (no rot-z)",
        "dataset": DS11, "configuration": "3d_fullres", "trainer": "nnUnet3DCustomTrainer",
        "plans": "nnUNetPlans",
        "trainer_config": t3(EXP_SPATIAL_AUGMENTATION=True, EXP_SPATIAL_AUGMENTATION_ID=4)})

print("done.")
