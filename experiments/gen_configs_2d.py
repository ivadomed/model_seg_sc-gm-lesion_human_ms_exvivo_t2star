#!/usr/bin/env python3
"""Generate 2D family configs from the EXACT historical exp_* flags (recovered from keep-dir
debug.json). Trainer-config experiments share Dataset021 (mag+phase, label-mode all); channel/
label-scheme experiments use their own base dataset. Hand-authored winning.json is left as-is."""
import json, os
D = os.path.join(os.path.dirname(__file__), "2D")
DS21 = {"id": 21, "name": "2D_MagPhase", "channels": "mag_phase", "label_mode": "all"}

# edge_params variants as defined in nnUNetTrainerWandb.on_train_start (default == variant 2)
EP1 = {"1": {"edge_weight": 0.9, "kernel_size": 7}, "2": {"edge_weight": 0.9, "kernel_size": 3},
       "3": {"edge_weight": 0.6, "kernel_size": 5}, "4": {"edge_weight": 0.4, "kernel_size": 7}}
EP3 = {"1": {"edge_weight": 0.7, "kernel_size": 5}, "2": {"edge_weight": 0.6, "kernel_size": 3},
       "3": {"edge_weight": 0.2, "kernel_size": 3}, "4": {"edge_weight": 0.2, "kernel_size": 3}}

def w(name, fam, flags, hist, dataset=DS21, extra=None):
    tc = {"num_epochs": 1000}
    for f in flags:
        tc[f] = True
    if extra:
        tc.update(extra)
    json.dump({"dim": "2D", "family": fam, "comment": f"historical {hist}", "dataset": dataset,
               "configuration": "2d", "trainer": "nnUNetTrainerWandb", "plans": "nnUNetPlans",
               "trainer_config": tc}, open(f"{D}/{name}.json", "w"), indent=2)
    print(f"  {name:18} flags={flags} ds={dataset['id']}")

# --- trainer-config families on Dataset021 (exact single flag from debug.json) ---
w("base",           "base_prepro", ["exp_base"],            "Dataset700")
w("base_no_otsu",   "base_prepro", ["exp_base_no_otsu"],    "Dataset708")
w("prepro_mag",     "base_prepro", ["exp_mag_only"],        "Dataset703")
w("phase_prepro",   "base_prepro", ["exp_phase_prepro"],    "Dataset704")
w("mag_prepro",     "base_prepro", ["exp_mag_prepro"],      "Dataset705")
w("soft_loss",      "loss",        ["exp_soft_loss"],       "Dataset701/711 (default edge_params)")
w("soft_loss_1",    "loss",        ["exp_soft_loss"],       "Dataset soft_loss variant 1", extra={"edge_params": EP1})
w("soft_loss_3",    "loss",        ["exp_soft_loss"],       "Dataset713 (edge_params v3)", extra={"edge_params": EP3})
w("mosaic",         "aug",         ["exp_mosaic"],          "Dataset702")
w("no_spatial_aug", "aug",         ["exp_no_spatial_aug"],  "Dataset706")
w("spatial_aug_3",  "aug",         ["exp_spatial_aug_3"],   "Dataset715")
w("sgd",            "optimizer",   ["exp_SGD"],             "Dataset716")

# --- data-defining families (own base dataset) ---
w("mag_one_channel", "channels",   ["exp_mag_one_channel"], "Dataset710",
  dataset={"id": 22, "name": "2D_Mag1ch", "channels": "mag-one-channel", "label_mode": "all"})
w("merge_lesions",   "label_scheme", ["exp_base"],          "Dataset707",
  dataset={"id": 23, "name": "2D_MergedLesion", "channels": "mag_phase", "label_mode": "merged_lesion"})
w("wm_gm",           "label_scheme", ["exp_base"],          "Dataset709",
  dataset={"id": 24, "name": "2D_Tissues", "channels": "mag_phase", "label_mode": "tissues"})
print("done.")
