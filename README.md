# model_seg_sc-gm-lesion_human_ms_exvivo_t2star

Segmentation models for spinal cord **white matter (WM)**, **gray matter (GM)**, and **MS lesions** from
high-resolution **ex vivo** GRE magnitude + phase MRI (nnU-Net based).

- **Final / recommended model**: 3D nnU-Net (reported results, external use).
- **Companion model**: 2D nnU-Net (comparison/ablation; the weakly-supervised slice model).

---

## Repository structure

```
model_seg_.../                       ← this repo (CODE only). Data lives OUTSIDE, at the project root.
  paths.sh                           central path config (sourced by every runner): PROJECT_ROOT, PY,
                                     nnUNet_raw/preprocessed/results, CLEAN_DATASET (ms-exvivo-nih), OUTPUTS
  install_trainers.sh                sync the custom nnU-Net trainers into the venv
  run_experiment_training_2D.sh   run_experiment_training_3D.sh
  run_experiment_inference_2D.sh  run_experiment_inference_3D.sh
  make_3d_pseudo_gt.sh               opt-in: regenerate weakly-supervised 3D labels (2D model -> stack)
  experiments/                       ALL experiment definitions (central registry)
    2D/<name>.json   3D/<name>.json  one JSON per experiment (dataset + trainer config)
    experiments.tsv                  catalog of every experiment (auto-generated)
    load_config.py  make_patch_plans.py  gen_configs*.py
  2D_workspace/   { dataset_prep/ (build_dataset_2d.py + slicing/convert deps),  inference_2D.py,  trainers/(+utils) }
  3D_workspace/   { dataset_prep/build_dataset_3d.py,  trainers/(+utils) }     # 3D inference = nnUNetv2_predict (see runner)
  helpers/                           shared utils: eval.py (metrics), metric_utils_2d, stats_utils, preprocessing, viz
  inference_publication/             standalone, arg-driven inference for the RELEASED model
  tests/run_smoke.sh                 fast 2-epoch end-to-end check (train + infer + metrics, 2D & 3D)
  doc/                               figures
```

The repo holds only code. **Data lives at `PROJECT_ROOT` (the repo's parent dir)** and `paths.sh` resolves it:
`ms-exvivo-nih/` (clean source of truth, git-annex), `nnUNet_data/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results}`,
`outputs/`, `.venv/`. Historical/published models live in `nnUNet_data/nnUNet_results/paper_results/{2D|3D}/{family}/{exp}/`.

> 2D needs a custom inference module (`inference_2D.py`: slice → predict → reconstruct the volume). 3D is
> volume-native and uses standard `nnUNetv2_predict` (wrapped by the runner), so it has no inference module.

---

## First-time setup

```bash
# Code (this repo) sits next to the data at PROJECT_ROOT (= the repo's parent dir):
#   <PROJECT_ROOT>/ms-exvivo-nih/                  clean dataset (git-annex)
#   <PROJECT_ROOT>/.venv/                          python env
#   <PROJECT_ROOT>/nnUNet_data/ , outputs/         created automatically
# paths.sh auto-resolves these; override with `PROJECT_ROOT=/path ...` or by editing paths.sh.
python -m venv ../.venv && source ../.venv/bin/activate && pip install -r requirements.txt
set_slot 3 bash install_trainers.sh        # sync custom trainers into the venv
set_slot 3 bash tests/run_smoke.sh         # 2-epoch end-to-end check (asserts checkpoints + metrics)
```

> All compute commands must run under `set_slot 3 …` (cluster GPU/CPU/RAM allocation).

---

## How to run

```bash
# TRAIN any experiment  (builds the dataset + preprocesses on first use; injects our 4-fold split)
set_slot 3 bash run_experiment_training_2D.sh <exp> [fold]        # e.g. winning 0
set_slot 3 bash run_experiment_training_3D.sh <exp> [fold]        # e.g. adamw_baseline 0
#   quick smoke (2 epochs):  NNUNET_NUM_EPOCHS=2 set_slot 3 bash run_experiment_training_3D.sh <exp> 0
#   enable wandb logging:    WANDB_MODE=online    set_slot 3 bash run_experiment_training_2D.sh <exp> 0

# INFER any experiment
set_slot 3 bash run_experiment_inference_2D.sh <exp> <validation|test|volume> [input] [--tta] [--single-fold] [--legacy]
set_slot 3 bash run_experiment_inference_3D.sh <exp> <nnunet_input_dir> [--no-tta] [--single-fold] [--gt <gt_dir>]
#   default = ensemble of folds 0-3 (TTA on for 3D, off for 2D); the flags flip those.
#   --legacy (2D): use the published model in paper_results instead of a freshly trained one.
#   --gt (3D): also score predictions vs ground truth (unified metrics, see below).
```

Trained models → `nnUNet_data/nnUNet_results/{2D|3D}/{family}/{exp}/`.
Inference outputs → `outputs/experiments/{family}/{exp}/<mode>_<ensemble|fold0>[_tta]/`.

### Metrics (unified across 2D and 3D)

`helpers/eval.py` is the single shared scorer; **metric computation is separated from inference** (it scores
already-saved predictions — you never re-run a model to recompute metrics). Both pipelines emit the same schema:

- `metrics_casewise.csv` — per case: `dice_<name>` & `hd95_<name>` for each class (`WM/GM/lesion_WM/lesion_GM`) and region (`cord/WM/GM/lesion`)
- `metrics_summary.csv` — per-metric mean/std/count · `metrics_global_dice.csv` — dataset-level Dice
- `metrics_crossfold.csv` — `python -m helpers.eval --crossfold <parent_dir>`

```bash
set_slot 3 bash run_experiment_inference_3D.sh <exp> <input_dir> --gt /path/to/gt_segmentations
python -m helpers.eval --pred-dir P --gt-dir G              # standalone, on any predictions
```

### Statistical comparison of two methods

**Paired, per-subject** comparison (the correct unit: with subject-level k-fold CV each subject is held out
once → one metric per method → pair across subjects, N = #subjects, not the 4 folds). Per metric (`dice_*`,
`hd95_*` separately): **paired Student t-test** + **Wilcoxon signed-rank** (robust to non-normal/bounded Dice & HD95).

```bash
python -m helpers.eval --compare A/metrics_casewise.csv B/metrics_casewise.csv --names 3D 2D --out cmp.csv
```

---

## Defining / re-running an experiment

An experiment is **one JSON** in `experiments/{2D,3D}/`. The dataset rarely changes — most experiments vary
only the **trainer config**; the base nnU-Net dataset is reused (one dataset, many experiments).

```jsonc
{
  "dim": "3D",
  "family": "adamw_baseline",                 // groups results: nnUNet_results/3D/<family>/<exp>/
  "dataset": {"id": 11, "name": "3D_MagPhase", "channels": "mag_phase"},   // 2D also takes "label_mode"
  "configuration": "3d_fullres",
  "trainer": "nnUnet3DCustomTrainer",
  "plans": "nnUNetPlans",
  "patch_size": [192, 64, 208],               // optional: collapsed patch sweep (reuses preprocessing)
  "trainer_config": {"num_epochs": 200, "EXP_SPATIAL_AUGMENTATION": true, "EXP_SPATIAL_AUGMENTATION_ID": 12}
}
```

- **Re-run**: `set_slot 3 bash run_experiment_training_<dim>.sh <exp> <fold>` — reuses the built dataset/preprocessing.
- **Add**: copy the closest JSON, change `trainer_config`/`dataset`/`patch_size`. `trainer_config` keys map to the
  trainer's `EXP_*` (3D) / `exp_*` (2D) flags; see `experiments.tsv` for the full catalog and historical mapping.

---

## Publication inference (released model)

Standalone, arg-driven entrypoints in `inference_publication/` for external users who have only the released
weights (no `experiments/` configs). Inputs are nnU-Net `imagesTs`-style folders (`CASE_0000.nii.gz` magnitude
[+ `CASE_0001.nii.gz` phase]).

```bash
# 3D (recommended model) — no editing, just pass paths:
bash inference_publication/run_infer_3d_public.sh <input_dir> <output_dir> <model_folder>
# 2D (companion model):
bash inference_publication/run_infer_2d_public.sh <input_dir> <output_dir> <model_folder>
#   override checkpoint/gpu via env (CHECKPOINT=..., GPU_ID=...); pass extra args like --tta after the 3 paths.
# Or call the Python entrypoint directly:
python inference_publication/infer_3d_public.py --input-dir IN --output-dir OUT --model-folder MODEL --folds 0 1 2 3 --overwrite
```

---

## Citation (placeholder)

```bibtex
@article{PLACEHOLDER_202X,
  title   = {PLACEHOLDER_TITLE},
  author  = {PLACEHOLDER_AUTHORS},
  journal = {PLACEHOLDER_JOURNAL},
  year    = {202X},
  doi     = {PLACEHOLDER_DOI}
}
```

## Data access & weights (placeholder)

- **Dataset**: `ms-exvivo-nih` — `PLACEHOLDER_DATASET_STATEMENT` / access: `PLACEHOLDER_ACCESS_POLICY` / `PLACEHOLDER_LINK_OR_EMAIL`
- **Released weights**: `PLACEHOLDER_RELEASE_URL` (3D: `PLACEHOLDER_3D_MODEL_FOLDER`, 2D: `PLACEHOLDER_2D_MODEL_FOLDER`)
