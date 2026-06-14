# model_seg_sc-gm-lesion_human_ms_exvivo_t2star

Segmentation models for spinal cord white matter (WM), gray matter (GM), and MS lesions from high-resolution ex vivo GRE magnitude + phase MRI.

## Status

- **Final publication model**: 3D nnU-Net model (recommended for reported results and external use).
- **Companion model**: 2D nnU-Net model (provided for comparison/ablation and legacy workflows).

## Repository structure

```
model_seg_.../                 ← this repo (code). Data lives OUTSIDE, at the project root (see paths.sh).
  paths.sh                     central path config: PROJECT_ROOT, ms-exvivo-nih, nnUNet_data, outputs, .venv
  install_trainers.sh          sync custom trainers -> venv (run once after setup / after editing a trainer)
  run_experiment_training_2D.sh  run_experiment_training_3D.sh
  run_experiment_inference_2D.sh run_experiment_inference_3D.sh
  make_3d_pseudo_gt.sh         opt-in: regenerate weakly-supervised 3D labels (2D model -> stack)
  experiments/                 ALL experiment definitions
    2D/<name>.json  3D/<name>.json     one JSON per experiment (dataset + trainer config)
    experiments.tsv            registry of every experiment (auto-generated)
    load_config.py  make_patch_plans.py  gen_configs*.py
  2D_workspace/  { dataset_prep/ (build_dataset_2d + slicing/convert deps),  inference/inference_2.py,  trainers/(+utils) }
  3D_workspace/  { dataset_prep/build_dataset_3d.py,  inference/inference_3d.py,  trainers/(+utils) }
  helpers/                     shared utils (metrics, preprocessing, stats, viz)
  inference_publication/       standalone published-model inference entrypoints
```

Data (outside the repo, at `PROJECT_ROOT` = the repo's parent dir): `ms-exvivo-nih/` (clean source of
truth), `nnUNet_data/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results}`, `outputs/`, `.venv/`.
Historical/published models live in `nnUNet_data/nnUNet_results/paper_results/{2D|3D}/{family}/{exp}/`.

## How to run (everything is one command under `set_slot`)

> All compute commands MUST be wrapped in `set_slot 3 …` (cluster GPU/CPU/RAM allocation).

```bash
# one-time: sync the custom trainers into the venv
set_slot 3 bash install_trainers.sh

# TRAIN any experiment (builds dataset + preprocesses on first use; injects our 4-fold split)
set_slot 3 bash run_experiment_training_2D.sh <exp> [fold]      # e.g. winning 0
set_slot 3 bash run_experiment_training_3D.sh <exp> [fold]      # e.g. adamw_baseline 0
#   quick smoke (2 epochs):   NNUNET_NUM_EPOCHS=2 set_slot 3 bash run_experiment_training_3D.sh <exp> 0
#   enable wandb logging:     WANDB_MODE=online    set_slot 3 bash run_experiment_training_2D.sh <exp> 0

# INFER any experiment  (--tta / --single-fold / --legacy are optional)
set_slot 3 bash run_experiment_inference_2D.sh <exp> <validation|test|volume> [input] [--tta] [--single-fold] [--legacy]
set_slot 3 bash run_experiment_inference_3D.sh <exp> <nnunet_input_dir>        [--no-tta] [--single-fold]
#   default = ensemble of folds 0-3, TTA on (3D) / off (2D); flags flip those.
#   --legacy (2D) uses the published model in paper_results instead of a freshly trained one.
```

Inference writes to `outputs/experiments/{family}/{exp}/<mode>_<ensemble|fold0>[_tta]/`. Trained
models go to `nnUNet_data/nnUNet_results/{2D|3D}/{family}/{exp}/`.

### Metrics (unified across 2D and 3D)

`helpers/eval.py` is the single, shared scorer. Pass `--gt <dir>` to an inference runner (or call it
standalone) to score predictions against ground-truth NIfTIs into one uniform schema:
- `metrics_casewise.csv`  — one row per case: `dice_<name>` and `hd95_<name>` for each class
  (`WM/GM/lesion_WM/lesion_GM`) and region (`cord/WM/GM/lesion`)
- `metrics_summary.csv`   — per-metric mean/std/count
- `metrics_global_dice.csv` — dataset-level (aggregate-count) Dice
- `metrics_crossfold.csv` — mean/std across folds (`python -m helpers.eval --crossfold <parent>`)

```bash
set_slot 3 bash run_experiment_inference_3D.sh <exp> <input_dir> --gt /path/to/gt_segmentations
python -m helpers.eval --pred-dir P --gt-dir G          # standalone, on any predictions
```
Both pipelines emit the **same** `metrics_casewise.csv` schema (the 2D runner scores its saved
predictions through `eval.py` too; `inference_2.py` additionally keeps its fine-grained per-slice
metrics). Metric computation is fully **separated from inference** — it runs on already-saved
predictions, so you never re-run a model to (re)compute metrics.

### Statistical comparison of two methods

`eval.py --compare` does a **paired, per-subject** comparison between two methods' `metrics_casewise.csv`
(the right unit: with subject-level k-fold CV each subject is held out once → one metric per method;
pairing across subjects gives N = #subjects, far more powerful than the 4 folds). For every metric
(`dice_*`, `hd95_*`, tested separately) it reports the **paired Student t-test** and the **Wilcoxon
signed-rank test** (robust to the non-normal, bounded Dice/HD95).

```bash
python -m helpers.eval --compare A/metrics_casewise.csv B/metrics_casewise.csv --names 3D 2D --out cmp.csv
```

## First-time setup

```bash
# 1. data layout: the repo (code) sits next to the data at the PROJECT_ROOT (the repo's parent dir):
#      <PROJECT_ROOT>/ms-exvivo-nih/      (clean dataset, git-annex)
#      <PROJECT_ROOT>/.venv/              (python env)
#      <PROJECT_ROOT>/nnUNet_data/ , outputs/   (created automatically)
#    paths.sh resolves these; override with e.g. `PROJECT_ROOT=/data ... ` or edit paths.sh.
python -m venv ../.venv && source ../.venv/bin/activate && pip install -r requirements.txt
set_slot 3 bash install_trainers.sh          # sync custom trainers into the venv
set_slot 3 bash tests/run_smoke.sh           # 2-epoch end-to-end check (train+infer+metrics, 2D & 3D)
```

## Defining / re-running an experiment

An experiment is **one JSON** in `experiments/{2D,3D}/`. The dataset rarely changes — most experiments
only vary the **trainer config**; the base nnUNet dataset is reused (one dataset, many experiments).

```jsonc
{
  "dim": "3D",
  "family": "adamw_baseline",                 // groups results: nnUNet_results/3D/<family>/<exp>/
  "dataset": {"id": 11, "name": "3D_MagPhase", "channels": "mag_phase"},  // label_mode for 2D
  "configuration": "3d_fullres",
  "trainer": "nnUnet3DCustomTrainer",
  "plans": "nnUNetPlans",
  "patch_size": [192,64,208],                 // optional: collapsed patch sweep (reuses preprocessing)
  "trainer_config": {"num_epochs": 200, "EXP_SPATIAL_AUGMENTATION": true, "EXP_SPATIAL_AUGMENTATION_ID": 12}
}
```

- **Re-run an existing experiment**: `set_slot 3 bash run_experiment_training_<dim>.sh <exp> <fold>` — it
  reuses the built dataset/preprocessing and writes to the experiment's own results folder.
- **Add an experiment**: drop a new JSON in `experiments/<dim>/` (copy the closest one, change
  `trainer_config` / `dataset` / `patch_size`). `trainer_config` keys map to the trainer's `EXP_*`
  (3D) / `exp_*` (2D) flags; see `experiments.tsv` for the full catalog and historical mapping.

## Citation (Placeholder)

If you use this repository, please cite:

```bibtex
@article{PLACEHOLDER_202X,
	title   = {PLACEHOLDER_TITLE},
	author  = {PLACEHOLDER_AUTHORS},
	journal = {PLACEHOLDER_JOURNAL},
	year    = {202X},
	doi     = {PLACEHOLDER_DOI}
}
```

## Data Access (Placeholder)

- **Dataset statement**: `PLACEHOLDER_DATASET_STATEMENT`
- **Access policy**: `PLACEHOLDER_ACCESS_POLICY`
- **Download link/request contact**: `PLACEHOLDER_LINK_OR_EMAIL`

## Environment and Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Expected input format for inference

Both publication inference scripts expect nnU-Net-style input folders:

- One folder containing NIfTI files in `imagesTs` style.
- Channel suffixes for multichannel inputs, e.g.:
	- `CASE001_0000.nii.gz` (magnitude)
	- `CASE001_0001.nii.gz` (phase)

## Publication Inference (new standalone scripts)

The publication scripts are in `inference_publication/` and are intentionally separate from tailored internal scripts.

### 3D inference (final model)

Edit placeholders in:

- `inference_publication/run_infer_3d_public.sh`

Then run:

```bash
bash inference_publication/run_infer_3d_public.sh
```

Direct Python usage:

```bash
python inference_publication/infer_3d_public.py \
	--input-dir /path/to/nnunet_inputs_3d/imagesTs \
	--output-dir /path/to/output/predictions_3d \
	--model-folder /path/to/nnUNet_results/DatasetXXXX_<TASK_NAME>/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres \
	--folds 0 1 2 3 \
	--checkpoint checkpoint_final.pth \
	--gpu-id 0 \
	--overwrite
```

### 2D inference (companion model)

Edit placeholders in:

- `inference_publication/run_infer_2d_public.sh`

Then run:

```bash
bash inference_publication/run_infer_2d_public.sh
```

Direct Python usage:

```bash
python inference_publication/infer_2d_public.py \
	--input-dir /path/to/nnunet_inputs_2d/imagesTs \
	--output-dir /path/to/output/predictions_2d \
	--model-folder /path/to/nnUNet_results/DatasetXXXX_<TASK_NAME>/nnUNetTrainerWandb__nnUNetPlans__2d \
	--folds 0 1 2 3 \
	--checkpoint checkpoint_final.pth \
	--gpu-id 0 \
	--overwrite
```

## Pretrained Weights (Placeholder)

- **Release page**: `PLACEHOLDER_RELEASE_URL`
- **3D final model folder name**: `PLACEHOLDER_3D_MODEL_FOLDER`
- **2D model folder name**: `PLACEHOLDER_2D_MODEL_FOLDER`

## Repository Layout (relevant to publication)

- `inference_publication/`: clean, standalone publication inference entrypoints.
- `3D_workspace/`: full 3D training/inference research workspace.
- `2D_workspace/`: full 2D training/inference research workspace.
- `helpers/`: shared utility modules.

## Notes

- Existing tailored inference scripts in `2D_workspace/` and `3D_workspace/` are preserved as-is.
- Publication users should prefer the scripts in `inference_publication/`.
