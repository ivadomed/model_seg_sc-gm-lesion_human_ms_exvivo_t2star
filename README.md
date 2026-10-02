# model_seg_sc-gm-lesion_human_ms_exvivo_t2star

Segmentation of spinal cord **white matter**, **gray matter**, and **MS lesions** from high-resolution
**ex vivo** GRE magnitude + phase MRI (nnU-Net based).

- **3D model** — recommended, reported results.
- **2D model** — companion/ablation model.

## Setup

```bash
git clone git@github.com:ivadomed/model_seg_sc-gm-lesion_human_ms_exvivo_t2star.git
cd model_seg_sc-gm-lesion_human_ms_exvivo_t2star
python -m venv ../.venv && source ../.venv/bin/activate && pip install -r requirements.txt
bash install_trainers.sh   # registers the custom nnU-Net trainers in the venv
```

Data and run outputs live next to the repo, not inside it (`../ms-exvivo-nih`, `../nnUNet_data`,
`../outputs`), auto-detected by `paths.sh`. Override with `PROJECT_ROOT=/path/to/parent ...`.

### Get the data (lab members only, for now)

```bash
git clone git@data.neuro.polymtl.ca:datasets/ms-exvivo-nih ../ms-exvivo-nih
cd ../ms-exvivo-nih && git annex get . && cd -
```
~41 GB, git-annex. Ask for access to the lab's data server first.

## Run inference with the released model

```bash
# download and unzip a released model from https://github.com/ivadomed/model_seg_sc-gm-lesion_human_ms_exvivo_t2star/releases
bash inference_publication/run_infer_3d_public.sh <input_dir> <output_dir> <model_folder>
```

`<input_dir>` is an nnU-Net `imagesTs`-style folder: `CASE_0000.nii.gz` (magnitude) [+ `CASE_0001.nii.gz` (phase)].
Add `--tta` for test-time augmentation; override GPU/checkpoint with `GPU_ID=1 CHECKPOINT=checkpoint_final.pth`.
The 2D companion model works the same way via `run_infer_2d_public.sh`.

## Reproduce the paper's experiments

Every experiment is one JSON file under `2D_workspace/experiments/` or `3D_workspace/experiments/`
(dataset + trainer config). `reproduce/` holds the scripts that run them — each builds/preprocesses
its dataset on first use and injects the paper's 4-fold subject split (`splits/`):

```bash
bash reproduce/run_experiment_training_3D.sh adamw_baseline 0        # train one fold
bash reproduce/run_experiment_inference_3D.sh adamw_baseline <input_dir> --gt <gt_dir>   # predict + score
```

Same pattern for 2D (`run_experiment_training_2D.sh` / `run_experiment_inference_2D.sh`). Metrics are
written as `metrics_casewise.csv` / `metrics_summary.csv` by `helpers/eval.py`, shared by both pipelines.
`reproduce/run_subject_stats_campaign_{2D,3D}.sh` reproduces the paper's per-subject statistical tables.

To add an experiment, copy the closest JSON and edit `trainer_config` / `dataset`.

Sanity-check the whole pipeline end to end (2-epoch smoke run): `bash tests/run_smoke.sh`.

## Repository layout

```
paths.sh, install_trainers.sh      setup
reproduce/                          scripts that (re)train, (re)predict, and re-score the paper's experiments
inference_publication/              standalone inference with the released model
2D_workspace/, 3D_workspace/        dataset building + nnU-Net trainers, each with its own experiments/
helpers/                            shared code: metrics, config loading, splits, stats
splits/                             the paper's 4-fold subject splits
tests/                              smoke test
doc/                                figures
```
