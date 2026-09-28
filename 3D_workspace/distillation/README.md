# Offline Knowledge Distillation

Distill the winning 3D teacher (4-fold, `paper_results/3D/ablations/base3d/...`) into a smaller
3D nnU-Net student, using the teacher's cached logits as soft targets.

```
trainer/    the nnU-Net trainer plugin (copied flat into the venv by 01_install_trainer.sh)
scripts/    everything you run, in pipeline order
configs/    one JSON per training condition
```

## 0. Setup (fresh machine)

Everything below runs under `set_slot <N>` (this lab's cluster GPU/CPU/RAM reservation tool). If
your machine doesn't have `set_slot`, drop that prefix and run the command directly.

**a) Code + Python env** (from the repo root, one level up from `3D_workspace/`):
```bash
python -m venv ../.venv && source ../.venv/bin/activate && pip install -r requirements.txt
set_slot 3 bash install_trainers.sh   # base 3D trainer -> venv (required before step (c) below)
```
Needs Python 3.12 and `git-annex` installed on the machine.

**b) Data** (git-annex; ask for access to the lab's data server first):
```bash
git clone git@data.neuro.polymtl.ca:datasets/ms-exvivo-nih ../ms-exvivo-nih
cd ../ms-exvivo-nih && git annex get . && cd -
```
`paths.sh` expects this at `../ms-exvivo-nih` relative to the repo (override with `CLEAN_DATASET=...`).
~41 GB.

**c) The teacher.** The distillation scripts default to reusing an *already-trained* teacher at
`nnUNet_data/nnUNet_results/paper_results/3D/ablations/base3d/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres/`
— that's a frozen archive copy, not something any command regenerates for you. You have two options:

- **Recommended: download the published weights** from the repo's
  [GitHub release](https://github.com/ivadomed/model_seg_sc-gm-lesion_human_ms_exvivo_t2star/releases/tag/v1.0.0)
  (`Dataset1718_MagPhase_patchsize_5_adamw.zip`, ~1.3 GB, verified byte-identical to the checkpoints
  these distillation results were produced with):
  ```bash
  curl -LO https://github.com/ivadomed/model_seg_sc-gm-lesion_human_ms_exvivo_t2star/releases/download/v1.0.0/Dataset1718_MagPhase_patchsize_5_adamw.zip
  unzip Dataset1718_MagPhase_patchsize_5_adamw.zip
  mkdir -p ../nnUNet_data/nnUNet_results/paper_results/3D/ablations/base3d
  mv Dataset1718_MagPhase_patchsize_5_adamw/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres \
     ../nnUNet_data/nnUNet_results/paper_results/3D/ablations/base3d/
  ```
  That matches the distillation scripts' default `--teacher` path exactly — no override needed.

  The release only has the trained weights, not the *preprocessed training data* the distillation
  logits get cached against — you still need (b) done, plus a quick preprocessing pass (no need to
  actually train): `NNUNET_NUM_EPOCHS=1 set_slot 3 bash run_experiment_training_3D.sh adamw_baseline 0`
  builds + preprocesses Dataset011 and generates the patch-192x64x208 plans in a few minutes; the
  1-epoch fold it trains is throwaway — the released weights are your real teacher, not that checkpoint.

- **Alternative: retrain it yourself** (needs (a) and (b) done first; ~10h/fold on a single GPU):
  ```bash
  for fold in 0 1 2 3; do
    set_slot 3 bash run_experiment_training_3D.sh adamw_baseline $fold
  done
  ```
  This lands at a **different path** than the release download:
  `nnUNet_data/nnUNet_results/3D/adamw_baseline/adamw_baseline/Dataset011_3D_MagPhase/nnUnet3DCustomTrainer__nnUNetPlans_p192x64x208__3d_fullres/`.
  Pass `--teacher <that path>` in step 1 below (the script's default won't match).

## 1. Cache the teacher's soft targets (once)

```bash
python scripts/01_prepare/01_precompute_teacher_logits.py   # add --teacher <path> if you retrained (see 0c)
```

Writes one float16 logits `.b2nd` per training case to
`nnUNet_preprocessed/Dataset011_3D_MagPhase/teacher_logits__patchsize_5_adamw/`.

## 2. Derive the student's plans (once)

```bash
python scripts/01_prepare/02_make_student_plans.py --width-scale 0.5
```

Writes `nnUNetPlansDistillS.json` next to the teacher's plans (same patch/spacing, smaller network).

## 3. Train

```bash
bash scripts/02_train/01_install_trainer.sh                          # once
bash scripts/02_train/02_run_distill.sh 0 configs/distill.json       # one fold
bash scripts/02_train/03_run_all_conditions.sh                       # the full matrix
```

`DISTILL_TAG=<name>` tags where results land: `nnUNet_results/distillation/<name>/...`. Give
every condition its own tag, or runs overwrite each other.

Sweep KD temperature with an env var (the config defaults to T=2):

```bash
KD_TEMPERATURE=1.0 DISTILL_TAG=kd_T1 bash scripts/02_train/02_run_distill.sh 0 configs/distill.json
```

Smoke-test both code paths fast (2 epochs x 5 iters):

```bash
bash scripts/02_train/02_run_distill.sh 0 configs/smoke.json               # KD path
KD_WEIGHT=0 bash scripts/02_train/02_run_distill.sh 0 configs/smoke.json   # control path
```

## 4. Evaluate

```bash
bash scripts/03_evaluate/01_eval_teacher.sh
bash scripts/03_evaluate/02_eval_all.sh
```

Writes `metrics_casewise.csv` / `metrics_summary.csv` next to each prediction folder.

## 5. Benchmark latency

```bash
python scripts/04_benchmark/benchmark_latency.py --device cuda
python scripts/04_benchmark/benchmark_latency.py --device cpu
```

## Configs

| File | Condition |
|---|---|
| `configs/control.json` | no KD, supervised only |
| `configs/distill.json` | KD, T=2 (override `KD_TEMPERATURE` for a sweep) |
| `configs/smoke.json` | fast wiring check, 2 epochs |
