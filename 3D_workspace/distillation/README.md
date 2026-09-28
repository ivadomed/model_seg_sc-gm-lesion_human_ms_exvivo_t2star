# Offline Knowledge Distillation

Distill the winning 3D teacher (4-fold, `paper_results/3D/ablations/base3d/...`) into a smaller
3D nnU-Net student, using the teacher's cached logits as soft targets.

```
trainer/    the nnU-Net trainer plugin (copied flat into the venv by 01_install_trainer.sh)
scripts/    everything you run, in pipeline order
configs/    one JSON per training condition
```

## 1. Cache the teacher's soft targets (once)

```bash
python scripts/01_prepare/01_precompute_teacher_logits.py
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
