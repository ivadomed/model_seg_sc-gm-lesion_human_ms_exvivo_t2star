#!/usr/bin/env bash
# Subject-level statistics for the 2D ablation tables (reviewer concern #2).
# For every subject-mappable (Dataset021 / id-21) 2D config:
#   1. re-predict held-out cross-validation (folds 0-3, legacy paper weights) on the CLEAN
#      Dataset021 via the current inference module (2D_workspace/inference_2D.py, validation mode),
#      using the CANONICAL split (splits/subject_split_2D.json) the paper's models were
#      trained with -- materialized with helpers/make_splits.py. Using any other split leaks.
#   2. re-score pooling voxels WITHIN each subject -> per-subject Dice/HD95 (helpers/eval_subject_stats);
#   3. paired Wilcoxon + t-test across the 12 subjects per ablation table (Holm/BH corrected).
# Runs FAST, sharded across the 4 GPUs (one config at a time per GPU). Every compute step under set_slot.
#
# NOTE: set_slot (systemd-run) does NOT propagate env vars to child processes, so DS/OUT are passed
# to the per-GPU workers as ARGUMENTS, not the environment.
#
# fold_4 does not exist anymore (was nnU-Net's remix fold); the canonical split is 4-fold.
# id-22/23/24 configs (mag_only, merge_lesions, wm_gm) have no id-21 manifest and are NOT here.
#
# SAVING LAYOUT (under $OUT, default outputs/subject_stats_2D/):
#   _dataset_canonical_split/   symlinked images/labels + the canonical splits_final.json used here
#   predictions/<name>/fold_*/predictions_nifti/   re-predicted held-out slices (+ run.log)
#   by_subject/metrics_by_subject__<name>.csv      one row per subject: dice_/hd95_ per class & region
#   stats/stats__<table>.csv                       paired variant-vs-baseline tests (Holm/BH)
#   README.txt                                     provenance of this run
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")/.."; source paths.sh
PR="$nnUNet_results/paper_results/2D"

# name | -experiment string (drives on-the-fly preprocessing!) | legacy model subpath under $PR
CONFIGS=(
  "base|base|ablations/base_aug1_hard"                       # mag+phase, Otsu, Aug1, Hard  = shared baseline
  "mag_prepro|mag_prepro|ablations/mag_clahe"          # magnitude CLAHE  (exp=mag_prepro triggers prepro)
  "phase_prepro|phase_prepro|ablations/phase_stretch"    # phase stretch    (exp=phase_prepro)
  "no_aug|no_spatial_aug|ablations/no_aug"
  "aug2|spatial_aug_2|ablations/aug2"
  "aug3|spatial_aug_3|ablations/aug3"
  "soft1|soft_loss|ablations/soft1"
  "soft2|soft_loss_2|ablations/soft2"
  "soft3|soft_loss_3|ablations/soft3"
  "sgd|sgd|ablations/sgd"
  "winning|winning|ablations/synergy_aug2_soft2"
)

do_one() {  # name exp model_subpath gpu DS OUT
  local name="$1" exp="$2" sub="$3" gpu="$4" DS="$5" OUT="$6"
  local mf="$PR/$sub/nnUNetTrainerWandb__nnUNetPlans__2d" o="$OUT/predictions/$1"
  mkdir -p "$o"
  [ -d "$mf" ] || { echo "[gpu$gpu] MISSING model $sub, skip $name"; return; }
  echo "[gpu$gpu] $name  exp=$exp  model=$sub"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_DIR" "$PY" 2D_workspace/inference_2D.py \
    -experiment "$exp" -model_folder "$mf" -mode validation -path "$DS" \
    -folds 0 1 2 3 -output_root "$o" > "$o/run.log" 2>&1 \
    || { echo "[gpu$gpu] INFER FAILED $name (see $o/run.log)"; return; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_DIR" "$PY" -m helpers.eval_subject_stats score \
    --pred-glob "$o/fold_*/predictions_nifti/*.nii.gz" \
    --labels-dir "$DS/labelsTr" --manifest "$DS/inference_manifest.json" \
    --out "$OUT/by_subject/metrics_by_subject__$name.csv" >> "$o/run.log" 2>&1 \
    || echo "[gpu$gpu] SCORE FAILED $name"
  echo "[gpu$gpu] done $name"
}

# ---- worker mode: --worker <gpu> <DS> <OUT> <spec...> (DS/OUT passed as ARGS; env not propagated) ----
if [ "${1:-}" = "--worker" ]; then
  gpu="$2"; DS="$3"; OUT="$4"; shift 4
  for spec in "$@"; do IFS='|' read -r n e s <<< "$spec"; do_one "$n" "$e" "$s" "$gpu" "$DS" "$OUT"; done
  exit 0
fi

# ---- top level ----
OUT="${OUT_DIR:-$OUTPUTS/subject_stats_2D}"
rm -rf "$OUT/predictions" "$OUT/by_subject" "$OUT/stats"   # clear stale artifacts (a partial prior run once contaminated a config)
mkdir -p "$OUT/predictions" "$OUT/by_subject" "$OUT/stats"

# Build a private dataset dir carrying the CANONICAL split (unless DS_DIR overrides it).
if [ -n "${DS_DIR:-}" ]; then
  DS="$DS_DIR"
else
  RAW="$nnUNet_raw/Dataset021_2D_MagPhase"; DS="$OUT/_dataset_canonical_split"
  mkdir -p "$DS"
  ln -sfn "$RAW/imagesTr" "$DS/imagesTr"; ln -sfn "$RAW/labelsTr" "$DS/labelsTr"
  cp -f "$RAW/dataset.json" "$RAW/inference_manifest.json" "$DS/"
  PYTHONPATH="$REPO_DIR" "$PY" -m helpers.make_splits --dataset-dir "$DS" --inject
fi
{ echo "subject-level 2D ablation stats -- generated $(date)";
  echo "dataset: $DS ; split: canonical (splits/subject_split_2D.json), the one the paper models used";
  echo "weights: legacy paper_results folds 0-3 ; unit: subject (voxels pooled within subject)";
  echo "test: paired t + Wilcoxon across 12 subjects, Holm/BH per table"; } > "$OUT/README.txt"

# shard round-robin across 4 GPUs; pass DS+OUT as ARGS
declare -a S0 S1 S2 S3
i=0; for spec in "${CONFIGS[@]}"; do case $((i%4)) in 0) S0+=("$spec");;1) S1+=("$spec");;2) S2+=("$spec");;3) S3+=("$spec");; esac; i=$((i+1)); done
[ ${#S0[@]} -gt 0 ] && set_slot 0 bash "$SELF" --worker 0 "$DS" "$OUT" "${S0[@]}" &
[ ${#S1[@]} -gt 0 ] && set_slot 1 bash "$SELF" --worker 1 "$DS" "$OUT" "${S1[@]}" &
[ ${#S2[@]} -gt 0 ] && set_slot 2 bash "$SELF" --worker 2 "$DS" "$OUT" "${S2[@]}" &
[ ${#S3[@]} -gt 0 ] && set_slot 3 bash "$SELF" --worker 3 "$DS" "$OUT" "${S3[@]}" &
wait
echo "=== all re-predictions + per-subject scoring done -> $OUT/by_subject ==="

# ---- per-table paired statistics (baseline = 'base') ----
B="base=$OUT/by_subject/metrics_by_subject__base.csv"
run_stats() { local tbl="$1"; shift; local vs=(); for n in "$@"; do vs+=("$n=$OUT/by_subject/metrics_by_subject__$n.csv"); done
  set_slot 0 env PYTHONPATH="$REPO_DIR" "$PY" -m helpers.eval_subject_stats paired --baseline "$B" --variants "${vs[@]}" --out "$OUT/stats/stats__$tbl.csv"; }
run_stats preprocessing mag_prepro phase_prepro
run_stats augmentation  no_aug aug2 aug3
run_stats soft_labels   soft1 soft2 soft3
run_stats optimizer     sgd
run_stats winning       winning
echo "=== stats -> $OUT/stats ; per-subject -> $OUT/by_subject ==="
