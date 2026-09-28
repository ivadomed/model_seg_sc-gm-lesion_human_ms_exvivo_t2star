#!/usr/bin/env bash
# Subject-level statistics for the 3D ablation tables (reviewer concern #2), WITHOUT re-prediction.
# The paper's 3D held-out predictions are already stored (predictions_3d volumes, subject-mappable via
# the Dataset011 manifest, each in its held-out fold). We re-score them on the manually annotated axial
# slices, pooling voxels WITHIN each subject (helpers/eval_subject_stats_3d), then run the paired
# Wilcoxon/t-test across the 12 subjects per ablation table (same stats step as 2D).
# No GPU needed (loads stored volumes); parallelized across slots for I/O. Every step under set_slot.
#
# SAVING LAYOUT (under $OUT, default outputs/subject_stats_3D/):
#   by_subject/metrics_by_subject__<name>.csv   per-subject dice/hd95 (same schema as 2D)
#   verify/<name>.log                           per-config global (paper aggregation) for the repro check
#   stats/stats__<table>.csv                    paired variant-vs-base3d tests (Holm/BH)
set -euo pipefail
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$0")"; source paths.sh
P3="$nnUNet_results/paper_results/3D"
TR="nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"

# name | family/exp under paper_results/3D  (base3d = winning 3D = the shared baseline)
CONFIGS=(
  "base3d|ablations/base3d"
  "mag|ablations/mag_only"
  "mag_prepro|ablations/mag_clahe"
  "phase_prepro|ablations/phase_stretch"
  "aug1|ablations/aug1"
  "aug2|ablations/aug2"
  "aug3|ablations/aug3"
  "soft1|ablations/soft1"
  "soft2|ablations/soft2"
  "soft3|ablations/soft3"
  "otsu|ablations/otsu"
)

score_one() {  # name familyexp OUT
  local name="$1" fe="$2" OUT="$3"
  local base="$P3/$fe/$TR/inference_results"
  local ed; ed="$(find "$base" -maxdepth 1 -type d -name "*_validation_EVAL" 2>/dev/null | head -1)"
  [ -n "$ed" ] || { echo "[3d] MISSING validation_EVAL for $name ($fe)"; return; }
  echo "[3d] scoring $name  <- ${ed#$P3/}"
  PYTHONPATH="$REPO_DIR" PROJECT_ROOT="$PROJECT_ROOT" "$PY" -m helpers.eval_subject_stats_3d score \
    --eval-dir "$ed" --out "$OUT/by_subject/metrics_by_subject__$name.csv" --verify \
    > "$OUT/verify/$name.log" 2>&1 || echo "[3d] SCORE FAILED $name"
  echo "[3d] done $name"
}

if [ "${1:-}" = "--worker" ]; then
  OUT="$2"; shift 2
  for spec in "$@"; do IFS='|' read -r n fe <<< "$spec"; score_one "$n" "$fe" "$OUT"; done
  exit 0
fi

OUT="${OUT_DIR:-$OUTPUTS/subject_stats_3D}"
rm -rf "$OUT/by_subject" "$OUT/verify" "$OUT/stats"   # clear stale artifacts before a re-run
mkdir -p "$OUT/by_subject" "$OUT/verify" "$OUT/stats"
{ echo "subject-level 3D ablation stats -- generated $(date)";
  echo "stored held-out predictions_3d re-scored on manual annotated slices; unit: subject";
  echo "baseline: base3d (winning 3D = ablations/base3d); paired t + Wilcoxon, Holm/BH per table"; } > "$OUT/README.txt"

declare -a S0 S1 S2 S3
i=0; for spec in "${CONFIGS[@]}"; do case $((i%4)) in 0) S0+=("$spec");;1) S1+=("$spec");;2) S2+=("$spec");;3) S3+=("$spec");; esac; i=$((i+1)); done
[ ${#S0[@]} -gt 0 ] && set_slot 0 bash "$SELF" --worker "$OUT" "${S0[@]}" &
[ ${#S1[@]} -gt 0 ] && set_slot 1 bash "$SELF" --worker "$OUT" "${S1[@]}" &
[ ${#S2[@]} -gt 0 ] && set_slot 2 bash "$SELF" --worker "$OUT" "${S2[@]}" &
[ ${#S3[@]} -gt 0 ] && set_slot 3 bash "$SELF" --worker "$OUT" "${S3[@]}" &
wait
echo "=== all 3D subject scoring done -> $OUT/by_subject ==="

B="base3d=$OUT/by_subject/metrics_by_subject__base3d.csv"
run_stats() { local tbl="$1"; shift; local vs=(); for n in "$@"; do
    [ -f "$OUT/by_subject/metrics_by_subject__$n.csv" ] && vs+=("$n=$OUT/by_subject/metrics_by_subject__$n.csv"); done
  [ ${#vs[@]} -gt 0 ] && set_slot 0 env PYTHONPATH="$REPO_DIR" "$PY" -m helpers.eval_subject_stats paired \
    --baseline "$B" --variants "${vs[@]}" --out "$OUT/stats/stats__$tbl.csv"; }
run_stats mag_phase   mag
run_stats preprocessing mag_prepro phase_prepro otsu
run_stats augmentation  aug1 aug2 aug3
run_stats soft_labels   soft1 soft2 soft3
echo "=== 3D stats -> $OUT/stats ==="