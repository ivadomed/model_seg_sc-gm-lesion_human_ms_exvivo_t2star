#!/usr/bin/env bash
# Run 2D inference for an experiment. Invoke UNDER set_slot:
#   set_slot 3 bash run_experiment_inference_2D.sh <exp> <mode> [input_path] [flags]
#   mode  = validation | test | volume        (volume: input_path = a clean mag volume)
#   flags = --tta           enable test-time augmentation
#           --single-fold   use only fold 0 (default: ensemble folds 0-3)
#           --legacy        use the historical published model (paper_results) instead of a fresh one
# Output -> outputs/experiments/{family}/{exp}/<mode>/  (per-fold + cross-fold metric summary).
set -euo pipefail
EXP="${1:?usage: run_experiment_inference_2D.sh <exp> <mode> [input] [--tta --single-fold --legacy]}"
MODE="${2:?mode = validation|test|volume}"
INPUT=""; TTA=""; FOLDS="0 1 2 3"; LEGACY=""; ENS="ensemble"
shift 2
for a in "$@"; do case "$a" in
  --tta) TTA="--use_tta";; --single-fold) FOLDS="0"; ENS="fold0";;
  --legacy) LEGACY=1;; --*) echo "unknown flag $a"; exit 1;; *) INPUT="$a";; esac; done
cd "$(dirname "$0")"; source paths.sh
eval "$("$PY" experiments/load_config.py "experiments/2D/${EXP}.json" | tr -d '\r')"
DS="Dataset$(printf %03d "$DATASET_ID")_${DATASET_NAME}"

if [ -n "$LEGACY" ]; then
  MODEL="$nnUNet_results/paper_results/2D/winning/winning_combination/nnUNetTrainerWandb__nnUNetPlans__2d"
else
  MODEL="$nnUNet_results/2D/${EXP_FAMILY}/${EXP_NAME}/$DS/${TRAINER}__${PLANS}__2d"
fi
[ -d "$MODEL" ] || { echo "ERROR: model folder not found: $MODEL"; exit 1; }
OUT="$OUTPUTS/experiments/${EXP_FAMILY}/${EXP_NAME}${LEGACY:+_legacy}/${MODE}_${ENS}${TTA:+_tta}"
mkdir -p "$OUT"
# -path: the input volume (volume mode) or the nnUNet_raw dataset dir (validation/test, has imagesTr/labelsTr/splits)
if [ "$MODE" = "volume" ]; then PATH_ARG="${INPUT:?volume mode needs an input volume path}"; else PATH_ARG="$nnUNet_raw/$DS"; fi
echo "=== infer 2D | exp=$EXP mode=$MODE folds=[$FOLDS] tta=${TTA:-no} legacy=${LEGACY:-0} ==="
echo "    model -> $MODEL"
echo "    path  -> $PATH_ARG"
echo "    out   -> $OUT"
PYTHONPATH="$REPO_DIR" "$PY" 2D_workspace/inference/inference_2.py \
  -experiment "$EXP_NAME" -model_folder "$MODEL" -mode "$MODE" \
  -path "$PATH_ARG" -folds $FOLDS $TTA -output_root "$OUT"

# Unified case-level metrics (identical schema to the 3D pipeline), scored from the saved
# predictions -- no re-inference. For validation/test the per-slice GT is the dataset's labelsTr.
if [ "$MODE" != "volume" ]; then
  for FD in "$OUT"/fold_*/predictions_nifti; do
    [ -d "$FD" ] && PYTHONPATH="$REPO_DIR" "$PY" -m helpers.eval --pred-dir "$FD" --gt-dir "$nnUNet_raw/$DS/labelsTr" --out-dir "$(dirname "$FD")"
  done
  PYTHONPATH="$REPO_DIR" "$PY" -m helpers.eval --crossfold "$OUT" 2>/dev/null || true
fi
echo "=== done -> $OUT ==="
