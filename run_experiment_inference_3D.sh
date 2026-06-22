#!/usr/bin/env bash
# Run 3D inference for an experiment (standard nnUNet predictor). Invoke UNDER set_slot:
#   set_slot 3 bash run_experiment_inference_3D.sh <exp> <input_dir> [flags]
#   <input_dir> = folder of nnUNet-format images (case_0000.nii.gz [+ _0001.nii.gz for phase]).
#   flags = --no-tta        disable test-time augmentation (default: on, nnUNet default)
#           --single-fold   predict with fold 0 only (default: ensemble folds 0-3)
#           --gt <dir>      after predicting, score vs ground-truth NIfTIs via helpers/eval.py
#                           (uniform metrics_casewise.csv / metrics_summary.csv, same schema as 2D)
# Output -> outputs/experiments/{family}/{exp}/<tag>/.
set -euo pipefail
EXP="${1:?usage: run_experiment_inference_3D.sh <exp> <input_dir> [--no-tta --single-fold --gt DIR]}"
INPUT="${2:?need an nnUNet-format input dir}"
FOLDS="0 1 2 3"; TTA_FLAG=""; ENS="ensemble"; TAG="tta"; GT=""
shift 2
while [ $# -gt 0 ]; do case "$1" in
  --no-tta) TTA_FLAG="--disable_tta"; TAG="notta";; --single-fold) FOLDS="0"; ENS="fold0";;
  --gt) GT="$2"; shift;; *) echo "unknown flag $1"; exit 1;; esac; shift; done
cd "$(dirname "$0")"; source paths.sh
eval "$("$PY" experiments/load_config.py "experiments/3D/${EXP}.json" | tr -d '\r')"
# patch-variant experiments were trained under a variant plans name -> match it at inference
[ -n "${PATCH_SIZE:-}" ] && PLANS="${PLANS}_p$(echo "$PATCH_SIZE" | tr ',' 'x')"
export nnUNet_results="$PROJECT_ROOT/nnUNet_data/nnUNet_results/${EXP_DIM}/${EXP_FAMILY}/${EXP_NAME}"
OUT="$OUTPUTS/experiments/${EXP_FAMILY}/${EXP_NAME}/predict_${ENS}_${TAG}"
mkdir -p "$OUT"
echo "=== infer 3D | exp=$EXP dataset=$DATASET_ID conf=$CONFIGURATION trainer=$TRAINER folds=[$FOLDS] tta=$([ -n "$TTA_FLAG" ] && echo no || echo yes) ==="
echo "    input  -> $INPUT"
echo "    output -> $OUT"
"$NNUNET_BIN/nnUNetv2_predict" -i "$INPUT" -o "$OUT" -d "$DATASET_ID" -c "$CONFIGURATION" \
  -tr "$TRAINER" -p "$PLANS" -f $FOLDS -chk checkpoint_best.pth $TTA_FLAG
# unified metrics (shared with the 2D pipeline) if ground truth is provided
[ -n "$GT" ] && { echo "[eval] scoring vs $GT"; PYTHONPATH="$REPO_DIR" "$PY" -m helpers.eval --pred-dir "$OUT" --gt-dir "$GT"; }
echo "=== done -> $OUT ==="
