#!/usr/bin/env bash
# Re-score the teacher's existing predictions (Dice + HD95) against ground truth, folds 1-3, +-TTA.
set -uo pipefail
REPO_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source "$REPO_DIR/paths.sh"

GT="$nnUNet_preprocessed/Dataset011_3D_MagPhase/gt_segmentations"
TEACHER_PRED="$nnUNet_results/paper_results/3D/ablations/base3d/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres/inference_results/patchsize_5_adamw_validation_EVAL"
cd "$REPO_DIR"

i=0
for suffix in "" "_TTA"; do
  for fold in 1 2 3; do
    slot=$((i % 4))
    PRED_DIR="$TEACHER_PRED/fold_${fold}${suffix}/predictions_3d"
    echo "launch teacher fold_${fold}${suffix} on slot $slot"
    set_slot "$slot" "$PY" -m helpers.eval --pred-dir "$PRED_DIR" --gt-dir "$GT" --out-dir "$PRED_DIR" \
      > "$PRED_DIR/eval_recompute.log" 2>&1 &
    i=$((i + 1))
    [ $((i % 4)) -eq 0 ] && wait
  done
done
wait
echo "TEACHER_EVAL_DONE ($i)"
