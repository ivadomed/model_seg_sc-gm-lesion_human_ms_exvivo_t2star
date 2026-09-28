#!/usr/bin/env bash
# Score every finished student (condition x fold in {1,2,3}) against ground truth, using its
# existing end-of-training validation predictions (single-model, no-TTA). 4 jobs at a time.
set -uo pipefail
REPO_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source "$REPO_DIR/paths.sh"

GT="$nnUNet_preprocessed/Dataset011_3D_MagPhase/gt_segmentations"
STUDENT_RUN="nnUNetPlansDistillS/Dataset011_3D_MagPhase/nnUnet3DDistillTrainer__nnUNetPlansDistillS__3d_fullres"
cd "$REPO_DIR"

i=0
for condition in control kd_T1 kd_T2 kd_T4; do
  for fold in 1 2 3; do
    slot=$((i % 4))
    FOLD_DIR="$nnUNet_results/distillation/$condition/$STUDENT_RUN/fold_$fold"
    echo "launch [$condition fold_$fold] on slot $slot"
    set_slot "$slot" "$PY" -m helpers.eval \
      --pred-dir "$FOLD_DIR/validation" --gt-dir "$GT" --out-dir "$FOLD_DIR/validation" \
      > "$FOLD_DIR/validation/eval.log" 2>&1 &
    i=$((i + 1))
    [ $((i % 4)) -eq 0 ] && wait
  done
done
wait
echo "ALL_EVAL_DONE ($i jobs)"
