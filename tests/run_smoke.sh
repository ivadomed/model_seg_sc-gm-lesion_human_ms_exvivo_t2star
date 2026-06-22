#!/usr/bin/env bash
# Fast end-to-end smoke test: 2-epoch training + inference + metrics for both 2D and 3D.
# Proves the whole pipeline runs (NOT a real model). Run UNDER set_slot:
#   set_slot 3 bash tests/run_smoke.sh
# Exits non-zero if any step fails.
set -uo pipefail
cd "$(dirname "$0")/.."; source paths.sh
export NNUNET_NUM_EPOCHS=2 WANDB_MODE=disabled
PASS=0; FAIL=0
ok(){ if eval "$2"; then echo "  PASS  $1"; PASS=$((PASS+1)); else echo "  FAIL  $1"; FAIL=$((FAIL+1)); fi; }
R(){ echo; echo ">>> $*"; "$@" >/tmp/smoke_step.log 2>&1 || { echo "  (step exited $?)"; tail -3 /tmp/smoke_step.log; }; }

echo "===== nnU-Net smoke test (2 epochs) ====="
R bash install_trainers.sh
ok "trainers import" "grep -q 'OK  nnUnet3DCustomTrainer' /tmp/smoke_step.log"

R bash run_experiment_training_2D.sh winning 0
ok "2D training -> checkpoint" "[ -f \"$nnUNet_results/2D/winning/winning/Dataset021_2D_MagPhase/nnUNetTrainerWandb__nnUNetPlans__2d/fold_0/checkpoint_best.pth\" ]"

R bash run_experiment_training_3D.sh adamw_baseline 0
ok "3D training -> checkpoint" "find \"$nnUNet_results/3D/adamw_baseline\" -name checkpoint_best.pth | grep -q ."

# tiny 3D input (2 cases) for inference + eval
IN=$(mktemp -d); RAW="$nnUNet_raw/Dataset011_3D_MagPhase/imagesTr"
for c in MagPhase_0000 MagPhase_0001; do for ch in 0000 0001; do ln -sf "$(realpath "$RAW/${c}_${ch}.nii.gz")" "$IN/${c}_${ch}.nii.gz"; done; done
R bash run_experiment_inference_3D.sh adamw_baseline "$IN" --single-fold --gt "$nnUNet_preprocessed/Dataset011_3D_MagPhase/gt_segmentations"
ok "3D inference + unified metrics" "[ -f \"$OUTPUTS/experiments/adamw_baseline/adamw_baseline/predict_fold0_tta/metrics_casewise.csv\" ]"
rm -rf "$IN"

R bash run_experiment_inference_2D.sh winning validation --single-fold
ok "2D validation inference + metrics" "find \"$OUTPUTS/experiments/winning/winning/validation_fold0\" -name 'metrics_2d_casewise.csv' | grep -q ."

echo; echo "===== smoke result: $PASS passed, $FAIL failed ====="
exit $FAIL
