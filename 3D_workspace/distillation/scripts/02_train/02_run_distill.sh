#!/usr/bin/env bash
# Train the distillation student for one fold.
#   02_run_distill.sh <fold> [config.json] [plans_name]
#
# DISTILL_TAG (env, optional) tags the results dir: nnUNet_results/distillation/<tag>/<plans>/...
# SKIP_INSTALL=1 (env, optional) skips the trainer install -- do this for parallel fold launches
# (concurrent copies into the shared venv dir race), after installing it once yourself.
# KD_TEMPERATURE / KD_WEIGHT / TEACHER_LOGITS_DIR (env, optional) override the config file.
set -euo pipefail
CALLER_DIR="$(pwd)"
REPO_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source "$REPO_DIR/paths.sh"

FOLD="${1:?usage: 02_run_distill.sh <fold> [config.json] [plans_name]}"
CONFIG="${2:-$REPO_DIR/3D_workspace/distillation/configs/distill.json}"
PLANS="${3:-nnUNetPlansDistillS}"
DATASET_ID=11
DATASET_DIR="Dataset011_3D_MagPhase"
CONFIGURATION=3d_fullres
TRAINER=nnUnet3DDistillTrainer

case "$CONFIG" in /*) ;; *) CONFIG="$CALLER_DIR/$CONFIG" ;; esac
[ -f "$CONFIG" ] || { echo "ERROR: config not found: $CONFIG"; exit 1; }
export NNUNET_EXP_CONFIG="$CONFIG"

export nnUNet_results="$PROJECT_ROOT/nnUNet_data/nnUNet_results/distillation${DISTILL_TAG:+/$DISTILL_TAG}/${PLANS}"
mkdir -p "$nnUNet_results"

if [ -z "${SKIP_INSTALL:-}" ]; then
  bash "$REPO_DIR/3D_workspace/distillation/scripts/02_train/01_install_trainer.sh"
fi

DS_PP="$nnUNet_preprocessed/$DATASET_DIR"
[ -f "$DS_PP/${PLANS}.json" ] || { echo "ERROR: student plans $DS_PP/${PLANS}.json missing -- run 01_prepare/02_make_student_plans.py"; exit 1; }
[ -f "$DS_PP/splits_final.json" ] || { echo "ERROR: splits_final.json missing in $DS_PP"; exit 1; }

echo "=== distill | fold=$FOLD trainer=$TRAINER plans=$PLANS config=$CONFIG ==="
echo "    results -> $nnUNet_results"
"$NNUNET_BIN/nnUNetv2_train" "$DATASET_ID" "$CONFIGURATION" "$FOLD" -tr "$TRAINER" -p "$PLANS" --npz
echo "=== done: distill fold $FOLD ==="
