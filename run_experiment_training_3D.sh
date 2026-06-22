#!/usr/bin/env bash
# Train a 3D experiment from its config. Invoke UNDER set_slot (GPU/CPU/RAM):
#   set_slot 3 bash run_experiment_training_3D.sh <exp_name> [fold]
# Builds the symlinked dataset + preprocesses on first use; always injects our 4-fold split.
# Quick smoke: prefix NNUNET_NUM_EPOCHS=2.
set -euo pipefail
EXP="${1:?usage: run_experiment_training_3D.sh <exp_name> [fold]}"
FOLD="${2:-0}"
cd "$(dirname "$0")"; source paths.sh         # REPO_DIR, PROJECT_ROOT, PY, NNUNET_BIN, nnUNet_*, CLEAN_DATASET
CFG="experiments/3D/${EXP}.json"
[ -f "$CFG" ] || { echo "ERROR: no config $CFG (see experiments/3D/)"; exit 1; }
eval "$("$PY" experiments/load_config.py "$CFG" | tr -d '\r')"
export NNUNET_EXP_CONFIG="$TRAINER_CONFIG_PATH"
export nnUNet_results="$PROJECT_ROOT/nnUNet_data/nnUNet_results/${EXP_DIM}/${EXP_FAMILY}/${EXP_NAME}"
mkdir -p "$nnUNet_results"
DS="Dataset$(printf %03d "$DATASET_ID")_${DATASET_NAME}"; DS_RAW="$nnUNet_raw/$DS"; DS_PP="$nnUNet_preprocessed/$DS"
echo "=== train 3D | exp=$EXP_NAME family=$EXP_FAMILY dataset=$DS conf=$CONFIGURATION trainer=$TRAINER fold=$FOLD ==="
echo "    results -> $nnUNet_results"
[ -n "${NNUNET_NUM_EPOCHS:-}" ] && echo "    NNUNET_NUM_EPOCHS=$NNUNET_NUM_EPOCHS (override)"

# 1) symlinked raw dataset (no image copies)
[ -d "$DS_RAW" ] || { echo "[build] $DS"; "$PY" 3D_workspace/dataset_prep/build_dataset_3d.py \
    --clean-root "$CLEAN_DATASET" --out-raw "$nnUNet_raw" --dataset-id "$DATASET_ID" --name "$DATASET_NAME" --channels "$DATASET_CHANNELS"; }
# 2) preprocess for this configuration
[ -d "$DS_PP/${PLANS}_${CONFIGURATION}" ] || { echo "[preprocess]"; "$NNUNET_BIN/nnUNetv2_plan_and_preprocess" -d "$DATASET_ID" -c "$CONFIGURATION" --verify_dataset_integrity; }
# 3) ALWAYS inject our 4-fold subject split (nnUNet writes its own 5-fold otherwise)
cp "$DS_RAW/splits_final.json" "$DS_PP/splits_final.json"
# 3b) optional patch-size variant (collapsed patch sweep), reusing the same preprocessing
[ -n "${PATCH_SIZE:-}" ] && { PLANS=$("$PY" experiments/make_patch_plans.py "$DS_PP" "$PLANS" "$CONFIGURATION" "$PATCH_SIZE" | tr -d '\r'); echo "[patch] plans=$PLANS ($PATCH_SIZE)"; }
# 4) train
echo "[train] nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD -tr $TRAINER -p $PLANS"
"$NNUNET_BIN/nnUNetv2_train" "$DATASET_ID" "$CONFIGURATION" "$FOLD" -tr "$TRAINER" -p "$PLANS" --npz
echo "=== done: $EXP_NAME fold $FOLD ==="
