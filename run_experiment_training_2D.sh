#!/usr/bin/env bash
# Train a 2D experiment from its config. Invoke UNDER set_slot (GPU/CPU/RAM):
#   set_slot 3 bash run_experiment_training_2D.sh <exp_name> [fold]
# Builds the 2D slice dataset (axis-2, test subjects excluded) + preprocesses on first use;
# always injects our 4-fold subject split. Quick smoke: prefix NNUNET_NUM_EPOCHS=2.
set -euo pipefail
EXP="${1:?usage: run_experiment_training_2D.sh <exp_name> [fold]}"
FOLD="${2:-0}"
cd "$(dirname "$0")"; source paths.sh
CFG="experiments/2D/${EXP}.json"
[ -f "$CFG" ] || { echo "ERROR: no config $CFG (see experiments/2D/)"; exit 1; }
eval "$("$PY" experiments/load_config.py "$CFG" | tr -d '\r')"
export NNUNET_EXP_CONFIG="$TRAINER_CONFIG_PATH"
export nnUNet_results="$PROJECT_ROOT/nnUNet_data/nnUNet_results/${EXP_DIM}/${EXP_FAMILY}/${EXP_NAME}"
mkdir -p "$nnUNet_results"
DS="Dataset$(printf %03d "$DATASET_ID")_${DATASET_NAME}"; DS_RAW="$nnUNet_raw/$DS"; DS_PP="$nnUNet_preprocessed/$DS"
echo "=== train 2D | exp=$EXP_NAME family=$EXP_FAMILY dataset=$DS conf=$CONFIGURATION trainer=$TRAINER fold=$FOLD ==="
echo "    results -> $nnUNet_results"
[ -n "${NNUNET_NUM_EPOCHS:-}" ] && echo "    NNUNET_NUM_EPOCHS=$NNUNET_NUM_EPOCHS (override)"

# 1) 2D slice dataset (annotated slices, axis 2, test subjects excluded)
[ -d "$DS_RAW" ] || { echo "[build] $DS"; "$PY" 2D_workspace/dataset_prep/build_dataset_2d.py \
    --clean-root "$CLEAN_DATASET" --out-raw "$nnUNet_raw" --work "$PROJECT_ROOT/nnUNet_data/work/sliced_2d" \
    --dataset-id "$DATASET_ID" --name "$DATASET_NAME" --label-mode "$DATASET_LABEL_MODE" --channels "$DATASET_CHANNELS"; }
# 2) preprocess
[ -d "$DS_PP/${PLANS}_${CONFIGURATION}" ] || { echo "[preprocess]"; "$NNUNET_BIN/nnUNetv2_plan_and_preprocess" -d "$DATASET_ID" -c "$CONFIGURATION" --verify_dataset_integrity; }
# 3) ALWAYS inject our 4-fold subject split
cp "$DS_RAW/splits_final.json" "$DS_PP/splits_final.json"
# 4) train
echo "[train] nnUNetv2_train $DATASET_ID $CONFIGURATION $FOLD -tr $TRAINER -p $PLANS"
"$NNUNET_BIN/nnUNetv2_train" "$DATASET_ID" "$CONFIGURATION" "$FOLD" -tr "$TRAINER" -p "$PLANS" --npz
echo "=== done: $EXP_NAME fold $FOLD ==="
