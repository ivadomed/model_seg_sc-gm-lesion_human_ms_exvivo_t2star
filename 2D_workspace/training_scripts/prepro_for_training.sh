#!/bin/bash

set -e

# This script performs the one-time setup for a nnU-Net experiment.
# It should be run once before starting the training for the 5 folds.

# Arguments:
# $1: DATASET_ID (e.g., 525)
# $2: EXPERIMENT_NAME (e.g., specific_experiment OR mag-one-channel)
# $3: LABEL_MODE (e.g., "all", "lesions", "sc_and_lesion")

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <DATASET_ID> <EXPERIMENT_NAME> <LABEL_MODE>"
    echo "LABEL_MODE can be 'all', 'lesions', or 'sc_and_lesion'"
    exit 1
fi

DATASET_ID=$1
EXPERIMENT_NAME=$2
LABEL_MODE=$3
TASK_NAME="MagPhaseExp_simple_training_${EXPERIMENT_NAME}"

echo "--- Starting Preprocessing for Dataset ID: ${DATASET_ID}, Task: ${TASK_NAME} ---"

# --- Activate Environment and Set Paths ---
cd /home/ge.polymtl.ca/pahoa/nih_project
source .venv/bin/activate
cd /home/ge.polymtl.ca/pahoa/nih_project/model_seg_sc-gm-lesion_human_ms_exvivo_t2star/2D_workspace/training_scripts

export PATH_DATA="/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_dataset/bids_dataset"
export PATH_PROCESSED="/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel"
export SLICED_DATA_DIR="${PATH_PROCESSED}/data_slice"
export ORIGINAL_LABELS_DIR="${PATH_DATA}/derivatives/labels"

export nnUNet_raw="${PATH_PROCESSED}/nnunet_raw"
export nnUNet_preprocessed="${PATH_PROCESSED}/nnunet_preprocessed"
export nnUNet_results="${PATH_PROCESSED}/nnunet_results"

# --- Determine Channel Configuration ---
# If the experiment name is explicitly 'mag-one-channel', we use the single channel mode.
# Otherwise, we default to the standard 2-channel (mag + phase) mode.
CHANNEL_CONFIG="mag_phase"

if [[ "$EXPERIMENT_NAME" == *"mag-one-channel"* ]]; then
    CHANNEL_CONFIG="mag-one-channel"
    echo "ℹ️  Detected 'mag-one-channel' experiment. Using single channel (Magnitude only)."
else
    echo "ℹ️  Using standard dual channel mode (Magnitude + Phase)."
fi

# --- Data Conversion (BIDS to nnU-Net format) ---
echo "Step 1: Converting BIDS to nnU-Net format with label mode: ${LABEL_MODE} and channel config: ${CHANNEL_CONFIG}"
python ../dataset_preprocessing_scripts/convert_bids_to_nnunet_multichannel.py \
  --path-data "$SLICED_DATA_DIR" \
  --path-out "$nnUNet_raw" \
  --tasknumber "$DATASET_ID" \
  --taskname "$TASK_NAME" \
  --label-suffixes SC GM lesion \
  --label-mode "$LABEL_MODE" \
  --channel-config "$CHANNEL_CONFIG"

# --- Adapt Splits ---
# Note: The 'mag-one-channel' experiment will likely create a dataset structure that differs
# significantly in file naming (_0000 vs _0000 + _0001 presence), but the sample IDs stay the same.
# We can still reuse the split file if the sample IDs (case names) match.
echo "Step 2: Adapting the golden standard data split..."
export GOLDEN_TASK_NAME="MagPhaseExp_simple_training"
export GOLDEN_SPLIT_FILE="${nnUNet_raw}/Dataset515_MagPhaseExp_simple_training/splits_final.json"
TARGET_DATASET_PATH="${nnUNet_raw}/Dataset${DATASET_ID}_${TASK_NAME}"

python ../dataset_preprocessing_scripts/adapt_split_file.py \
  --source-split-json "$GOLDEN_SPLIT_FILE" \
  --target-dataset-dir "$TARGET_DATASET_PATH" \
  --old-task-name "$GOLDEN_TASK_NAME" \
  --new-task-name "$TASK_NAME"

# --- nnU-Net Plan and Preprocess ---
echo "Step 3: Running nnU-Net plan and preprocess..."
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity

echo "--- ✅ Preprocessing complete. ---"