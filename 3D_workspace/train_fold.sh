#!/bin/bash
# Called by the Orchestrator inside tmux -> set_slot
# Usage: bash train_fold.sh <FOLD> <DATASET_ID> <TRAINER_CLASS> <VENV_PATH>

FOLD=$1
DATASET_ID=$2
TRAINER_CLASS=$3
VENV_PATH=$4
GPU_ID=$5 

# 1. Activate Environment
source "${VENV_PATH}/bin/activate"
NNUNET_BASE="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/nnunet_dataset"

# 2. Export Paths
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE}/nnUNet_results"

export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "------------------------------------------------"
echo "STARTING: Fold $FOLD | ID: $DATASET_ID | Trainer: $TRAINER_CLASS"
echo "------------------------------------------------"

# 3. Run Training
# Note: set_slot is handled by the calling script, so we just run the command here.
CUDA_VISIBLE_DEVICES=$GPU_ID nnUNetv2_train $DATASET_ID 3d_fullres $FOLD -tr $TRAINER_CLASS --npz -device cuda

echo "✅ Fold $FOLD Finished."