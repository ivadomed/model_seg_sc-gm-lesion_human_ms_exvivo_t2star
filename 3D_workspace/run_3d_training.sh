#!/bin/bash

# --- CONFIGURATION ---
BIDS_DATA_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset"
NNUNET_BASE="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/nnunet_dataset"
DATASET_ID=1700
DATASET_NAME="Dataset${DATASET_ID}_MagPhase"
TRAINER_CLASS="nnUNetTrainerMagPhase" # Ensure this python file is in PYTHONPATH
GPU_ID=0
# ---------------------

# Exports
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE}/nnUNet_results"

# PYTHONPATH needs to include current dir to find the CustomTrainer
export PYTHONPATH=$PYTHONPATH:$(pwd)

# mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results

# # 1. VERIFICATION
# echo "------------------------------------------------"
# echo "Step 1: Verifying Data Integrity (Skipped)"
# echo "------------------------------------------------"
# # python3 verify_bids_integrity.py --bids "$BIDS_DATA_PATH"

# # read -p "Check the output above. If labels are wrong, press Ctrl+C. Press Enter to continue."

# 2. PREPROCESSING & SPLITTING
echo "------------------------------------------------"
echo "Step 2: Converting BIDS to nnU-Net Raw"
echo "------------------------------------------------"
python3 preprocess_bids.py \
    --bids "$BIDS_DATA_PATH" \
    --nnunet_raw "$nnUNet_raw" \
    --id $DATASET_ID

# 3. PLANNING
echo "------------------------------------------------"
echo "Step 3: nnU-Net Planning (Analysis & Fingerprinting)"
echo "------------------------------------------------"
# This creates the preprocessed folder structure
nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 3d_fullres --verify_dataset_integrity

# 4. INJECT SPLITS
echo "------------------------------------------------"
echo "Step 4: Injecting Subject-Based Splits"
echo "------------------------------------------------"
# The python script created 'splits_final.json' in the current directory.
# We must move it to the preprocessed directory so nnU-Net finds it.
TARGET_SPLIT_DIR="${nnUNet_preprocessed}/${DATASET_NAME}"

if [ -d "$TARGET_SPLIT_DIR" ]; then
    mv splits_final.json "${TARGET_SPLIT_DIR}/splits_final.json"
    echo "Custom splits installed to ${TARGET_SPLIT_DIR}/splits_final.json"
else
    echo "ERROR: Preprocessed directory not found: $TARGET_SPLIT_DIR"
    exit 1
fi

# 5. TRAINING
echo "------------------------------------------------"
echo "Step 5: Training"
echo "------------------------------------------------"
# Since you have a 50GB GPU, 3d_fullres is definitely the way to go.
# Fold 0 will use the splits defined in our custom file.
CUDA_VISIBLE_DEVICES=$GPU_ID nnUNetv2_train $DATASET_ID 3d_fullres 0 -tr $TRAINER_CLASS

echo "Experiment launched."