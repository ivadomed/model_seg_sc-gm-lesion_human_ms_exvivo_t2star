
# #!/bin/bash

# set -e

# echo "--- ✅ Successfully running commands inside the allocated slot ---"
# echo "--- Using GPU: $CUDA_VISIBLE_DEVICES ---"

# cd /home/ge.polymtl.ca/pahoa/nih_project
# source .venv/bin/activate
# cd /home/ge.polymtl.ca/pahoa/nih_project/model_seg_sc-gm-lesion_human_ms_exvivo_t2star/training_scripts


# export PATH_DATA="/home/ge.polymtl.ca/pahoa/nih_project/datasets/bids_dataset"
# export PATH_PROCESSED="/home/ge.polymtl.ca/pahoa/nih_project/datasets/processed_data_full_multichannel"
# # export DATASET_ID=522
# # export TASK_NAME="MagPhaseExp_simple_training_ostu_only_(base)_2"
# export DATASET_ID=$3
# export TASK_NAME="MagPhaseExp_simple_training_${4}"
# # export DATASET_ID=520
# # export TASK_NAME="MagPhaseExp_simple_training_only_lesion_(base)"
# # export DATASET_ID=521
# # export TASK_NAME="MagPhaseExp_simple_training_2_mags"


# # Derivative and output paths for the new workflow
# export SLICED_DATA_DIR="${PATH_PROCESSED}/data_slice"
# export ORIGINAL_LABELS_DIR="${PATH_DATA}/derivatives/labels" # Path to original SC, GM, lesion labels

# # nnU-Net paths
# export nnUNet_raw="${PATH_PROCESSED}/nnunet_raw"
# export nnUNet_preprocessed="${PATH_PROCESSED}/nnunet_preprocessed"
# export nnUNet_results="${PATH_PROCESSED}/nnunet_results"



# # export GOLDEN_TASK_NAME="MagPhaseExp_simple-training_lesions-only"
# # export GOLDEN_DATASET_ID=516
# # export GOLDEN_TASK_NAME="MagPhaseExp_simple_training_soft_loss"
# # export GOLDEN_DATASET_ID=517
# # 2. Construct the full path to the golden splits file
# export GOLDEN_SPLIT_FILE="${nnUNet_raw}/Dataset515_MagPhaseExp_simple_training/splits_final.json"



# DATASET_PATH="${nnUNet_raw}/Dataset${DATASET_ID}_${TASK_NAME}"
# # --- REMOVED STEP 2 (combine_labels.py) ---

# ### NOT NECESSARY UNLESS NEW DATA
# # echo "Step 1: Extracting 2D slices for all individual labels..."
# # python extract_slices_multichannel.py \
# #   --path-data "$PATH_DATA" \
# #   --path-out "$SLICED_DATA_DIR" \
# #   --labels SC GM lesion \
# #   --label-folder "$ORIGINAL_LABELS_DIR" # Use the original, uncombined labels

# ## Need to run different versions of combine_labels.py depending on the desired label configuration, should be an input argument
# # echo "Step 2b: Creating Lesions-Only Dataset (lesion_WM, lesion_GM)"
# # python convert_bids_to_nnunet_multichannel.py \
# #   --path-data "$SLICED_DATA_DIR" \
# #   --path-out "$nnUNet_raw" \
# #   --tasknumber $DATASET_ID \
# #   --taskname "$TASK_NAME" \
# #   --label-suffixes SC GM lesion \
# #   --label-mode "lesions"

# # echo "Step 2a: Creating All Label Dataset"
# # python convert_bids_to_nnunet_multichannel.py \
# #   --path-data "$SLICED_DATA_DIR" \
# #   --path-out "$nnUNet_raw" \
# #   --tasknumber $DATASET_ID \
# #   --taskname "$TASK_NAME" \
# #   --label-suffixes SC GM lesion \
# #   --label-mode "all"


# # echo "Step 2c: Creating Merged-Lesion Dataset (WM, GM, lesion)"
# # python convert_bids_to_nnunet_multichannel.py \
# #   --path-data "$SLICED_DATA_DIR" \
# #   --path-out "$nnUNet_raw" \
# #   --tasknumber $DATASET_ID \
# #   --taskname "$TASK_NAME" \
# #   --label-suffixes SC GM lesion \
# #   --label-mode "sc_and_lesion"


# echo "Step 3: Adapting the golden standard data split..."
# python adapt_split_file.py \
#   --source-split-json "$GOLDEN_SPLIT_FILE" \
#   --target-dataset-dir "$CURRENT_DATASET_PATH" \
#   --old-task-name "$GOLDEN_TASK_NAME" \
#   --new-task-name "$TASK_NAME"

# #### NOT NEEDED UNLESS NEW DATA
# # echo "Step 3: Creating subject-aware data splits..."
# # python create_subject_split.py \
# #   --dataset-path "$DATASET_PATH" \
# #   --manifest-path "${DATASET_PATH}/inference_manifest.json"


# # # # echo "Step 4: Running nnU-Net..."
# # # # # # These commands remain the same

# # nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity
# CUDA_VISIBLE_DEVICES=$2 nnUNetv2_train $DATASET_ID 2d $1 --npz -tr nnUNetTrainerWandb -device cuda


#!/bin/bash

set -e

# This script runs nnU-Net training for a specific fold.
# It assumes that the environment is activated and preprocessing is complete.

# Arguments:
# $1: FOLD (e.g., 0)
# $2: GPU_ID (e.g., 0)
# $3: DATASET_ID (e.g., 525)

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <FOLD> <GPU_ID> <DATASET_ID>"
    exit 1
fi

FOLD=$1
GPU_ID=$2
DATASET_ID=$3

echo "--- Starting training for Fold ${FOLD} on GPU ${GPU_ID} for Dataset ${DATASET_ID} ---"
echo "--- Using GPU from environment: $CUDA_VISIBLE_DEVICES ---"

# --- Activate Environment and Set Paths ---
cd /home/ge.polymtl.ca/pahoa/nih_project
source .venv/bin/activate
cd /home/ge.polymtl.ca/pahoa/nih_project/model_seg_sc-gm-lesion_human_ms_exvivo_t2star/2D_workspace/training_scripts

export PATH_PROCESSED="/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/processed_data_full_multichannel"
export nnUNet_raw="${PATH_PROCESSED}/nnunet_raw"
export nnUNet_preprocessed="${PATH_PROCESSED}/nnunet_preprocessed"
export nnUNet_results="${PATH_PROCESSED}/nnunet_results"

# --- Run Training ---
# The CUDA_VISIBLE_DEVICES is set here to ensure the correct GPU is used.
# The script receives the GPU_ID from the calling script (run_five_folds_exp.sh).
echo "Running nnU-Net training on GPU ${GPU_ID}..."
CUDA_VISIBLE_DEVICES=$GPU_ID nnUNetv2_train "$DATASET_ID" 2d "$FOLD" --npz -tr nnUNetTrainerWandb -device cuda

echo "--- ✅ Finished training for Fold ${FOLD} ---"