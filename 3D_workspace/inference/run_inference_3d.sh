#!/bin/bash

# --- Configuration ---
GPU_ID=0
MODE="validation" # Options: validation (uses labelsTr), test (uses labelsTs)
FOLDS=(0 1 2 3)

# Define your experiments here: [DATASET_ID]="EXPERIMENT_NAME"
declare -A EXPERIMENT_MAP
EXPERIMENT_MAP=( 
    # ["1700"]="mag_only" 
    # ["1701"]="mag-one-channel"
    # ["1702"]="mag_and_phase"
    # ["1703"]="otsu_threshold"
    # ["1704"]="spatial_augmentation"
    # ["1705"]="spatial_augmentation_1"
    # ["1706"]="spatial_augmentation_2"
    # ["1707"]="soft_edges_loss"
    # ["1708"]="patchsize_"
    # ["1709"]="patchsize_192,64,168"
    # ["1710"]="patchsize_192,96,160"
    # ["1711"]="patchsize_4"
    # ["1712"]="patchsize_5"
    # ["1713"]="patchsize_5_spatial_aug_2"
    ["1714"]="patchsize_5_spatial_aug_1"
    ["1715"]="patchsize_5_otsu_masking"
)
USE_TTA=False
ENSEMBLE=False

# --- Path Definitions ---
PROJECT_ROOT="/home/ge.polymtl.ca/pahoa/nih_project"

# 1. NNUNET RAW ROOT (Input Data)
# The script will look for Dataset{ID}_MagPhase_{EXP} inside this folder
NNUNET_RAW_ROOT="${PROJECT_ROOT}/datasets/3D_datasets/train_datasets/nnunet_dataset/nnUNet_raw"

# 2. PATH TO 2D GT (For 2D Slice Evaluation - Manual GT)
# Note: Ensure "nnUNet_raw" capitalization matches your system (sometimes it is nnUNet_raw vs nnunet_raw)
PATH_2D_GT="${PROJECT_ROOT}/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/labelsTr"

# 3. 2D MANIFEST (Mapping 2D filenames to 3D Volume + Slice Index)
MANIFEST_2D_PATH="${PROJECT_ROOT}/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/inference_manifest.json"

# 4. OUTPUT ROOT
PATH_RESULTS="${PROJECT_ROOT}/datasets/3D_datasets/train_datasets/nnunet_dataset/nnUNet_results"

# --- Logic Loop ---

for DATASET_ID in "${!EXPERIMENT_MAP[@]}"; do
    EXPERIMENT="${EXPERIMENT_MAP[$DATASET_ID]}"
    
    # Define Task Name (Adjust if your folder naming convention differs)
    # Based on your prompt: "Dataset1700_MagPhase_mag_only"
    TASK_NAME="MagPhase_${EXPERIMENT}" 
    
    # 1. Define Input Data Path (The nnUNet_raw folder for this dataset)
    CURRENT_RAW_DATA_PATH="${NNUNET_RAW_ROOT}/Dataset${DATASET_ID}_${TASK_NAME}"
    
    # 2. Define 3D Manifest Path (This is inside the raw data folder)
    MANIFEST_3D_PATH="${CURRENT_RAW_DATA_PATH}/manifest.json"

    # 3. Define Model Folder
    # Adjust "nnUnet3DCustomTrainer" if you used the standard trainer
    MODEL_FOLDER="${PATH_RESULTS}/Dataset${DATASET_ID}_${TASK_NAME}/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"
    
    # 4. Output Directory
    OUTPUT_ROOT="${MODEL_FOLDER}/inference_results/${EXPERIMENT}_${MODE}_EVAL"

    echo "========================================================"
    echo "🚀 Experiment: $EXPERIMENT (ID: $DATASET_ID)"
    echo "   Input Data: $CURRENT_RAW_DATA_PATH"
    echo "   3D Manifest: $MANIFEST_3D_PATH"
    echo "========================================================"

    if [ ! -f "$MANIFEST_3D_PATH" ]; then
        echo "❌ Error: 3D Manifest not found at $MANIFEST_3D_PATH"
        continue
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID python inference_3d.py \
        -experiment "$EXPERIMENT" \
        -model_folder "$MODEL_FOLDER" \
        -mode "$MODE" \
        -input_path_raw "$CURRENT_RAW_DATA_PATH" \
        -gt_path_2d "$PATH_2D_GT" \
        -manifest_2d "$MANIFEST_2D_PATH" \
        -manifest_3d "$MANIFEST_3D_PATH" \
        -folds "${FOLDS[@]}" \
        -output_root "$OUTPUT_ROOT" \
        --save_predictions \
        --use_tta "$USE_TTA" \
        --ensemble "$ENSEMBLE"

    echo "🏁 Finished Experiment: $EXPERIMENT"
done