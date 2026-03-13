#!/bin/bash

# --- Configuration ---
GPU_IDS_STR="0,1,2,3" # Comma-separated list of GPUs (e.g., "0,1")
VENV_PATH="/home/ge.polymtl.ca/pahoa/nih_project/.venv"
MODE="validation" # Options: validation, test, single_volume
FOLDS=(0 1 2 3)
MAX_PARALLEL_EXPERIMENTS=3

# --- Single Volume Mode Configuration ---
# This is only used if MODE="single_volume"
# SINGLE_VOLUME_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset/sub-TNU004/anat/S3_81.0cm_T2s_75i_TR45TE9_cor_12avg_1500_part-mag.nii.gz"
# SINGLE_VOLUME_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset/sub-TNU004/anat/S3_81.0cm_T2s_75i_TR45TE9_cor_12avg_1500_part-mag.nii.gz"
# SINGLE_VOLUME_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset/sub-TNU004/anat/S3_81.0cm_T2s_75i_TR45TE9_cor_12avg_1500_part-mag.nii.gz"
# SINGLE_VOLUME_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset/sub-TNU004/anat/S3_81.0cm_T2s_75i_TR45TE9_cor_12avg_1500_part-mag.nii.gz"


# Parse GPUs
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_STR"
NUM_GPUS=${#GPU_IDS[@]}

if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPU IDs provided."
    exit 1
fi

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
    # ["1714"]="patchsize_5_spatial_aug_1"


    # # ["1715"]="patchsize_5_otsu_masking"
    # ["1716"]="patchsize_5_soft_loss_fixed"
    # ["1717"]="patchsize_5_soft_loss_2_fixed"


    ["1718"]="patchsize_5_adamw"
    


    # ["1719"]="patchsize_5_adamw_otsu"
    # ["1720"]="patchsize_5_adamw_phase_prepro"
    # ["1721"]="patchsize_5_adamw_mag_prepro"
    # ["1722"]="patch_5_adamw_soft_loss_3"

    
    # ["1723"]="patch_5_adamw_spatial_aug_3"

    # ["1724"]="patch_5_adamw_mag_one_channel"

    # ["1725"]="patch_5_adamw_aug_no_rot_z"

    # ["1726"]="patch_5_adamw_aug_5"

    # ["1727"]="patch_5_adamw_aug_6"
    # ["1728"]="patch_5_adamw_aug_7"
    # ["1729"]="patch_5_adamw_aug_8"

    # ["1730"]="patch_5_adamw_aug_9"
    # ["1731"]="patch_5_adamw_aug_10"
    # ["1732"]="patch_5_adamw_aug_11"

    # ["1733"]="patch_5_adamw_aug_12" # all rotation

    # ["1734"]="patch_5_adamw_aug_13" # y rotatino
    # ["1735"]="patch_5_adamw_aug_14" # x rotation
    # ["1736"]="patch_5_adamw_aug_15" # z rotation

    # ["1737"]="patch_5_adamw_aug_16" # all translation

    # ["1738"]="patch_5_adamw_aug_17" # y translation
    # ["1739"]="patch_5_adamw_aug_18" # x translation
    # ["1740"]="patch_5_adamw_aug_19" # z translation

    # ["1741"]="patch_5_adamw_aug_20" # x z rotation
    # ["1742"]="patch_5_adamw_aug_21" # y z rotation
    # ["1743"]="patch_5_adamw_aug_22" # x y rotation
    # ["1744"]="patch_5_adamw_aug_23" # x z translation
)

USE_TTA=False
ENSEMBLE=False

# --- Path Definitions ---
PROJECT_ROOT="/home/ge.polymtl.ca/pahoa/nih_project"
SKIP_TEST_PREPRO=True
if [ "$MODE" = "test" ]; then
    TEST_TASK_ID="520" 
    TEST_TASK_NAME="TEST_SET_AXIAL"

    echo "⚠️  TEST MODE ACTIVATED"
    # Override paths for Test Mode
    # Use BIDS dataset root as raw data path
    NNUNET_RAW_ROOT="${PROJECT_ROOT}/datasets/3D_datasets/test_datasets/bids_test_dataset_REORIENTED"
    
    # Define locations for intermediate processing
    # We use a DYNAMIC folder to ensure we always process the latest BIDS data
    PROCESSED_TEST_ROOT="${PROJECT_ROOT}/datasets/2D_datasets/test_datasets/processed_test_set_dynamic"
    SLICED_TEST_DATA="${PROCESSED_TEST_ROOT}/data_slice_axial"
    NNUNET_RAW_TEST="${PROCESSED_TEST_ROOT}/nnunet_raw"
    
    # FORCE PREPROCESSING on for Test Mode to ensure new labels are picked up
    SKIP_TEST_PREPRO=False

    if [ "$SKIP_TEST_PREPRO" = "True" ] || [ "$SKIP_TEST_PREPRO" = "true" ]; then
        echo "⚠️  Skipping test set preprocessing as per configuration."
        echo "   Ensure that $NNUNET_RAW_TEST is already prepared with the correct structure."
        
        

        PATH_2D_GT="${NNUNET_RAW_TEST}/Dataset${TEST_TASK_ID}_${TEST_TASK_NAME}/labelsTr"
        MANIFEST_2D_PATH="${NNUNET_RAW_TEST}/Dataset${TEST_TASK_ID}_${TEST_TASK_NAME}/inference_manifest.json"
        
        echo "✅ Test set preprocessing skipped."
        echo "   GT Path: $PATH_2D_GT"
        echo "   Manifest: $MANIFEST_2D_PATH"

    else
        # Ensure directories exist and CLEAN them to avoid mixing slices from different axes
        rm -rf "$SLICED_TEST_DATA"
        mkdir -p "$SLICED_TEST_DATA"
        mkdir -p "$PROCESSED_TEST_ROOT"
        
        # 1. EXTRACT SLICES (From 3D BIDS -> 2D BIDS Slices)
        echo "--- 1. Extracting 2D slices from BIDS Test Set (AXIAL) ---"
        set_slot "${GPU_IDS[0]}" bash -c "source ${VENV_PATH}/bin/activate && python ../../2D_workspace/dataset_preprocessing_scripts/extract_slices_multichannel.py \
            --path-data "$NNUNET_RAW_ROOT" \
            --labels SC GM lesion \
            --path-out "$SLICED_TEST_DATA" \
            --label-folder "derivatives/labels" \
            --axis 2
            "

        # 2. CONVERT TO NNUNET (From 2D BIDS Slices -> nnUNet format for GT loading)
        echo "--- 2. Converting Slices to nnUNet Format (for Labels GT) ---"
        # Use a unique ID for the test set to avoid conflicts
        
        
        set_slot "${GPU_IDS[0]}" bash -c "source ${VENV_PATH}/bin/activate && python ../../2D_workspace/dataset_preprocessing_scripts/convert_bids_to_nnunet_multichannel.py \
            --path-data "$SLICED_TEST_DATA" \
            --path-out "$NNUNET_RAW_TEST" \
            --tasknumber "$TEST_TASK_ID" \
            --taskname "$TEST_TASK_NAME" \
            --label-suffixes SC GM lesion \
            --label-mode "all" \
            --channel-config "mag_phase"
            "

        # Test Set 2D GT and Manifest
        # These are now pointing to the dynamically generated ones based on current BIDS
        PATH_2D_GT="${NNUNET_RAW_TEST}/Dataset${TEST_TASK_ID}_${TEST_TASK_NAME}/labelsTr"
        MANIFEST_2D_PATH="${NNUNET_RAW_TEST}/Dataset${TEST_TASK_ID}_${TEST_TASK_NAME}/inference_manifest.json"
        
        echo "✅ Dynamic Test Set Preparation Complete."
        echo "   GT Path: $PATH_2D_GT"
        echo "   Manifest: $MANIFEST_2D_PATH"
    fi

else
    # 1. NNUNET RAW ROOT (Input Data)
    # The script will look for Dataset{ID}_MagPhase_{EXP} inside this folder
    NNUNET_RAW_ROOT="${PROJECT_ROOT}/datasets/3D_datasets/train_datasets/nnunet_dataset/nnUNet_raw"

    # 2. PATH TO 2D GT (For 2D Slice Evaluation - Manual GT)
    # Note: Ensure "nnUNet_raw" capitalization matches your system (sometimes it is nnUNet_raw vs nnunet_raw)
    PATH_2D_GT="${PROJECT_ROOT}/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/labelsTr"

    # 3. 2D MANIFEST (Mapping 2D filenames to 3D Volume + Slice Index)
    MANIFEST_2D_PATH="${PROJECT_ROOT}/datasets/2D_datasets/train_datasets/processed_data_full_multichannel/nnunet_raw/Dataset700_MagPhaseExp_simple_training_base/inference_manifest.json"
fi

# 4. OUTPUT ROOT
PATH_RESULTS="${PROJECT_ROOT}/datasets/3D_datasets/train_datasets/nnunet_dataset/nnUNet_results"

# --- Logic Loop ---
EXPERIMENT_KEYS=("${!EXPERIMENT_MAP[@]}")

# Calculate the number of experiments
NUM_EXPERIMENTS=${#EXPERIMENT_KEYS[@]}

# Check if experiments exist
if [ $NUM_EXPERIMENTS -eq 0 ]; then
    echo "Error: No experiments found in EXPERIMENT_MAP."
    exit 1
fi

# --- SINGLE VOLUME MODE ---
if [ "$MODE" = "single_volume" ]; then
    echo "🚨 RUNNING IN SINGLE VOLUME MODE 🚨"
    
    # We only need one experiment definition for this mode
    DATASET_ID="${EXPERIMENT_KEYS[0]}"
    EXPERIMENT="${EXPERIMENT_MAP[$DATASET_ID]}"
    TASK_NAME="MagPhase_${EXPERIMENT}" 
    CURRENT_RAW_DATA_PATH="${NNUNET_RAW_ROOT}/Dataset1702_MagPhase_mag_and_phase" 
    MANIFEST_3D_PATH="${CURRENT_RAW_DATA_PATH}/manifest.json"
    MODEL_FOLDER="${PATH_RESULTS}/Dataset${DATASET_ID}_${TASK_NAME}/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"
    OUTPUT_ROOT="${MODEL_FOLDER}/inference_results/${EXPERIMENT}_${MODE}_EVAL"

    if [ ! -f "$SINGLE_VOLUME_PATH" ]; then
        echo "❌ Error: SINGLE_VOLUME_PATH not found: $SINGLE_VOLUME_PATH"
        exit 1
    fi

    GPU_ID="${GPU_IDS[0]}" # Use the first available GPU
    LOG_DIR="${OUTPUT_ROOT}/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="${LOG_DIR}/inference_single_vol.log"

    echo "========================================================"
    echo "🚀 Exp: $EXPERIMENT | SINGLE VOLUME | GPU: $GPU_ID"
    echo "   Volume: $SINGLE_VOLUME_PATH"
    echo "   Log: $LOG_FILE"
    echo "========================================================"

    CMD="python inference_3d.py \
        -experiment \"$EXPERIMENT\" \
        -model_folder \"$MODEL_FOLDER\" \
        -mode \"$MODE\" \
        -input_path_raw \"$CURRENT_RAW_DATA_PATH\" \
        -gt_path_2d \"$PATH_2D_GT\" \
        -manifest_2d \"$MANIFEST_2D_PATH\" \
        -manifest_3d \"$MANIFEST_3D_PATH\" \
        -folds ${FOLDS[*]} \
        -output_root \"$OUTPUT_ROOT\" \
        --save_predictions \
        --use_tta \"$USE_TTA\" \
        --ensemble \"$ENSEMBLE\" \
        --single_volume_path \"$SINGLE_VOLUME_PATH\""

    FULL_CMD="source ${VENV_PATH}/bin/activate && $CMD"
    set_slot "$GPU_ID" bash -c "$FULL_CMD"

    echo "⏳ Waiting for single volume inference to finish..."
    wait
    echo "✅ Single volume inference completed."
    exit 0
fi


# Global counter for round-robin distribution across all experiments and folds
GLOBAL_CTR=0

for (( batch_start=0; batch_start<NUM_EXPERIMENTS; batch_start+=MAX_PARALLEL_EXPERIMENTS )); do
    echo "--- Batch Processing: Starting experiments from index $batch_start ---"
    for (( i=batch_start; i<batch_start+MAX_PARALLEL_EXPERIMENTS && i<NUM_EXPERIMENTS; i++ )); do
        DATASET_ID="${EXPERIMENT_KEYS[$i]}"
    EXPERIMENT="${EXPERIMENT_MAP[$DATASET_ID]}"
    
    TASK_NAME="MagPhase_${EXPERIMENT}" 
    
    if [ "$MODE" = "test" ]; then
        CURRENT_RAW_DATA_PATH="${NNUNET_RAW_ROOT}"
        # Dummy 3D manifest to pass check (Python script ignores it in test mode)
        MANIFEST_3D_PATH="${MANIFEST_2D_PATH}"
    else
        # CURRENT_RAW_DATA_PATH="${NNUNET_RAW_ROOT}/Dataset${DATASET_ID}_${TASK_NAME}"
        CURRENT_RAW_DATA_PATH="${NNUNET_RAW_ROOT}/Dataset1702_MagPhase_mag_and_phase" 
        MANIFEST_3D_PATH="${CURRENT_RAW_DATA_PATH}/manifest.json"
    fi

    MODEL_FOLDER="${PATH_RESULTS}/Dataset${DATASET_ID}_${TASK_NAME}/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"
    OUTPUT_ROOT="${MODEL_FOLDER}/inference_results/${EXPERIMENT}_${MODE}_EVAL"

    if [ ! -f "$MANIFEST_3D_PATH" ] && [ "$MODE" != "test" ]; then
        echo "❌ Error: 3D Manifest not found at $MANIFEST_3D_PATH "
        continue
    fi

    # Create logs directory
    LOG_DIR="${OUTPUT_ROOT}/logs"
    mkdir -p "$LOG_DIR"

    if [ "$ENSEMBLE" = "True" ] || [ "$ENSEMBLE" = "true" ]; then
        # --- ENSEMBLE MODE (SINGLE JOB, ALL FOLDS) ---
        # Cannot distribute folds because ensemble requires them in one process.
        
        GPU_INDEX=$((GLOBAL_CTR % NUM_GPUS))
        GPU_ID="${GPU_IDS[$GPU_INDEX]}"
        GLOBAL_CTR=$((GLOBAL_CTR + 1))
        
        LOG_FILE="${LOG_DIR}/inference_ensemble.log"
        
        echo "========================================================"
        echo "🚀 Exp: $EXPERIMENT | ENSEMBLE MODE | GPU: $GPU_ID"
        echo "   Log: $LOG_FILE"
        echo "========================================================"

        CMD="python inference_3d.py \
            -experiment \"$EXPERIMENT\" \
            -model_folder \"$MODEL_FOLDER\" \
            -mode \"$MODE\" \
            -input_path_raw \"$CURRENT_RAW_DATA_PATH\" \
            -gt_path_2d \"$PATH_2D_GT\" \
            -manifest_2d \"$MANIFEST_2D_PATH\" \
            -manifest_3d \"$MANIFEST_3D_PATH\" \
            -folds ${FOLDS[*]} \
            -output_root \"$OUTPUT_ROOT\" \
            --save_predictions \
            --use_tta \"$USE_TTA\" \
            --ensemble \"$ENSEMBLE\" \
            --no_summary \
            --slice_axis 0"

        FULL_CMD="source ${VENV_PATH}/bin/activate && $CMD"
        set_slot "$GPU_ID" bash -c "$FULL_CMD" &

    else
        # --- STANDARD MODE (DISTRIBUTED FOLDS) ---
        for FOLD in "${FOLDS[@]}"; do
            # Round-robin GPU assignment
            GPU_INDEX=$((GLOBAL_CTR % NUM_GPUS))
            GPU_ID="${GPU_IDS[$GPU_INDEX]}"
            GLOBAL_CTR=$((GLOBAL_CTR + 1))

            # Log file for this specific fold
            LOG_FILE="${LOG_DIR}/inference_fold${FOLD}.log"

            echo "========================================================"
            echo "🚀 Exp: $EXPERIMENT | Fold: $FOLD | GPU: $GPU_ID"
            echo "   Log: $LOG_FILE"
            echo "========================================================"

            CMD="python inference_3d.py \
                -experiment \"$EXPERIMENT\" \
                -model_folder \"$MODEL_FOLDER\" \
                -mode \"$MODE\" \
                -input_path_raw \"$CURRENT_RAW_DATA_PATH\" \
                -gt_path_2d \"$PATH_2D_GT\" \
                -manifest_2d \"$MANIFEST_2D_PATH\" \
                -manifest_3d \"$MANIFEST_3D_PATH\" \
                -folds $FOLD \
                -output_root \"$OUTPUT_ROOT\" \
                --save_predictions \
                --use_tta \"$USE_TTA\" \
                --ensemble \"$ENSEMBLE\" \
                --no_summary \
                --slice_axis 0"
                
            FULL_CMD="source ${VENV_PATH}/bin/activate && $CMD"

            # Execute in background via set_slot
            set_slot "$GPU_ID" bash -c "$FULL_CMD" &
        done
    fi
    done
    
    # Wait for the current batch of experiments to complete
    echo "⏳ Waiting for batch to finish..."
    wait
done

echo "✅ All inference jobs completed."

# --- SUMMARY AGGREGATION LOOP ---
echo "📊 Generating Cross-Fold Summaries..."

for (( i=0; i<NUM_EXPERIMENTS; i++ )); do
    DATASET_ID="${EXPERIMENT_KEYS[$i]}"
    EXPERIMENT="${EXPERIMENT_MAP[$DATASET_ID]}"
    TASK_NAME="MagPhase_${EXPERIMENT}" 
    MODEL_FOLDER="${PATH_RESULTS}/Dataset${DATASET_ID}_${TASK_NAME}/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"
    OUTPUT_ROOT="${MODEL_FOLDER}/inference_results/${EXPERIMENT}_${MODE}_EVAL" # Same as above
    
    # Re-define required vars for the python script (it needs them even for summary mode)
    if [ "$MODE" = "test" ]; then
        CURRENT_RAW_DATA_PATH="${NNUNET_RAW_ROOT}"
        MANIFEST_3D_PATH="${MANIFEST_2D_PATH}"
    else
        # CURRENT_RAW_DATA_PATH="${NNUNET_RAW_ROOT}/Dataset${DATASET_ID}_${TASK_NAME}"
        CURRENT_RAW_DATA_PATH="${NNUNET_RAW_ROOT}/Dataset1702_MagPhase_mag_and_phase" 
        MANIFEST_3D_PATH="${CURRENT_RAW_DATA_PATH}/manifest.json" # Needed for parsing logic in init
    fi

    # We run this on the first available GPU (or CPU if possible, but script uses cuda check)
    # Using GPU 0 for summary is safe as inference is done.
    GPU_ID="${GPU_IDS[0]}"
    
    SUMMARY_CMD="python inference_3d.py \
        -experiment \"$EXPERIMENT\" \
        -model_folder \"$MODEL_FOLDER\" \
        -mode \"$MODE\" \
        -input_path_raw \"$CURRENT_RAW_DATA_PATH\" \
        -gt_path_2d \"$PATH_2D_GT\" \
        -manifest_2d \"$MANIFEST_2D_PATH\" \
        -manifest_3d \"$MANIFEST_3D_PATH\" \
        -folds ${FOLDS[*]} \
        -output_root \"$OUTPUT_ROOT\" \
        --save_predictions \
        --use_tta \"$USE_TTA\" \
        --ensemble \"$ENSEMBLE\" \
        --summary_only"

    FULL_SUMMARY_CMD="source ${VENV_PATH}/bin/activate && $SUMMARY_CMD"
    
    echo "   Aggregating $EXPERIMENT..."
    # We can run summary sequentially per experiment
    set_slot "$GPU_ID" bash -c "$FULL_SUMMARY_CMD"
done

echo "🎉 All Done."