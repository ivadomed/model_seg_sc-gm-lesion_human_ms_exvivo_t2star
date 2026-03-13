#!/bin/bash

# --- Configuration ---
GPU_ID=0
MODE="validation" # Options: validation, test, volume
FOLDS=(0 1 2 3)

# Define your experiments here: [DATASET_ID]="EXPERIMENT_NAME"
declare -A EXPERIMENT_MAP
EXPERIMENT_MAP=( 
    # ["700"]="base" 
    # ["701"]="soft_loss"
    # ["702"]="mosaic" ## never do mosaic, because of the sliding window, it doesn't make much sense
    # ["703"]="prepro_mag"
    # ["704"]="phase_prepro"
    # ["705"]="mag_prepro"
    # ["706"]="no_spatial_aug"
    # # ["707"]="merge_lesions"


    # ["708"]="base_no_otsu"
    # # ["709"]="wm_gm_segmentation"
    # ["710"]="mag-one-channel"
    # ["711"]="soft_loss_fixed"
    # ["712"]="soft_loss_2_fixed"

    # ["713"]="soft_loss_3_fixed"
    # ["714"]="spatial_aug_2"
    # ["715"]="spatial_aug_3"
    # ["716"]="sgd_optimizer"
    # ["717"]="winning_combination" ## aug 2 and soft 2, do not yield the best results when combiend!!
    ["stacking"]="stacking"
)

export nnUNet_raw=""
export nnUNet_preprocessed=""
export nnUNet_results=""
# --- Stacking Configuration ---
# Define the dataset IDs for the stacking experiment
# First model should be the one segmenting WM and GM
# Second model should be the one segmenting lesions
STACKING_DS_ID_1="709"
STACKING_TASK_NAME_1="wm_gm_segmentation"
STACKING_DS_ID_2="707"
STACKING_TASK_NAME_2="merge_lesions"


# --- Path Definitions ---
PROJECT_ROOT="/home/ge.polymtl.ca/pahoa/nih_project"
PATH_PROCESSED_TRAIN="${PROJECT_ROOT}/datasets/2D_datasets/train_datasets/processed_data_full_multichannel"
PATH_PROCESSED_TEST="${PROJECT_ROOT}/datasets/2D_datasets/test_datasets/processed_test_set_dynamic"


# 📂 VOLUME MODE CONFIGURATION
# 1. The common root folder for your specific volumes
VOLUME_ROOT_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset" 

# 2. The list of relative paths (folders) inside that root
#    These are the subfolders you want to preserve in the output.
MANUAL_VOLUME_LIST=(
    "sub-PML014/anat/S0_68.4cm_T2s_75i_TR45TE9_cor_18avg_Redo_3200_part-mag.nii.gz"
    "sub-PML014/anat/S1_73.4cm_T2s_75i_TR45TE9_cor_21avg_Redo_3500_part-mag.nii.gz"
    "sub-PML014/anat/S2_78.4cm_T2s_75i_TR45TE9_cor_18avg_1900_part-mag.nii.gz"
    "sub-PML014/anat/S3_83.4cm_T2s_75i_TR45TE9_cor_12avg_2200_part-mag.nii.gz"
    "sub-PML014/anat/S4_88.4cm_T2s_75i_TR45TE9_cor_21avg_2600_part-mag.nii.gz"
    "sub-PML019/anat/S0_66.0cm_T2s_75i_TR45TE9_cor_18avg_600_part-mag.nii.gz"
    "sub-PML019/anat/S1_71.0cm_T2s_75i_TR45TE9_cor_18avg_900_part-mag.nii.gz"
    "sub-PML019/anat/S2_76.0cm_T2s_75i_TR45TE9_cor_15avg_2500_part-mag.nii.gz"
    "sub-PML019/anat/S3_81.0cm_T2s_75i_TR45TE9_cor_12avg_1500_part-mag.nii.gz"
    "sub-PML019/anat/S4_86.0cm_T2s_75i_TR45TE9_cor_12avg_4700_part-mag.nii.gz"
    "sub-PML023/anat/S0_81.5cm_T2s_75i_TR45TE9_cor_12avg_1900_part-mag.nii.gz"
    "sub-PML023/anat/S1_86.5cm_T2s_75i_TR45TE9_cor_18avg_1000_part-mag.nii.gz"
    "sub-PML023/anat/S2_91.5cm_T2s_75i_TR45TE9_cor_15avg_1400_part-mag.nii.gz"
    "sub-PML024/anat/S0_65.0cm_T2s_75i_TR45TE9_cor_15avg_500_part-mag.nii.gz"
    "sub-PML024/anat/S1_70.0cm_T2s_75i_TR45TE9_cor_12avg_1500_part-mag.nii.gz"
    "sub-PML024/anat/S2_75.0cm_T2s_75i_TR45TE9_cor_15avg_2000_part-mag.nii.gz"
    "sub-PML024/anat/S3_80.0cm_T2s_75i_TR45TE9_cor_15avg_2300_part-mag.nii.gz"
    "sub-PML024/anat/S4_85.0cm_T2s_75i_TR45TE9_cor_15avg_3500_part-mag.nii.gz"
    "sub-TNU003/anat/S0_68.0cm_T2s_75i_TR45TE9_cor_19avg_60001_1600_part-mag.nii.gz"
    "sub-TNU003/anat/S1_73.0cm_T2s_75i_TR45TE9_cor_16avg_1200_part-mag.nii.gz"
    "sub-TNU003/anat/S2_78.0cm_T2s_75i_TR45TE9_cor_15avg_2200_part-mag.nii.gz"
    "sub-TNU003/anat/S3_83.0cm_T2s_75i_TR45TE9_cor_16avg_2500_part-mag.nii.gz"
    "sub-TNU003/anat/S4_88.0cm_T2s_75i_TR45TE9_cor_15avg_3000_part-mag.nii.gz"
    "sub-TNU003/anat/S5_93.0cm_T2s_75i_TR45TE9_cor_16avg_3300_part-mag.nii.gz"
    "sub-TNU003/anat/S6_98.0cm_T2s_75i_TR45TE9_cor_14avg_3300_part-mag.nii.gz"
    "sub-TNU004/anat/S0_66.0cm_T2s_75i_TR45TE9_cor_12avg_500_part-mag.nii.gz"
    "sub-TNU004/anat/S1_71.0cm_T2s_75i_TR45TE9_cor_21avg_800_part-mag.nii.gz"
    "sub-TNU004/anat/S2_76.0cm_T2s_75i_TR45TE9_cor_28avg_2000_part-mag.nii.gz"
    "sub-TNU004/anat/S3_81.0cm_T2s_75i_TR45TE9_cor_12avg_1500_part-mag.nii.gz"
    "sub-TNU004/anat/S4_86.0cm_T2s_75i_TR45TE9_cor_12avg_2300_part-mag.nii.gz"
    "sub-TNU004/anat/S5_91.0cm_T2s_75i_TR45TE9_cor_18avg_2600_part-mag.nii.gz"
    "sub-TNU004/anat/S6_96.0cm_T2s_75i_TR45TE9_cor_12avg_3100_part-mag.nii.gz"
    "sub-TNU008/anat/S0_68.5cm_T2s_75i_TR45TE9_cor_20avg_300_part-mag.nii.gz"
    "sub-TNU008/anat/S1_73.5cm_T2s_75i_TR45TE9_cor_24avg_800_part-mag.nii.gz"
    "sub-TNU008/anat/S2_78.5cm_T2s_75i_TR45TE9_cor_20avg_1100_part-mag.nii.gz"
    "sub-TNU008/anat/S3_83.5cm_T2s_75i_TR45TE9_cor_20avg_1800_part-mag.nii.gz"
    "sub-TNU008/anat/S4_88.5cm_T2s_75i_TR45TE9_cor_12avg_2900_part-mag.nii.gz"
    "sub-TNU008/anat/S5_93.5cm_T2s_75i_TR45TE9_cor_16avg_3100_part-mag.nii.gz"
    "sub-TNU008/anat/S6_98.2cm_T2s_75i_TR45TE9_cor_12avg_4000_part-mag.nii.gz"
    "sub-TNU009/anat/S0_T2s_75i_TR45TE9_cor_21avg_part-mag.nii.gz"
    "sub-TNU009/anat/S1_T2s_100i_TR25TE9_cor_12avg_part-mag.nii.gz"
    "sub-TNU009/anat/S2_T2s_75i_TR45TE9_cor_18avg_part-mag.nii.gz"
    "sub-TNU009/anat/S3_T2s_75i_TR45TE9_cor_19avg_part-mag.nii.gz"
    "sub-TNU009/anat/S4_T2s_100i_TR25TE9_cor_7avg_part-mag.nii.gz"
    "sub-TNU010/anat/S0_65.0cm_T2s_75i_TR45TE9_cor_15avg_500_part-mag.nii.gz"
    "sub-TNU010/anat/S1_70.1cm_T2s_75i_TR45TE9_cor_18avg_1700_part-mag.nii.gz"
    "sub-TNU010/anat/S2_75.1cm_T2s_75i_TR45TE9_cor_18avg_2000_part-mag.nii.gz"
    "sub-TNU010/anat/S3_80.1cm_T2s_75i_TR45TE9_cor_18avg_2300_part-mag.nii.gz"
    "sub-TNU010/anat/S4_85.1cm_T2s_75i_TR45TE9_cor_18avg_3300_part-mag.nii.gz"
    "sub-TNU012/anat/S0_68.5cm_T2s_75i_TR45TE9_cor_12avg_600_part-mag.nii.gz"
    "sub-TNU012/anat/S1_73.5cm_T2s_75i_TR45TE9_cor_12avg_1000_part-mag.nii.gz"
    "sub-TNU012/anat/S2_78.5cm_T2s_75i_TR45TE9_cor_15avg_Redo_1900_part-mag.nii.gz"
    "sub-TNU012/anat/S3_83.5cm_T2s_75i_TR45TE9_cor_12avg_Redo_3100_part-mag.nii.gz"
    "sub-TNU012/anat/S4_88.5cm_T2s_75i_TR45TE9_cor_15avg_Redo_2600_part-mag.nii.gz"
    "sub-TNU014/anat/S0_63.0cm_T2s_75i_TR45TE9_cor_12avg_600_part-mag.nii.gz"
    "sub-TNU014/anat/S1_68.0cm_T2s_75i_TR45TE9_cor_12avg_900_part-mag.nii.gz"
    "sub-TNU014/anat/S2_73.0cm_T2s_75i_TR45TE9_cor_15avg_1200_part-mag.nii.gz"
    "sub-TNU014/anat/S3_78.0cm_T2s_75i_TR45TE9_cor_12avg_1700_part-mag.nii.gz"
    "sub-TNU014/anat/S4_83.0cm_T2s_75i_TR45TE9_cor_12avg_2000_part-mag.nii.gz"
    "sub-TNU014/anat/S5_88.0cm_T2s_75i_TR45TE9_cor_18avg_2300_part-mag.nii.gz"
    "sub-TNU014/anat/S6_92.0cm_T2s_75i_TR45TE9_cor_21avg_2700_part-mag.nii.gz"
    "sub-TNU017/anat/S0_65.0cm_T2s_75i_TR45TE9_cor_15avg_400_part-mag.nii.gz"
    "sub-TNU017/anat/S1_70.0cm_T2s_75i_TR45TE9_cor_12avg_700_part-mag.nii.gz"
    "sub-TNU017/anat/S2_75.7cm_T2s_75i_TR45TE9_cor_12avg_2800_part-mag.nii.gz"
    "sub-TNU017/anat/S3_80.7cm_T2s_75i_TR45TE9_cor_12avg_3600_part-mag.nii.gz"
)

# --- Logic Loop ---

for DATASET_ID in "${!EXPERIMENT_MAP[@]}"; do
    EXPERIMENT="${EXPERIMENT_MAP[$DATASET_ID]}"
    
    # Define Model Folder
    if [ "$EXPERIMENT" == "stacking" ]; then
        TASK_NAME_1="MagPhaseExp_simple_training_${STACKING_TASK_NAME_1}"
        MODEL_FOLDER_1="${PATH_PROCESSED_TRAIN}/nnunet_results/Dataset${STACKING_DS_ID_1}_${TASK_NAME_1}/nnUNetTrainerWandb__nnUNetPlans__2d"
        TASK_NAME_2="MagPhaseExp_simple_training_${STACKING_TASK_NAME_2}"
        MODEL_FOLDER_2="${PATH_PROCESSED_TRAIN}/nnunet_results/Dataset${STACKING_DS_ID_2}_${TASK_NAME_2}/nnUNetTrainerWandb__nnUNetPlans__2d"
        TRAINED_MODEL_FOLDER=$MODEL_FOLDER_1 # Use first model for output path
    else
        TASK_NAME="MagPhaseExp_simple_training_${EXPERIMENT}"
        TRAINED_MODEL_FOLDER="${PATH_PROCESSED_TRAIN}/nnunet_results/Dataset${DATASET_ID}_${TASK_NAME}/nnUNetTrainerWandb__nnUNetPlans__2d"
    fi
    
    echo "========================================================"
    echo "🚀 Experiment: $EXPERIMENT (ID: $DATASET_ID)"
    echo "========================================================"

    # --- Mode Switching Logic ---
    
    # We define arrays for the loop based on the mode.
    # We need parallel arrays: one for the full input path, one for the relative structure.
    declare -a PROCESS_PATHS
    declare -a RELATIVE_STRUCTURES
    
    # 1. VOLUME MODE (External lists)
    if [ "$MODE" == "volume" ]; then
        # Define where the outputs for this specific batch of volumes should live
        # e.g. /.../nnunet_results/.../inference_results/EXPERIMENT_EXTERNAL_VOLUMES
        OUTPUT_ROOT_NAME="${EXPERIMENT}_EXTERNAL_VOLUMES"

        for rel in "${MANUAL_VOLUME_LIST[@]}"; do
            FULL_PATH="${VOLUME_ROOT_PATH}/${rel}"
            
            if [ -e "$FULL_PATH" ]; then
                PROCESS_PATHS+=("$FULL_PATH")
                
                # Check if it's a file, if so, use the directory name for the output structure
                if [ -f "$FULL_PATH" ]; then
                    DIR_NAME=$(dirname "$rel")
                    RELATIVE_STRUCTURES+=("$DIR_NAME")
                else
                    RELATIVE_STRUCTURES+=("$rel")
                fi
            else
                echo "⚠️  Skipping missing path: $FULL_PATH"
            fi
        done

    # 2. VALIDATION MODE (Standard Dataset)
    elif [ "$MODE" == "validation" ]; then
        OUTPUT_ROOT_NAME="${EXPERIMENT}_FULL_EVAL"
        PROCESS_PATHS=("$PATH_PROCESSED_TRAIN")
        RELATIVE_STRUCTURES=("") # No sub-folder structure needed inside the eval folder

    # 3. TEST MODE (Standard Dataset)
    elif [ "$MODE" == "test" ]; then
        OUTPUT_ROOT_NAME="${EXPERIMENT}_TEST_SET"
        # Point to the specific imagesTr folder inside the dynamic test set structure
        PROCESS_PATHS=("$PATH_PROCESSED_TEST/nnunet_raw/Dataset520_TEST_SET_AXIAL/imagesTr")
        RELATIVE_STRUCTURES=("") 

    else
        echo "Error: Invalid mode '$MODE'."
        exit 1
    fi

    # --- Execution Loop ---
    
    # Loop through indices of the array
    for i in "${!PROCESS_PATHS[@]}"; do
        CURRENT_INPUT_PATH="${PROCESS_PATHS[$i]}"
        CURRENT_REL_PATH="${RELATIVE_STRUCTURES[$i]}"
        
        # We pass the root output folder to Python. 
        # Python will join: GLOBAL_OUTPUT_ROOT + CURRENT_REL_PATH
        GLOBAL_OUTPUT_ROOT="${TRAINED_MODEL_FOLDER}/inference_results/${OUTPUT_ROOT_NAME}"

        echo "   👉 Processing: $CURRENT_REL_PATH"
        echo "      Input: $CURRENT_INPUT_PATH"
        echo "      Output Base: $GLOBAL_OUTPUT_ROOT"

        if [ "$EXPERIMENT" == "stacking" ]; then
            CUDA_VISIBLE_DEVICES=$GPU_ID python inference_2.py \
                -experiment "$EXPERIMENT" \
                -model_folder "$MODEL_FOLDER_1" \
                -model_folder_2 "$MODEL_FOLDER_2" \
                -mode "$MODE" \
                -path "$CURRENT_INPUT_PATH" \
                -folds "${FOLDS[@]}" \
                -output_root "$GLOBAL_OUTPUT_ROOT" \
                -rel_path "$CURRENT_REL_PATH" \
                # --ensemble \
                # --use_tta
        else
            CUDA_VISIBLE_DEVICES=$GPU_ID python inference_2.py \
                -experiment "$EXPERIMENT" \
                -model_folder "$TRAINED_MODEL_FOLDER" \
                -mode "$MODE" \
                -path "$CURRENT_INPUT_PATH" \
                -folds "${FOLDS[@]}" \
                -output_root "$GLOBAL_OUTPUT_ROOT" \
                -rel_path "$CURRENT_REL_PATH" 
                # --use_tta \
                # --ensemble
        fi

        echo "   ✅ Done."
    done
    
    # Clean up arrays for next experiment iteration
    unset PROCESS_PATHS
    unset RELATIVE_STRUCTURES

    echo "🏁 Finished Experiment: $EXPERIMENT"
done

echo "--------------------------------------------------------"
echo "--- 🎉 All Experiments Completed Successfully ---"