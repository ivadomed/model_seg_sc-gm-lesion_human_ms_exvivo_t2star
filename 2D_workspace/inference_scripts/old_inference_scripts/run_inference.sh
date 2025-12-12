# set -e

# echo "--- ✅ Successfully running commands inside the allocated slot ---"
# echo "--- Using GPU: $CUDA_VISIBLE_DEVICES ---"

# cd /home/ge.polymtl.ca/pahoa/nih_project
# source .venv/bin/activate
# cd /home/ge.polymtl.ca/pahoa/nih_project/model_seg_sc-gm-lesion_human_ms_exvivo_t2star/training_scripts


# export PATH_DATA="/home/ge.polymtl.ca/pahoa/nih_project/datasets/bids_test_dataset"
# export PATH_PROCESSED="/home/ge.polymtl.ca/pahoa/nih_project/datasets/processed_test_set"

# export DATASET_ID=522
# export TASK_NAME="MagPhaseExp_simple_training_ostu_only_(base)_2"

# # Derivative and output paths for the new workflow
# export SLICED_DATA_DIR="${PATH_PROCESSED}/data_slice"
# export ORIGINAL_LABELS_DIR="${PATH_DATA}/derivatives/labels" # Path to original SC, GM, lesion labels

# # nnU-Net paths
# export nnUNet_raw="${PATH_PROCESSED}/nnunet_raw"
# export nnUNet_preprocessed="${PATH_PROCESSED}/nnunet_preprocessed"
# export nnUNet_results="${PATH_PROCESSED}/nnunet_results"


# DATASET_PATH="${nnUNet_raw}/Dataset${DATASET_ID}_${TASK_NAME}"

# echo "Step 2: Running inference on the entire test/val set..."

# # Assuming you have a trained model from a previous run you want to use for inference.
# export TRAINED_MODEL_FOLDER="/home/ge.polymtl.ca/pahoa/nih_project/datasets/processed_data_full_multichannel/nnunet_results/Dataset${DATASET_ID}_${TASK_NAME}/nnUNetTrainerWandb__nnUNetPlans__2d/"

# # Define the FULL path to the test set you just created.
# export TEST_SET_FULL_PATH="${nnUNet_raw}/Dataset${DATASET_ID}_${TASK_NAME}"


# FOLD=4

# # # Define a clear output directory for the inference results.
# # export INFERENCE_OUTPUT_DIR="${TRAINED_MODEL_FOLDER}/fold_${FOLD}/test_set_inference_on_ds${DATASET_ID}/"


# # # Call the script with the new, simpler argument.
# # CUDA_VISIBLE_DEVICES=1 python run_inference.py \
# #   --model_folder "$TRAINED_MODEL_FOLDER" \
# #   --test_set_path "$TEST_SET_FULL_PATH" \
# #   --output_folder "$INFERENCE_OUTPUT_DIR" \
# #   --gpu_id 1 \
# #   --fold $FOLD


# # 2. Define the path to the ORIGINAL processed data directory where that model was trained.
# #    This is where the script will find the 'splits_final.json' file.
# export ORIGINAL_PROCESSED_DATA="/home/ge.polymtl.ca/pahoa/nih_project/datasets/processed_data_full_multichannel"

# # 3. Define a clear output directory.
# export VALIDATION_OUTPUT_DIR="${TRAINED_MODEL_FOLDER}/fold_${FOLD}/validation_run_outputs"


# # 4. Call the script.
# # python run_inference.py \
# #   --model_folder "$TRAINED_MODEL_FOLDER" \
# #   --path_processed "$ORIGINAL_PROCESSED_DATA" \
# #   --output_folder "$VALIDATION_OUTPUT_DIR" \
# #   --gpu_id 1 \
# #   --fold $FOLD

# VOLUME_FOLDER_OUTPUT_DIR="${TRAINED_MODEL_FOLDER}/fold_${FOLD}/volume_output_results"
# VOLUME_MAG_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/bids_test_dataset/sub-TNU025/anat/S2_76.2cm_T2s_75i_TR45TE9_cor_20avg_Redo_2100_part-mag.nii.gz"
# VOLUME_PHASE_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/bids_test_dataset/sub-TNU025/anat/S2_76.2cm_T2s_75i_TR45TE9_cor_20avg_Redo_2100_part-phase.nii.gz"

# python run_inference.py \
#     --mode volume \
#     --model_folder $TRAINED_MODEL_FOLDER \
#     --volume_mag_path $VOLUME_MAG_PATH \
#     --volume_phase_path $VOLUME_PHASE_PATH \
#     --output_folder $VOLUME_FOLDER_OUTPUT_DIR \
#     --gpu_id 0 \
#     --fold $FOLD


# echo "--- 🎉 Inference complete. Results are in $INFERENCE_OUTPUT_DIR ---"


#!/bin/bash
set -e

# ==============================================================================
# ⚙️  CONFIGURATION SECTION
# ==============================================================================

# 1. MODE: "validation", "test", or "volume"
MODE="validation"

# 2. EXPERIMENT: Name of the experiment settings
EXPERIMENT="exp_base"

# 3. FOLDS: Which folds to process?
# Default to all 5 folds for robust evaluation
FOLDS=(3 4)

# 4. PATHS
GPU_ID=1
DATASET_ID=700
TASK_NAME="MagPhaseExp_simple_training_base"

PROJECT_ROOT="/home/ge.polymtl.ca/pahoa/nih_project"
PATH_PROCESSED_TRAIN="${PROJECT_ROOT}/datasets/processed_data_full_multichannel"
PATH_PROCESSED_TEST="${PROJECT_ROOT}/datasets/processed_test_set"
TRAINED_MODEL_FOLDER="${PATH_PROCESSED_TRAIN}/nnunet_results/Dataset${DATASET_ID}_${TASK_NAME}/nnUNetTrainerWandb__nnUNetPlans__2d/"

# Base folder for all results of this experiment run
# Result structure: inference_results/exp_name/fold_X/...
GLOBAL_OUTPUT_ROOT="${TRAINED_MODEL_FOLDER}/inference_results/${EXPERIMENT}_FULL_EVAL"

# ==============================================================================
# 🚀 EXECUTION
# ==============================================================================

echo "--- 🚀 Starting Multi-Fold Pipeline ---"
echo "   Experiment: $EXPERIMENT"
echo "   Mode:       $MODE"
echo "   Folds:      ${FOLDS[*]}"
echo "   Output:     $GLOBAL_OUTPUT_ROOT"

source "${PROJECT_ROOT}/.venv/bin/activate"
cd "${PROJECT_ROOT}/model_seg_sc-gm-lesion_human_ms_exvivo_t2star/training_scripts"

# Loop through folds
for FOLD in "${FOLDS[@]}"; do
    echo ""
    echo "================================================="
    echo "▶️  Processing FOLD $FOLD"
    echo "================================================="

    # Define specific fold output folder
    # Structure: inference_results/exp_full_eval/exp_name_foldX/validation_plots
    FOLD_OUTPUT_BASE="${GLOBAL_OUTPUT_ROOT}/${EXPERIMENT}_fold${FOLD}"

    if [ "$MODE" == "validation" ]; then
        OUTPUT_DIR="${FOLD_OUTPUT_BASE}/validation_plots"
        python run_inference.py \
            --mode plot \
            --experiment "$EXPERIMENT" \
            --model_folder "$TRAINED_MODEL_FOLDER" \
            --path_processed "$PATH_PROCESSED_TRAIN" \
            --output_folder "$OUTPUT_DIR" \
            --gpu_id "$GPU_ID" \
            --fold "$FOLD" \
            --debug \

    elif [ "$MODE" == "test" ]; then
        OUTPUT_DIR="${FOLD_OUTPUT_BASE}/test_set_plots"
        TEST_SET_FULL_PATH="${PATH_PROCESSED_TEST}/nnunet_raw/Dataset${DATASET_ID}_${TASK_NAME}"
        
        python run_inference.py \
            --mode plot \
            --experiment "$EXPERIMENT" \
            --model_folder "$TRAINED_MODEL_FOLDER" \
            --test_set_path "$TEST_SET_FULL_PATH" \
            --output_folder "$OUTPUT_DIR" \
            --gpu_id "$GPU_ID" \
            --fold "$FOLD"
    fi
done

# --- AGGREGATION STEP ---
if [ "$MODE" != "volume" ]; then
    echo ""
    echo "================================================="
    echo "📊 Generating Final Experiment Report"
    echo "================================================="
    
    # We pass the root folder. The script will look for {EXPERIMENT}_fold{X} subfolders inside it.
    python run_inference.py \
        --mode aggregate \
        --experiment "$EXPERIMENT" \
        --output_folder "$GLOBAL_OUTPUT_ROOT" \
        --folds_to_aggregate "${FOLDS[@]}"

    echo ""
    echo "🎉 Done! Final report saved in: $GLOBAL_OUTPUT_ROOT"
fi