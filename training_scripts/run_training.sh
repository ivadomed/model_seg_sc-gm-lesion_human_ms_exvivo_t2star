
#!/bin/bash


# export PATH_DATA="/Users/homefolder/Local/NeuroPoly/project1_seg_lesion/datasets/Processed_BIDS_full"

# python combine_labels.py --path-label-in ${PATH_DATA}/derivatives/labels --path-label-out ${PATH_DATA}/derivatives/labels_combined-sc-gm --suffixes SC GM lesion --priors GM:SC lesion:SC

# export PATH_PROCESSED="/Users/homefolder/Local/NeuroPoly/project1_seg_lesion/datasets/processed_data_full"

# python extract_slices.py --path-data ${PATH_DATA} --label-folder derivatives/labels_combined-sc-gm --labels combined --path-out ${PATH_PROCESSED}/data_slice

# python convert_bids_to_nnunet.py --path-data ${PATH_PROCESSED}/data_slice --label-json ${PATH_DATA}/derivatives/labels_combined-sc-gm/sub-01/anat/sub-01_part-mag_chunk-02_T2starw_label-combined_classes.json --tasknumber 502

# export nnUNet_raw="${PATH_PROCESSED}/nnunet_raw"
# export nnUNet_preprocessed="${PATH_PROCESSED}/nnunet_preprocessed"
# export nnUNet_results="${PATH_PROCESSED}/nnunet_results"

# nnUNetv2_plan_and_preprocess -d 502 --verify_dataset_integrity




# rm -rf /Users/homefolder/Local/NeuroPoly/project1_seg_lesion/datasets/processed_data_full_multichannel
export PATH_DATA="/home/ge.polymtl.ca/pahoa/nih_project/datasets/bids_dataset"
export PATH_PROCESSED="/home/ge.polymtl.ca/pahoa/nih_project/datasets/processed_data_full_multichannel"
export DATASET_ID=503
export TASK_NAME="MagPhaseExp_no_soft_edges"

# Derivative and output paths for the new workflow
export SLICED_DATA_DIR="${PATH_PROCESSED}/data_slice"
export ORIGINAL_LABELS_DIR="${PATH_DATA}/derivatives/labels" # Path to original SC, GM, lesion labels

# nnU-Net paths
export nnUNet_raw="${PATH_PROCESSED}/nnunet_raw"
export nnUNet_preprocessed="${PATH_PROCESSED}/nnunet_preprocessed"
export nnUNet_results="${PATH_PROCESSED}/nnunet_results"

# --- REMOVED STEP 2 (combine_labels.py) ---

echo "Step 1: Extracting 2D slices for all individual labels..."
python extract_slices_multichannel.py \
  --path-data "$PATH_DATA" \
  --path-out "$SLICED_DATA_DIR" \
  --labels SC GM lesion \
  --label-folder "$ORIGINAL_LABELS_DIR" # Use the original, uncombined labels

echo "Step 2: Converting to nnU-Net format with multi-channel labels..."
python convert_bids_to_nnunet_multichannel.py \
  --path-data "$SLICED_DATA_DIR" \
  --path-out "$nnUNet_raw" \
  --taskname "$TASK_NAME" \
  --tasknumber $DATASET_ID \
  --label-suffixes SC GM lesion \
  # The script will now find the class definitions automatically from the BIDS source

echo "Step 3: Running nnU-Net..."
# These commands remain the same
CUDA_VISIBLE_DEVICES=2 nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train $DATASET_ID 2d 0 --npz -tr nnUNetTrainerWandb
