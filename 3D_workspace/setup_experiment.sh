#!/bin/bash
# Called by the Orchestrator inside set_slot
# Usage: bash setup_experiment.sh <DATASET_ID> <VENV_PATH> <EXP_NAME> <SKIP_BOOL> <PATCH_SIZE>

DATASET_ID=$1
VENV_PATH=$2
EXPERIMENT_NAME=$3
SKIP_PREPROCESSING=$4
CUSTOM_PATCH_SIZE=$5 # Format expected: "x,y,z"

# 1. Activate Environment
source "${VENV_PATH}/bin/activate"
export NNUNET_BASE="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/nnunet_dataset"
export BIDS_DATA_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset"

# 2. Export Paths
export nnUNet_raw="${NNUNET_BASE}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_BASE}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_BASE}/nnUNet_results"
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Standard base dataset (Dual Channel)
BASE_DATASET_NAME="Dataset1702_MagPhase_mag_and_phase"
# Define Target Name immediately for checking
DATASET_NAME="Dataset${DATASET_ID}_MagPhase_${EXPERIMENT_NAME}"
TARGET_PREPRO_DIR="${nnUNet_preprocessed}/${DATASET_NAME}"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"

# --- CHECK FOR MAG-ONLY FLAG ---
if [[ "$EXPERIMENT_NAME" == *"mag-one-channel"* ]]; then
    IS_MAG_ONLY=true
else
    IS_MAG_ONLY=false
fi

if [ "$SKIP_PREPROCESSING" == "true" ]; then
    echo ">> [Setup] Request to skip processing. Checking for existing data..."

    # --- NEW CHECK: DOES PROCESSED DATA ALREADY EXIST? ---
    if [ -d "$TARGET_PREPRO_DIR" ]; then
        echo "✅ [Setup] Found existing preprocessed data at: $TARGET_PREPRO_DIR"
        echo "   Skipping duplication and BIDS conversion."
    else
        echo "⚠️ [Setup] Preprocessed data not found at $TARGET_PREPRO_DIR. Attempting duplication strategy..."

        if [ "$IS_MAG_ONLY" == "true" ]; then
            echo "❌ Error: You requested 'mag-one-channel' but 'SKIP_PREPROCESSING=true' and no preprocessed data exists."
            echo "   The base dataset ($BASE_DATASET_NAME) is 2-channel. You cannot clone it for a 1-channel experiment."
            exit 1
        fi

        echo ">> [Setup] Duplicating base dataset..."
        
        if [ -d "${nnUNet_raw}/${DATASET_NAME}" ]; then
             echo ">> [Setup] Target raw dataset ${DATASET_NAME} already exists. Skipping copy."
        else
            if [ ! -d "${nnUNet_raw}/${BASE_DATASET_NAME}" ]; then
                echo "Error: Base dataset ${nnUNet_raw}/${BASE_DATASET_NAME} does not exist."
                exit 1
            fi

            echo ">> [Setup] Copying ${BASE_DATASET_NAME} to ${DATASET_NAME}..."
            cp -r "${nnUNet_raw}/${BASE_DATASET_NAME}" "${nnUNet_raw}/${DATASET_NAME}"
            cp -r "${nnUNet_preprocessed}/${BASE_DATASET_NAME}" "${nnUNet_preprocessed}/${DATASET_NAME}"
        fi
    fi

    # Update json names
    DATASET_JSON="${nnUNet_raw}/${DATASET_NAME}/dataset.json"
    DATASET_JSON_2="${nnUNet_preprocessed}/${DATASET_NAME}/dataset.json"
    PLANS_JSON="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans.json"
    
    # Simple sed for renaming dataset identifier in files
    if [ -f "$DATASET_JSON" ]; then sed -i "s/\"${BASE_DATASET_NAME}\"/\"${DATASET_NAME}\"/" "$DATASET_JSON"; fi
    if [ -f "$DATASET_JSON_2" ]; then sed -i "s/\"${BASE_DATASET_NAME}\"/\"${DATASET_NAME}\"/" "$DATASET_JSON_2"; fi
    if [ -f "$PLANS_JSON" ]; then sed -i "s/\"${BASE_DATASET_NAME}\"/\"${DATASET_NAME}\"/" "$PLANS_JSON"; fi

else
    echo ">> [Setup] Converting BIDS..."
    python3 preprocess_bids.py \
        --bids "$BIDS_DATA_PATH" \
        --nnunet_raw "$nnUNet_raw" \
        --id $DATASET_ID \
        --experiment_name "$EXPERIMENT_NAME"
    
    echo ">> [Setup] Planning and Preprocessing..."
    nnUNetv2_plan_and_preprocess -d $DATASET_ID -c 3d_fullres --verify_dataset_integrity
fi


# --- NEW: INJECT CUSTOM PATCH SIZE (Python) ---
# This runs regardless of SKIP_PREPROCESSING to ensure the plan matches the current run request
PLANS_JSON="${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans.json"

if [ -n "$CUSTOM_PATCH_SIZE" ] && [ -f "$PLANS_JSON" ]; then
    echo ">> [Setup] Injecting Custom Patch Size: [$CUSTOM_PATCH_SIZE] into nnUNetPlans.json"
    
    python3 -c "
import json
import sys

plans_path = '$PLANS_JSON'
patch_str = '$CUSTOM_PATCH_SIZE'

# Convert string '64,288,112' to list of ints [64, 288, 112]
try:
    patch_size = [int(x) for x in patch_str.split(',')]
except ValueError:
    print(f'❌ Error: Invalid patch string format: {patch_str}')
    sys.exit(1)

try:
    with open(plans_path, 'r') as f:
        data = json.load(f)

    # Locate 3d_fullres config
    if 'configurations' in data and '3d_fullres' in data['configurations']:
        old_size = data['configurations']['3d_fullres'].get('patch_size', 'unknown')
        print(f'   Old patch size: {old_size}')
        
        # UPDATE
        data['configurations']['3d_fullres']['patch_size'] = patch_size
        
        # Also safe to update median_image_size_in_voxels if you want, 
        # but here we strictly update patch_size as requested.
        
        with open(plans_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'✅ New patch size set: {patch_size}')
    else:
        print('⚠️ Warning: configurations/3d_fullres key not found in plans.json')

except Exception as e:
    print(f'❌ Error updating JSON: {e}')
    sys.exit(1)
"
else
    if [ -n "$CUSTOM_PATCH_SIZE" ]; then
        echo "⚠️ Warning: Custom patch size requested, but $PLANS_JSON not found."
    fi
fi


echo ">> [Setup] Injecting Splits..."
TARGET_SPLIT_DIR="${nnUNet_preprocessed}/${DATASET_NAME}"

if [ -d "$TARGET_SPLIT_DIR" ]; then
    if [ -f "splits_final.json" ]; then
        echo ">> [Setup] Found locally generated splits. Using them."
        cp splits_final.json "${TARGET_SPLIT_DIR}/splits_final.json"
    elif [ "$SKIP_PREPROCESSING" == "true" ] && [ "$IS_MAG_ONLY" == "false" ]; then
        echo ">> [Setup] Copying splits from Base Dataset..."
        cp ${nnUNet_preprocessed}/${BASE_DATASET_NAME}/splits_final.json "${TARGET_SPLIT_DIR}/splits_final.json"
    else
        if [ -f "${TARGET_SPLIT_DIR}/splits_final.json" ]; then
             echo ">> [Setup] Using existing splits in preprocessed folder."
        else
             echo "⚠️ Warning: No splits found to inject."
        fi
    fi
    echo ">> [Setup] Splits check complete."
else
    echo "⚠️ Warning: Preprocessed directory $TARGET_SPLIT_DIR not found."
fi