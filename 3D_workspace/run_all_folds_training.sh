#!/bin/bash

set -e
# set -x # Uncomment for debugging

GPU_IDS_STR=0,1,2,3
FOLD_IDS_STR=0,1,2,3

# Convert comma-separated strings to arrays
IFS=',' read -r -a GPU_IDS <<< "$GPU_IDS_STR"
IFS=',' read -r -a FOLD_IDS <<< "$FOLD_IDS_STR"

NUM_GPUS=${#GPU_IDS[@]}
NUM_FOLDS=${#FOLD_IDS[@]}

if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPU IDs provided."
    exit 1
fi

if [ $NUM_FOLDS -eq 0 ]; then
    echo "Error: No fold IDs provided."
    exit 1
fi

# --- CONFIGURATION ---
DATASET_ID=1747
EXPERIMENT_NAME="patch_5_adamw_aug_26" 
TRAINER_CLASS=nnUnet3DCustomTrainer
VENV_PATH=/home/ge.polymtl.ca/pahoa/nih_project/.venv
DEBUG_FOLD= # Optional

# --- NEW: CUSTOM PATCH SIZE ---
# Format: "x,y,z" (Comma separated, no spaces, no brackets)
# Leave empty to use the default patch size found in the copied plans
PATCH_SIZE="192,64,208"

# IMPORTANT: Must be false for 'mag-one-channel' on first run to generate 1-channel data
SKIP_PREPROCESSING=true 

SESSION_NAME="nnunet-${DATASET_ID}-${TRAINER_CLASS}-${EXPERIMENT_NAME}"
SOCKET_FILE="/tmp/tmux-socket-${USER}-${SESSION_NAME}.sock"

# We pass these to the sub-scripts so they know where the data is
export BIDS_DATA_PATH="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset"
export NNUNET_BASE="/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/nnunet_dataset"

# --- STEP 1: ONE-TIME PREPROCESSING (Inside set_slot) ---
echo "------------------------------------------------"
echo "Step 1: Running Setup (Prepro + Plans + Splits)"
echo "------------------------------------------------"

# Pass PATCH_SIZE as the 5th argument
SETUP_CMD="bash setup_experiment.sh $DATASET_ID '$VENV_PATH' '$EXPERIMENT_NAME' '$SKIP_PREPROCESSING' '$PATCH_SIZE'"
set_slot "${GPU_IDS[0]}" bash -c "$SETUP_CMD"
PREPRO_EXIT_CODE=$?

if [ $PREPRO_EXIT_CODE -ne 0 ]; then
    echo "❌ Error: Preprocessing failed. Aborting."
    exit 1
fi
echo "✅ Setup complete."

# --- BRANCH: DEBUG MODE ---
if [ -n "$DEBUG_FOLD" ]; then
    echo "----------------------------------------------------------------"
    echo "🛠️  DEBUG MODE DETECTED: Running Fold $DEBUG_FOLD in Foreground 🛠️"
    echo "----------------------------------------------------------------"
    echo "Skipping tmux session creation."
    
    # Run single fold in foreground via set_slot
    set_slot "${GPU_IDS[0]}" bash -c "bash train_fold.sh $DEBUG_FOLD $DATASET_ID $TRAINER_CLASS '$VENV_PATH' ${GPU_IDS[0]}"
    
    echo "--- Debug run for Fold $DEBUG_FOLD finished. ---"
    exit 0
fi

# --- BRANCH: STANDARD AUTOMATION (TMUX) ---
echo "--- Checking for existing tmux session... ---"
unset TMUX

if tmux -S "$SOCKET_FILE" has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "❌ Error: Session '$SESSION_NAME' already exists on socket $SOCKET_FILE."
    echo "Attach with: tmux -S $SOCKET_FILE attach -t $SESSION_NAME"
    exit 1
fi

echo "--- Starting tmux session '$SESSION_NAME'... ---"
tmux -S "$SOCKET_FILE" -f /dev/null new-session -d -s "$SESSION_NAME" -n "fold_0"

# --- LAUNCH FOLDS ---
echo "--- Queuing $NUM_FOLDS-fold training in separate tmux windows... ---"

for i in "${!FOLD_IDS[@]}"; do
    FOLD=${FOLD_IDS[$i]}
    GPU_ID_INDEX=$((i % NUM_GPUS))
    GPU_ID=${GPU_IDS[$GPU_ID_INDEX]}

    echo "Starting fold $FOLD on GPU $GPU_ID..."
    
    CMD_N="set_slot $GPU_ID bash -c 'bash train_fold.sh $FOLD $DATASET_ID $TRAINER_CLASS $VENV_PATH $GPU_ID'"

    if [ $i -eq 0 ]; then
        # Use the first window (already created)
        tmux -S "$SOCKET_FILE" send-keys -t "$SESSION_NAME:fold_0" "$CMD_N" C-m
    else
        # Create a new window for other folds
        tmux -S "$SOCKET_FILE" new-window -t "$SESSION_NAME" -n "fold_$FOLD"
        tmux -S "$SOCKET_FILE" send-keys -t "$SESSION_NAME:fold_$FOLD" "$CMD_N" C-m
    fi
done

echo "----------------------------------------------------------------"
echo "✅ All folds queued."
echo "👀 MONITOR: tmux -S $SOCKET_FILE attach -t $SESSION_NAME"
echo "----------------------------------------------------------------"