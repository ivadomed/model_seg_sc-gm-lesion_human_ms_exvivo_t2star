#!/bin/bash

set -e
set -x # Enable debugging

# --- Configuration ---
# Usage: 
# Standard: ./run_five_folds_exp.sh <GPU_ID> <DATASET_ID> <EXPERIMENT_NAME> <LABEL_MODE>
# Debug:    ./run_five_folds_exp.sh <GPU_ID> <DATASET_ID> <EXPERIMENT_NAME> <LABEL_MODE> <DEBUG_FOLD>

# $3: LABEL_MODE (e.g., "all", "lesions", "sc_and_lesion")


# if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
#     echo "Usage: $0 <GPU_ID> <DATASET_ID> <EXPERIMENT_NAME> <LABEL_MODE> [DEBUG_FOLD]"
#     echo "Example (Full Exp): ./run_five_folds_exp.sh 0 524 my_exp all"
#     echo "Example (Debug Fold 0): ./run_five_folds_exp.sh 0 524 my_exp all 0"
#     exit 1
# fi

GPU_ID=0
DATASET_ID=717
EXPERIMENT_NAME=winning_combination
LABEL_MODE='all'
DEBUG_FOLD= # Optional: 0, 1, 2, 3, or 4

SESSION_NAME="nnunet-${DATASET_ID}-${EXPERIMENT_NAME}"
SOCKET_FILE="/tmp/tmux-socket-${USER}-${SESSION_NAME}.sock"

# --- Step 1: Run One-Time Preprocessing ---
# We run this in both modes to ensure data is ready. 
# (nnU-Net usually skips this quickly if data is already preprocessed).
echo "--- Running preprocessing script... ---"
set_slot "$GPU_ID" bash -c "bash prepro_for_training.sh '$DATASET_ID' '$EXPERIMENT_NAME' '$LABEL_MODE'"
PREPRO_EXIT_CODE=$?

if [ $PREPRO_EXIT_CODE -ne 0 ]; then
    echo "Error: Preprocessing failed. Aborting."
    exit 1
fi

# --- BRANCH: DEBUG MODE ---
if [ -n "$DEBUG_FOLD" ]; then
    echo "----------------------------------------------------------------"
    echo "🛠️  DEBUG MODE DETECTED: Running Fold $DEBUG_FOLD in Foreground 🛠️"
    echo "----------------------------------------------------------------"
    echo "Skipping tmux session creation."
    echo "Running directly in bash. Press Ctrl+C to stop."
    
    # Run the single fold directly in the current shell (wrapped in set_slot)
    set_slot "$GPU_ID" bash -c "bash run_training.sh $DEBUG_FOLD $GPU_ID $DATASET_ID"
    
    echo "--- Debug run for Fold $DEBUG_FOLD finished. ---"
    exit 0
fi

# --- BRANCH: STANDARD AUTOMATION (TMUX) ---
# If we are here, DEBUG_FOLD is empty, so we proceed with the full 5-fold tmux setup.

echo "--- Checking for existing tmux session... ---"
unset TMUX

# Check if session exists using the custom socket
if tmux -S "$SOCKET_FILE" has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Error: A tmux session named '$SESSION_NAME' already exists on socket $SOCKET_FILE."
    echo "Attach with: tmux -S $SOCKET_FILE attach -t $SESSION_NAME"
    exit 1
fi

echo "--- Starting tmux session '$SESSION_NAME' on socket '$SOCKET_FILE'... ---"

# Create new session (ignoring user config with -f /dev/null)
tmux -S "$SOCKET_FILE" -f /dev/null new-session -d -s "$SESSION_NAME" -n "fold_0"

# --- Step 3: Launch 5-Fold Training in Tmux ---
echo "--- Starting 4-fold training in separate tmux windows... ---"

# Fold 0
sleep 1
TRAIN_CMD_0="set_slot 0 bash -c 'bash run_training.sh 0 0 $DATASET_ID'"
tmux -S "$SOCKET_FILE" send-keys -t "$SESSION_NAME:fold_0" "$TRAIN_CMD_0" C-m

# Folds 1-3
for FOLD in {1..3}
do
    echo "Starting fold $FOLD..."
    tmux -S "$SOCKET_FILE" new-window -t "$SESSION_NAME" -n "fold_$FOLD"
    TRAIN_CMD_N="set_slot $FOLD bash -c 'bash run_training.sh $FOLD $FOLD $DATASET_ID'"
    tmux -S "$SOCKET_FILE" send-keys -t "$SESSION_NAME:fold_$FOLD" "$TRAIN_CMD_N" C-m
done

echo "--- ✅ All training folds have been started. ---"
echo "To monitor the training, you MUST specify the socket:"
echo "   tmux -S $SOCKET_FILE attach -t $SESSION_NAME"