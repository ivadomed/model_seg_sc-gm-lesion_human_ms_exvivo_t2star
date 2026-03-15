#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Replace these placeholders for your setup.
INPUT_DIR="/path/to/nnunet_inputs_3d/imagesTs"
OUTPUT_DIR="/path/to/output/predictions_3d"
MODEL_FOLDER="/path/to/nnUNet_results/DatasetXXXX_<TASK_NAME>/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"

GPU_ID="${GPU_ID:-0}"
FOLDS=(0 1 2 3)
CHECKPOINT="checkpoint_final.pth"

python "${SCRIPT_DIR}/infer_3d_public.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-folder "${MODEL_FOLDER}" \
  --folds "${FOLDS[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --gpu-id "${GPU_ID}" \
  --overwrite
