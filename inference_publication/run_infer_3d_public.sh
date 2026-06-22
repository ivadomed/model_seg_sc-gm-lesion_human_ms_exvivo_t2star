#!/usr/bin/env bash
# Publication-facing 3D inference for the released model (recommended model). No editing needed:
#   bash run_infer_3d_public.sh <input_dir> <output_dir> <model_folder> [extra infer_3d_public.py args]
#   <input_dir>    : nnU-Net imagesTs-style folder (CASE_0000.nii.gz [+ _0001 phase])
#   <model_folder> : released 3D model dir (…/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres)
# Override checkpoint/gpu via env: CHECKPOINT=checkpoint_final.pth GPU_ID=0 ... ; pass --tta etc. as extra args.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${1:?usage: run_infer_3d_public.sh <input_dir> <output_dir> <model_folder> [args...]}"
OUTPUT_DIR="${2:?need <output_dir>}"
MODEL_FOLDER="${3:?need <model_folder>}"
shift 3
python "${SCRIPT_DIR}/infer_3d_public.py" \
  --input-dir "${INPUT_DIR}" --output-dir "${OUTPUT_DIR}" --model-folder "${MODEL_FOLDER}" \
  --folds 0 1 2 3 --checkpoint "${CHECKPOINT:-checkpoint_best.pth}" --gpu-id "${GPU_ID:-0}" --overwrite "$@"
