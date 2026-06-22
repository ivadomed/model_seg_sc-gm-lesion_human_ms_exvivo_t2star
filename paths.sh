# Central path config for the repo. Source this in every runner (it locates the external data,
# which lives OUTSIDE the repo, at the project root = the repo's parent directory).
#   REPO_DIR     = this repository (model_seg_sc-gm-lesion_human_ms_exvivo_t2star)
#   PROJECT_ROOT = parent dir holding the data: ms-exvivo-nih/, nnUNet_data/, outputs/, .venv/
# Override any of these by exporting them before sourcing (e.g. PROJECT_ROOT=/path ... ).
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_DIR
export PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$REPO_DIR")}"
export PY="${PY:-$PROJECT_ROOT/.venv/bin/python}"
export NNUNET_BIN="$(dirname "$PY")"   # location of nnUNetv2_train / nnUNetv2_predict / plan_and_preprocess
export CLEAN_DATASET="${CLEAN_DATASET:-$PROJECT_ROOT/ms-exvivo-nih}"
export nnUNet_raw="${nnUNet_raw:-$PROJECT_ROOT/nnUNet_data/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-$PROJECT_ROOT/nnUNet_data/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-$PROJECT_ROOT/nnUNet_data/nnUNet_results}"
export OUTPUTS="${OUTPUTS:-$PROJECT_ROOT/outputs}"
export WANDB_MODE="${WANDB_MODE:-disabled}"   # wandb logging opt-in; set WANDB_MODE=online to enable
mkdir -p "$nnUNet_preprocessed" "$nnUNet_results" "$OUTPUTS"
