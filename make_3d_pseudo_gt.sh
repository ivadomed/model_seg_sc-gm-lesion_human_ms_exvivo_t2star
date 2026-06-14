#!/usr/bin/env bash
# OPT-IN: regenerate the weakly-supervised 3D pseudo-GT by running a 2D model on ALL slices of
# every clean training volume and stacking the predictions (the 2D->3D step). Invoke UNDER set_slot:
#   set_slot 3 bash make_3d_pseudo_gt.sh <2d_model_folder> [out_dir] [subjects...]
#
# DEFAULT for the 3D pipeline is to CONSUME the stored ms-exvivo-nih/derivatives/labels_3d
# (exact reproducibility). This script only regenerates them and is NOT needed for normal use.
# NOTE: per-slice CPU-bound (~minutes/volume); verified correct (fg-Dice ~0.99 vs stored labels_3d).
set -euo pipefail
MODEL="${1:?usage: make_3d_pseudo_gt.sh <2d_model_folder> [out_dir] [subjects...]}"
cd "$(dirname "$0")"; source paths.sh
OUT="${2:-$OUTPUTS/pseudo_gt_3d}"
shift || true; shift || true
mkdir -p "$OUT"

if [ "$#" -gt 0 ]; then SUBJECTS=("$@"); else
  mapfile -t SUBJECTS < <(for d in "$CLEAN_DATASET"/derivatives/labels_3d/sub-*; do
    s=$(basename "$d"); ls "$d"/anat/*_T2star.nii.gz 2>/dev/null | grep -qv "_label-" && echo "$s"; done)
fi
echo "regenerating pseudo-GT for ${#SUBJECTS[@]} subjects -> $OUT  (model: $MODEL)"
for sub in "${SUBJECTS[@]}"; do
  for mag in "$CLEAN_DATASET/$sub/anat/"*_part-mag_T2star.nii.gz; do
    [ -e "$mag" ] || continue
    echo "  [$sub] $(basename "$mag")"
    PYTHONPATH="$REPO_DIR" "$PY" 2D_workspace/inference/inference_2.py \
      -experiment pseudo_gt -model_folder "$MODEL" -mode volume -path "$mag" \
      -folds 0 1 2 3 -output_root "$OUT/$sub" -rel_path ""
  done
done
echo "done -> $OUT  (compare to ms-exvivo-nih/derivatives/labels_3d to validate)"
