#!/usr/bin/env bash
# Reviewer concern #1: is the 3D model anatomically BETTER (not merely smoother)?
# Evaluate the WINNING 3D model vs the WINNING 2D model (stacked to a volume) on the ONE chunk with a
# manual 3D ground truth (sub-TNU026_acq-S1), scoring on the REAL 3D GT (not sparse slices):
# Dice/HD95 + inter-slice smoothness DSC_z + lesion connected-component counts (helpers/dense_metrics.py).
# Both models use their published ensemble(+TTA) inference. Runs under set_slot; two GPUs in parallel.
set -euo pipefail
cd "$(dirname "$0")"; source paths.sh
SUB="sub-TNU026_acq-S1_part-mag_T2star"
SRC="$CLEAN_DATASET/sub-TNU026/anat"
GT="$CLEAN_DATASET/derivatives/labels_3d/sub-TNU026/anat/${SUB}.nii.gz"
M3="$nnUNet_results/paper_results/3D/ablations/base3d/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"
M2="$nnUNet_results/paper_results/2D/ablations/synergy_aug2_soft2/nnUNetTrainerWandb__nnUNetPlans__2d"
OUT="$OUTPUTS/winning_dense_chunk"; IN3="$OUT/_in3d"
mkdir -p "$OUT" "$IN3"
[ -f "$GT" ] || { echo "GT missing: $GT"; exit 1; }

# stage nnU-Net-format input for the 3D predictor (mag=_0000, phase=_0001)
cp -f "$SRC/${SUB}.nii.gz" "$IN3/${SUB}_0000.nii.gz"
cp -f "$SRC/sub-TNU026_acq-S1_part-phase_T2star.nii.gz" "$IN3/${SUB}_0001.nii.gz"

# --- winning 3D (ensemble folds 0-3 + TTA) on GPU 0 ---
( CUDA_VISIBLE_DEVICES=0 set_slot 0 env PYTHONPATH="$REPO_DIR" "$PY" inference_publication/infer_3d_public.py \
    --input-dir "$IN3" --output-dir "$OUT/pred_3d" --model-folder "$M3" \
    --folds 0 1 2 3 --tta --overwrite > "$OUT/pred_3d.log" 2>&1 ) &
# --- winning 2D (ensemble folds 0-3 + TTA) volume mode -> stacked volume on GPU 1 ---
( CUDA_VISIBLE_DEVICES=1 set_slot 1 env PYTHONPATH="$REPO_DIR" "$PY" 2D_workspace/inference_2D.py \
    -experiment winning -model_folder "$M2" -mode volume -path "$SRC/${SUB}.nii.gz" \
    -folds 0 1 2 3 --use_tta -output_root "$OUT/pred_2d" > "$OUT/pred_2d.log" 2>&1 ) &
wait
echo "=== predictions done; scoring vs 3D GT ==="
P3="$(find "$OUT/pred_3d" -name "${SUB}.nii.gz" | head -1)"
P2="$(find "$OUT/pred_2d" -name "*.nii.gz" | head -1)"
echo "  3D pred: $P3"; echo "  2D pred: $P2"
set_slot 0 env PYTHONPATH="$REPO_DIR" "$PY" -m helpers.dense_metrics --pred "$P3" --gt "$GT" --name winning_3D       --out "$OUT/metrics_winning_3D.csv"
set_slot 0 env PYTHONPATH="$REPO_DIR" "$PY" -m helpers.dense_metrics --pred "$P2" --gt "$GT" --name winning_2D_stack --out "$OUT/metrics_winning_2D_stack.csv"
"$PY" - <<PYEOF
import pandas as pd, glob
df=pd.concat([pd.read_csv(f) for f in glob.glob("$OUT/metrics_winning_*.csv")], ignore_index=True)
df.to_csv("$OUT/dense_chunk_comparison.csv", index=False)
print(df.to_string(index=False))
PYEOF
echo "=== done -> $OUT/dense_chunk_comparison.csv ==="