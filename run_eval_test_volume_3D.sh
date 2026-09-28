#!/usr/bin/env bash
# Evaluate EVERY 3D paper model on the single manually-annotated 3D test volume
# (sub-TNU026_acq-S1), then build a leaderboard + statistical tests.
#
#   set_slot 3 bash run_eval_test_volume_3D.sh [--tta] [--folds "0 1 2 3"] [--restore-orient]
#
# What it does:
#   1. stage the test volume into nnU-Net input folders (2-channel mag+phase, and a
#      1-channel mag-only variant) named so each prediction basename == the GT basename;
#   2. loop over every trained 3D model under paper_results/3D, predicting with the
#      channel count that model expects (read from its own dataset.json);
#   3. score each prediction vs the 3D GT and run the single-volume statistics
#      (helpers/eval_test_volume.py -- read its STATS_NOTES.md, N=1 caveats apply).
#
# NOTE: all compute must run under set_slot (GPU/CPU/RAM allocation on this machine).
set -euo pipefail
cd "$(dirname "$0")"; source paths.sh

FOLDS="0 1 2 3"; TTA=""; RESTORE=0; LIMIT=0
GPU=0; SHARD_IDX=-1; SHARD_N=1                 # multi-GPU sharding (see run_eval_test_volume_3D_parallel.sh)
DO_STAGE=1; DO_INFER=1; DO_STATS=1
while [ $# -gt 0 ]; do case "$1" in
  --tta) TTA="--tta";;
  --folds) FOLDS="$2"; shift;;
  --restore-orient) RESTORE=1;;
  --limit) LIMIT="$2"; shift;;          # for smoke-testing: only run the first N models
  --gpu-id) GPU="$2"; shift;;           # which GPU this process predicts on
  --shard) SHARD_IDX="$2"; SHARD_N="$3"; shift 2;;  # process only models where (index % N == IDX)
  --stage-only) DO_INFER=0; DO_STATS=0;;
  --no-stage) DO_STAGE=0;;
  --no-stats) DO_STATS=0;;
  --stats-only) DO_STAGE=0; DO_INFER=0;;
  *) echo "unknown flag $1"; exit 1;; esac; shift; done

SUB="sub-TNU026_acq-S1_part-mag_T2star"
SRC="$CLEAN_DATASET/sub-TNU026/anat"
GT_SRC="$CLEAN_DATASET/derivatives/labels_3d/sub-TNU026/anat/${SUB}.nii.gz"
ROOT="$OUTPUTS/test_volume_eval"
IN2="$ROOT/_input_magphase"      # 2-channel models
IN1="$ROOT/_input_mag"           # 1-channel models
GTDIR="$ROOT/_gt"
PAPER="$PROJECT_ROOT/nnUNet_data/nnUNet_results/paper_results/3D"
mkdir -p "$IN2" "$IN1" "$GTDIR"

[ -f "$GT_SRC" ] || { echo "GT not found: $GT_SRC"; exit 1; }

echo "=== staging test volume + GT ==="
cp -f "$SRC/${SUB}.nii.gz"                          "$IN2/${SUB}_0000.nii.gz"
cp -f "$SRC/sub-TNU026_acq-S1_part-phase_T2star.nii.gz" "$IN2/${SUB}_0001.nii.gz"
cp -f "$SRC/${SUB}.nii.gz"                          "$IN1/${SUB}_0000.nii.gz"
cp -f "$GT_SRC" "$GTDIR/${SUB}.nii.gz"

if [ "$RESTORE" = 1 ]; then
  echo "=== restoring training orientation (affine-only, arrays unchanged) ==="
  for d in "$IN2" "$IN1" "$GTDIR"; do
    "$PY" -m helpers.restore_training_orientation "$d" --out "$d" --verify
  done
fi

echo "=== inference: every 3D model under $PAPER ==="
# each model leaf = the dir containing dataset.json + fold_*/checkpoint_best.pth
NRUN=0
while IFS= read -r LEAF; do
  [ "$LIMIT" -gt 0 ] && [ "$NRUN" -ge "$LIMIT" ] && break
  # skip junk / incomplete leaves: require all requested folds to have the checkpoint
  ok=1; for f in $FOLDS; do [ -f "$LEAF/fold_$f/checkpoint_best.pth" ] || ok=0; done
  [ "$ok" = 1 ] || { echo "SKIP (missing fold checkpoint): ${LEAF#$PAPER/}"; continue; }

  NCH=$("$PY" -c "import json;print(len(json.load(open('$LEAF/dataset.json'))['channel_names']))")
  IN="$IN2"; [ "$NCH" = 1 ] && IN="$IN1"

  # model id = <family>/<exp>  (drop the trainer__plans__config leaf); tag trainer to disambiguate
  REL="${LEAF#$PAPER/}"; FAMEXP="$(echo "$REL" | cut -d/ -f1-2)"
  TRAINER="$(basename "$LEAF" | cut -d_ -f1)"   # e.g. nnUnet3DCustomTrainer / nnUNetDistillationTrainer
  OUT="$ROOT/$FAMEXP/predict_ensemble_$([ -n "$TTA" ] && echo tta || echo notta)"
  # if two trainers share one family/exp, keep both
  case "$(basename "$LEAF")" in nnUNetDistillation*) OUT="${OUT}_distill";; esac
  mkdir -p "$OUT"

  echo "--- $FAMEXP  (channels=$NCH, folds=[$FOLDS]$([ -n "$TTA" ] && echo ', tta'))"
  set +e
  "$PY" inference_publication/infer_3d_public.py \
      --input-dir "$IN" --output-dir "$OUT" --model-folder "$LEAF" \
      --folds $FOLDS --checkpoint checkpoint_best.pth $TTA --overwrite
  rc=$?; set -e
  [ $rc -eq 0 ] || echo "SKIP (inference failed rc=$rc): $FAMEXP"
  NRUN=$((NRUN + 1))
done < <(find "$PAPER" -name dataset.json -exec dirname {} \; | sort -u)

echo "=== scoring + single-volume statistics ==="
PYTHONPATH="$REPO_DIR" "$PY" -m helpers.eval_test_volume \
  --pred-root "$ROOT" --pred-glob "*/*/predict_*" \
  --gt "$GTDIR/${SUB}.nii.gz" --primary dice_cord \
  --out "$ROOT"

echo "=== done ==="
echo "  leaderboard        -> $ROOT/leaderboard.csv"
echo "  pairwise vs winner -> $ROOT/pairwise_vs_reference.csv"
echo "  READ THE CAVEATS   -> $ROOT/STATS_NOTES.md"
