#!/usr/bin/env bash
# Run the full distillation experiment matrix: control + 3 KD temperatures, folds 1-3 (fold 0
# held back). One condition fully finishes (3 folds in parallel, one GPU slot each) before the
# next starts. Meant to be launched detached (nohup ... &) -- the whole matrix takes ~40h.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DISTILL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"   # scripts/02_train -> scripts -> distillation
LOGDIR="$DISTILL_DIR/run_logs"
mkdir -p "$LOGDIR"

# tag : config : extra env override, in run order
CONDITIONS=(
  "control:$DISTILL_DIR/configs/control.json:"
  "kd_T1:$DISTILL_DIR/configs/distill.json:KD_TEMPERATURE=1.0"
  "kd_T2:$DISTILL_DIR/configs/distill.json:"
  "kd_T4:$DISTILL_DIR/configs/distill.json:KD_TEMPERATURE=4.0"
)
SLOTS=(1 2 3)   # GPU slots available; fold N runs on SLOTS[N-1]
FOLDS=(1 2 3)

echo "$(date -Is) === installing distill trainer (once) ===" | tee -a "$LOGDIR/orchestrator.log"
if ! set_slot 1 bash "$SCRIPT_DIR/01_install_trainer.sh" >> "$LOGDIR/orchestrator.log" 2>&1; then
  echo "$(date -Is) FATAL: 01_install_trainer.sh failed, aborting" | tee -a "$LOGDIR/orchestrator.log"
  exit 1
fi

for condition in "${CONDITIONS[@]}"; do
  IFS=: read -r TAG CONFIG ENV_OVERRIDE <<< "$condition"
  echo "$(date -Is) === starting condition $TAG ($CONFIG), folds 1-3 ===" | tee -a "$LOGDIR/orchestrator.log"

  PIDS=()
  for i in "${!FOLDS[@]}"; do
    FOLD="${FOLDS[$i]}"
    SLOT="${SLOTS[$i]}"
    LOG="$LOGDIR/${TAG}_fold${FOLD}.log"
    echo "$(date -Is)   launching fold=$FOLD on slot=$SLOT -> $LOG" | tee -a "$LOGDIR/orchestrator.log"
    set_slot "$SLOT" bash -c "DISTILL_TAG='$TAG' SKIP_INSTALL=1 $ENV_OVERRIDE bash '$SCRIPT_DIR/02_run_distill.sh' '$FOLD' '$CONFIG'" \
      > "$LOG" 2>&1 &
    PIDS+=($!)
  done

  FAIL=0
  for pid in "${PIDS[@]}"; do
    wait "$pid" || FAIL=1
  done
  if [ "$FAIL" -ne 0 ]; then
    echo "$(date -Is) WARNING: at least one fold failed for condition $TAG (check $LOGDIR/${TAG}_fold*.log) -- continuing" | tee -a "$LOGDIR/orchestrator.log"
  else
    echo "$(date -Is) === condition $TAG done, all 3 folds OK ===" | tee -a "$LOGDIR/orchestrator.log"
  fi
done

echo "$(date -Is) === ALL CONDITIONS COMPLETE ===" | tee -a "$LOGDIR/orchestrator.log"
