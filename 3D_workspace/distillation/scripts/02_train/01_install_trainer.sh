#!/usr/bin/env bash
# Copy the distillation trainer files into the nnU-Net venv so `-tr nnUnet3DDistillTrainer` can
# find them (nnU-Net loads trainers from a single flat package dir).
# Requires the repo's base 3D trainer already installed: from the repo root, `bash install_trainers.sh`.
set -euo pipefail
REPO_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
source "$REPO_DIR/paths.sh"   # PY, NNUNET_BIN

TRAINER_DIR="$REPO_DIR/3D_workspace/distillation/trainer"
VARIANTS="$(dirname "$NNUNET_BIN")/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/variants"
[ -d "$VARIANTS" ] || { echo "ERROR: venv variants dir not found: $VARIANTS"; exit 1; }

for f in nnUnet3DCustomTrainer.py augmentation_3D.py custom_loss.py; do
  [ -f "$VARIANTS/$f" ] || { echo "ERROR: $f missing in $VARIANTS -- run the repo's install_trainers.sh first"; exit 1; }
done

echo "Syncing distillation trainer -> $VARIANTS"
for f in "$TRAINER_DIR"/*.py; do
  cp -f "$f" "$VARIANTS/"
  echo "  + $(basename "$f")"
done

echo "Verifying import..."
"$PY" -c "
import importlib
m = importlib.import_module('nnunetv2.training.nnUNetTrainer.variants.nnUnet3DDistillTrainer')
print('  OK ', m.nnUnet3DDistillTrainer.__name__)
"
