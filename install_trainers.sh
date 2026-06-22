#!/usr/bin/env bash
# Sync the repo's custom nnUNet trainers + helpers (single source of truth) into the active venv
# so nnUNet can load them by class name.
#
# Repo layout:
#   {2D,3D}_workspace/trainers/        the trainer (nnUNetTrainerWandb.py / nnUnet3DCustomTrainer.py)
#   {2D,3D}_workspace/trainers/utils/  custom_loss.py augmentation_*.py preprocessing_*.py visualization_*.py
# nnUNet loads trainers from a single FLAT package dir, so everything is copied flat into the venv
# variants/ dir. Trainers use relative imports (`from .custom_loss import ...`) -> resolve as siblings.
# custom_loss.py is shared by 2D and 3D (must stay identical).
set -euo pipefail
cd "$(dirname "$0")"; source paths.sh        # REPO_DIR, PY
VARIANTS="$(dirname "$NNUNET_BIN")/lib/python3.12/site-packages/nnunetv2/training/nnUNetTrainer/variants"
PINNED_NNUNET="2.6.2"
[ -d "$VARIANTS" ] || { echo "ERROR: venv variants dir not found: $VARIANTS"; exit 1; }

ver=$("$PY" -c "import importlib.metadata as m; print(m.version('nnunetv2'))" 2>/dev/null || echo "?")
[ "$ver" = "$PINNED_NNUNET" ] || echo "WARNING: nnunetv2 is $ver, trainers validated against $PINNED_NNUNET"

a=$(md5sum "$REPO_DIR/2D_workspace/trainers/utils/custom_loss.py" | cut -d' ' -f1)
b=$(md5sum "$REPO_DIR/3D_workspace/trainers/utils/custom_loss.py" | cut -d' ' -f1)
[ "$a" = "$b" ] || { echo "ERROR: 2D vs 3D custom_loss.py have drifted; they must stay identical."; exit 1; }

echo "Syncing trainers -> $VARIANTS"
for src in \
  "$REPO_DIR/2D_workspace/trainers/nnUNetTrainerWandb.py" \
  "$REPO_DIR/2D_workspace/trainers/utils/"*.py \
  "$REPO_DIR/3D_workspace/trainers/nnUnet3DCustomTrainer.py" \
  "$REPO_DIR/3D_workspace/trainers/utils/"*.py ; do
  cp -f "$src" "$VARIANTS/"; echo "  + $(basename "$src")"
done

echo "Verifying trainers import cleanly..."
"$PY" - <<'PY'
import importlib
for m in ("nnUNetTrainerWandb", "nnUnet3DCustomTrainer"):
    cls = getattr(importlib.import_module(f"nnunetv2.training.nnUNetTrainer.variants.{m}"), m)
    print(f"  OK  {m} ({cls.__name__})")
PY
echo "Done."
