# Reproducing published results after the dataset orientation correction

## What changed in the dataset

The public `ms-exvivo-nih` dataset received an **affine-only z-flip** (LPS → LPI)
that corrects the upside-down scanner mounting in world space. This changes only
the NIfTI **affine/header** — **voxel arrays are byte-identical**. The published
models were trained *before* this correction, i.e. on the original `+z` headers.

## Does this affect the models? No.

The pipeline is orientation-invariant at the level the network sees:

- **2D** (`convert_bids_to_nnunet_multichannel.py`) resets the affine diagonal to
  identity (`iso_affine`) and reads/slices arrays **by index** — the saved network
  inputs are identical arrays regardless of the source header sign.
- **3D / nnU-Net** consume raw voxel arrays with **positive spacing magnitudes**;
  nnU-Net v2 does not reorient by direction.

⇒ Network inputs are byte-identical before and after the correction, so training
and inference **reproduce exactly** on the corrected dataset. (Predictions now
also inherit the corrected header, so they land right-side-up in world space —
a bonus, with no effect on array-level results.)

> A *voxel-data* flip would break this; the correction and the tool below are
> **affine-only** and provably leave voxels untouched (`--verify`).

## Exact, literal reproduction (identical headers too)

For perfect byte-level parity with the training-time inputs (identical headers,
not just identical arrays), restore the original `+z` orientation before feeding
the models — for **both** training and testing:

```bash
# 1. Materialize training-orientation copies of what you need (header-only flip)
python -m helpers.restore_training_orientation \
    <ms-exvivo-nih or subset> --out /path/to/repro_root --verify

# 2. Build the nnU-Net datasets from the restored copy
python 3D_workspace/dataset_prep/build_dataset_3d.py --clean-root /path/to/repro_root ...
python 2D_workspace/dataset_prep/build_dataset_2d.py --clean-root /path/to/repro_root ...

# For inference on a new/held-out volume, restore it first, then predict:
python -m helpers.restore_training_orientation input_vol.nii.gz --out restored/ --verify
```

`restore_training_orientation.py` re-applies the same self-inverse affine z-flip:
a corrected (`-z`) file is returned to its original (`+z`) header, voxels
untouched (`--verify` asserts this). Files already in `+z` are copied through.
Because the flip is affine-only, this step **cannot change any result** — it only
guarantees the headers match the training-time state for airtight provenance.
