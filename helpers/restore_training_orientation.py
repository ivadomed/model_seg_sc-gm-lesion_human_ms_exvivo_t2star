#!/usr/bin/env python3
"""Restore the *training-time* orientation of ms-exvivo-nih volumes for exact
reproduction of the published models.

Background
----------
The public ms-exvivo-nih dataset received an orientation correction: an
AFFINE-ONLY z-flip (LPS -> LPI) that fixes the upside-down scanner mounting in
world space. That correction changes ONLY the NIfTI affine/header -- the voxel
arrays are byte-identical. The published models were trained BEFORE this
correction (i.e. on the +z / un-flipped headers).

Why this tool exists
--------------------
The segmentation pipeline is orientation-invariant at the level that matters:
  * 2D prep resets the affine diagonal to identity (iso) and slices by index;
  * 3D / nnU-Net consume raw voxel arrays with positive spacing magnitudes.
So the network inputs are identical whether the header is +z or -z, and results
reproduce exactly on the corrected dataset. This script is a belt-and-suspenders
step for perfect, literal reproduction (identical headers, not just identical
arrays): it re-applies the same affine-only z-flip, which is self-inverse, so a
corrected (-z) file is returned to its original (+z) training-time header while
its voxels are left exactly as-is.

It is safe to run before BOTH dataset building (training) and inference
(testing): it never alters a single voxel.

Usage
-----
  # Write training-orientation copies of a subset into a repro root, then point
  # build_dataset_*.py --clean-root at it:
  python -m helpers.restore_training_orientation IN_DIR --out REPRO_DIR

  # Verify a produced copy has identical voxels to the source (headers aside):
  python -m helpers.restore_training_orientation IN_DIR --out REPRO_DIR --verify
"""
import argparse, os, glob, shutil, sys
import numpy as np
import nibabel as nib


def zflip_affine(affine, nz):
    """Self-inverse affine-only z-flip (voxels unchanged)."""
    a = affine.copy()
    a[2, 3] = affine[2, 2] * (nz - 1) + affine[2, 3]
    a[2, 2] = -affine[2, 2]
    return a


def restore_file(src, dst):
    """Write dst = src with the z-axis affine flipped back to +z (header-only).
    If src is already +z, it is copied through unchanged. Voxels are untouched."""
    im = nib.load(src)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if im.affine[2, 2] > 0:                      # already training orientation
        shutil.copy2(src, dst)
        return "already+z"
    new_aff = zflip_affine(im.affine, im.shape[2])
    out = nib.Nifti1Image(np.asanyarray(im.dataobj), new_aff, im.header)
    out.set_qform(new_aff); out.set_sform(new_aff)
    nib.save(out, dst)
    return "restored"


def main():
    ap = argparse.ArgumentParser(description="Restore training-time (+z) orientation; header-only, voxels untouched.")
    ap.add_argument("in_path", help="file or directory tree of .nii.gz to restore")
    ap.add_argument("--out", required=True, help="output root (mirrors input tree)")
    ap.add_argument("--verify", action="store_true", help="assert output voxels == input voxels")
    a = ap.parse_args()

    if os.path.isfile(a.in_path):
        pairs = [(a.in_path, os.path.join(a.out, os.path.basename(a.in_path)))]
    else:
        pairs = [(f, os.path.join(a.out, os.path.relpath(f, a.in_path)))
                 for f in glob.glob(os.path.join(a.in_path, "**", "*.nii.gz"), recursive=True)]
    if not pairs:
        sys.exit(f"no .nii.gz found under {a.in_path}")

    n_restored = n_copied = 0
    for src, dst in pairs:
        r = restore_file(src, dst)
        n_restored += (r == "restored"); n_copied += (r == "already+z")
        if a.verify:
            vi = np.asanyarray(nib.load(src).dataobj)
            vo = np.asanyarray(nib.load(dst).dataobj)
            assert np.array_equal(vi, vo), f"VOXELS CHANGED for {dst} (bug!)"
    print(f"[restore-orientation] {len(pairs)} files -> {a.out}  "
          f"(restored {n_restored}, already +z {n_copied}"
          + (", voxels verified identical" if a.verify else "") + ")")


if __name__ == "__main__":
    main()
