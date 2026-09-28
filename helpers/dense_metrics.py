#!/usr/bin/env python3
"""
Dense-chunk metrics for the "is the 3D model anatomically better, not just smoother?" comparison
(reviewer concern #1). Scores ONE prediction volume against the manual 3D ground truth of the
annotated test chunk (sub-TNU026_acq-S1), reporting - per class and region:
  - Dice, HD95 (overlap + boundary, on REAL 3D GT, not sparse slices)
  - inter-slice Dice DSC_z (the paper's smoothness metric) -- high = smooth along Z
  - lesion connected-component count (pred vs GT) and lesion volume (voxels) -- to show smoothness
    does NOT come from deleting/merging true lesions (R4).
Reports pred-vs-GT so a smooth-but-wrong prediction is exposed (low Dice / wrong component count),
directly rebutting "smoothness != correctness".

CLI:
  python -m helpers.dense_metrics --pred P.nii.gz --gt G.nii.gz --name winning_3D --out row.csv
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np, nibabel as nib, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.eval import _dice, _hd95, DEFAULT_LABELS, DEFAULT_REGIONS
from scipy import ndimage


def dsc_z(mask):
    """Mean inter-slice Dice over Z (axis 2), over valid transitions (class present in either slice)."""
    ds = []
    for z in range(mask.shape[2] - 1):
        a, b = mask[:, :, z], mask[:, :, z + 1]
        s = a.sum() + b.sum()
        if s > 0:
            ds.append(2.0 * np.logical_and(a, b).sum() / s)
    return float(np.mean(ds)) if ds else np.nan


def score(pred, gt, spacing, name):
    row = {"model": name}
    for lab, nm in DEFAULT_LABELS.items():
        pm, gm = pred == lab, gt == lab
        row[f"dice_{nm}"] = _dice(pm, gm)
        row[f"hd95_{nm}"] = _hd95(pm.transpose(2, 0, 1), gm.transpose(2, 0, 1), spacing[:2])
    for nm, labs in DEFAULT_REGIONS.items():
        pm, gm = np.isin(pred, labs), np.isin(gt, labs)
        row[f"dice_{nm}"] = _dice(pm, gm)
        row[f"dscz_{nm}"] = dsc_z(pm)          # smoothness of the PREDICTION for this region
        row[f"dscz_{nm}_gt"] = dsc_z(gm)       # smoothness of the GT (reference)
    # lesion connected components (26-conn) + volume, pred vs GT
    for tag, arr in (("pred", pred), ("gt", gt)):
        les = np.isin(arr, [3, 4])
        _, n = ndimage.label(les, structure=np.ones((3, 3, 3)))
        row[f"lesion_ncomp_{tag}"] = int(n)
        row[f"lesion_vox_{tag}"] = int(les.sum())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True); ap.add_argument("--gt", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gi = nib.load(a.gt); gt = np.asarray(gi.dataobj).astype(np.int16)
    pred = np.asarray(nib.load(a.pred).dataobj).astype(np.int16)
    if pred.shape != gt.shape:
        # try to match orientation (2D-stacked reconstructions can be transposed)
        for axes in [(1, 2, 0), (2, 0, 1), (0, 2, 1), (2, 1, 0), (1, 0, 2)]:
            if pred.transpose(axes).shape == gt.shape:
                pred = pred.transpose(axes); print(f"  matched pred orientation via transpose {axes}"); break
    if pred.shape != gt.shape:
        sys.exit(f"shape mismatch pred {pred.shape} vs gt {gt.shape}")
    spacing = tuple(float(s) for s in gi.header.get_zooms()[:3])
    row = score(pred, gt, spacing, a.name)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    pd.DataFrame([row]).to_csv(a.out, index=False)
    print(pd.Series(row).to_string())


if __name__ == "__main__":
    main()
