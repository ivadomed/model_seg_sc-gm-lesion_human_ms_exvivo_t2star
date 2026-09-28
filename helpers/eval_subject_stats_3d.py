#!/usr/bin/env python3
"""
Subject-level stats for the 3D ablation tables -- WITHOUT re-prediction.

The paper's 3D models were validated with their own (shuffled) split, and the held-out
predictions are already stored as full volumes at
  paper_results/3D/<fam>/<exp>/.../inference_results/<exp>_validation_EVAL/fold_*/predictions_3d/MagPhase_XXXX.nii.gz
Each MagPhase_XXXX maps to a subject via Dataset011_3D_MagPhase/manifest.json, so these stored
predictions ARE subject-mappable and leakage-free (each volume sits in the fold that held it out).

The paper evaluates 3D on the manually annotated axial slices only. We reproduce that here: for every
2D annotated slice (Dataset021 manifest -> subject/acq/slice-N + its manual label), we extract the same
slice from the matching 3D prediction volume (in-plane matches exactly; slice-N indexes axis 2, i.e.
vol[:, :, N]) and pool voxels WITHIN subject -> one Dice/HD95 per subject per class/region.

Output schema is identical to helpers/eval_subject_stats (metrics_by_subject__<name>.csv), so the SAME
`eval_subject_stats paired` step runs the paired Wilcoxon/t-test across subjects.

CLI:
  python -m helpers.eval_subject_stats_3d score --eval-dir <..._validation_EVAL> --out OUT.csv [--verify]
"""
from __future__ import annotations
import os, sys, glob, re, json, argparse
from collections import defaultdict
import numpy as np, nibabel as nib, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.eval import DEFAULT_LABELS, DEFAULT_REGIONS
from helpers.metric_utils_2d import compute_surface_distances_2d, calculate_hd95

ROOT = os.environ.get("PROJECT_ROOT", "/home/ge.polymtl.ca/pahoa/nih_project")
D11 = f"{ROOT}/nnUNet_data/nnUNet_raw/Dataset011_3D_MagPhase/manifest.json"
D21 = f"{ROOT}/nnUNet_data/nnUNet_raw/Dataset021_2D_MagPhase"
_SL = re.compile(r"slice-(\d+)")
_VID = re.compile(r"(MagPhase_\d+)")


def _annotated_slices():
    """List of (vol_key(subject,acq_tag), slice_N, label_path, subject) for every 2D annotated slice."""
    man = json.load(open(f"{D21}/inference_manifest.json"))
    out = []
    for case, val in man.items():
        subj = val.split("/")[0]
        mo = _SL.search(val)
        acq = re.search(r"(sub-[^/_]+_acq-S\d+)", val)
        if not (mo and acq):
            continue
        idx = case.split("_")[-1]
        out.append(((subj, acq.group(1)), int(mo.group(1)),
                    f"{D21}/labelsTr/2D_MagPhase_{idx}.nii.gz", subj))
    return out


def _volkey_to_id():
    d = json.load(open(D11))
    return {(r["subject"], r["acq"].split("_part")[0]): r["nnunet_id"] for r in d}


def score_config_3d(eval_dir, labels=DEFAULT_LABELS, regions=DEFAULT_REGIONS, verify=False):
    volkey2id = _volkey_to_id()
    slices = _annotated_slices()
    # vol_id -> (pred_path, fold) across all folds (each vol is in exactly its held-out fold)
    vid2pred = {}
    for p in glob.glob(f"{eval_dir}/fold_*/predictions_3d/*.nii.gz"):
        m = _VID.search(os.path.basename(p))
        fold = re.search(r"fold_(\d+)", p)
        if m and fold and "_TTA" not in p:
            vid2pred.setdefault(m.group(1), (p, int(fold.group(1))))
    # group annotated slices by vol id
    by_vol = defaultdict(list)
    for vk, N, lab, subj in slices:
        vid = volkey2id.get(vk)
        if vid:
            by_vol[vid].append((N, lab, subj))

    counts = defaultdict(lambda: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0}))   # subject -> key -> counts
    dists = defaultdict(lambda: defaultdict(list))
    gcounts = defaultdict(lambda: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0}))  # fold -> key -> counts (for verify)
    n_sl = 0
    for vid, items in by_vol.items():
        if vid not in vid2pred:
            continue
        pred_path, fold = vid2pred[vid]
        vol = np.asarray(nib.load(pred_path).dataobj)
        for N, lab_path, subj in items:
            if N >= vol.shape[2] or not os.path.exists(lab_path):
                continue
            ps = vol[:, :, N].astype(np.int16)
            gs = np.asarray(nib.load(lab_path).dataobj).squeeze().astype(np.int16)
            if ps.shape != gs.shape:
                continue

            def acc(key, pm, gm):
                for store, fk in ((counts[subj][key], None), (gcounts[fold][key], None)):
                    store["TP"] += int(np.logical_and(pm, gm).sum())
                    store["FP"] += int(np.logical_and(pm, ~gm).sum())
                    store["FN"] += int(np.logical_and(~pm, gm).sum())
                d = compute_surface_distances_2d(pm, gm, (0.075, 0.075))
                if d is not None:
                    dists[subj][key].append(np.asarray(d))
            for l, nm in labels.items():
                acc(nm, ps == l, gs == l)
            for nm, labs in regions.items():
                acc(nm, np.isin(ps, labs), np.isin(gs, labs))
            n_sl += 1

    keys = list(labels.values()) + list(regions.keys())
    def dice(c):
        # absent class in GT (TP+FN==0) -> undefined -> NaN (excluded), ignore false positives here
        den = 2 * c["TP"] + c["FP"] + c["FN"]; return (2 * c["TP"] / den) if (c["TP"] + c["FN"]) > 0 else np.nan
    rows = []
    for subj in sorted(counts):
        r = {"case": subj}
        for k in keys:
            r[f"dice_{k}"] = dice(counts[subj][k])
            dd = dists[subj][k]; r[f"hd95_{k}"] = calculate_hd95(np.concatenate(dd)) if dd else np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    print(f"  scored {n_sl} annotated slices across {len(df)} subjects")
    if verify:  # paper aggregation = mean over folds of per-fold global dice
        for k in ["WM", "GM", "lesion_WM", "lesion_GM"]:
            vals = [dice(gcounts[f][k]) for f in sorted(gcounts)]
            print(f"    [verify] {k}: per-fold global {[f'{v:.3f}' for v in vals]}  mean {np.nanmean(vals):.3f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--eval-dir", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.cmd == "score":
        df = score_config_3d(a.eval_dir, verify=a.verify)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
