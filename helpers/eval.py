#!/usr/bin/env python3
"""
Unified segmentation evaluator shared by the 2D and 3D pipelines.

Given a folder of PREDICTIONS and a folder of GROUND-TRUTH NIfTIs (matching filenames), it writes a
uniform pair of CSVs next to the predictions:
  metrics_casewise.csv  - one row per case: Dice + HD95 for each class and each region
  metrics_summary.csv   - per-metric mean/std/n across cases, plus dataset-level (global) Dice
And, given a parent dir of fold_*/ subdirs each with metrics_casewise.csv, it writes a cross-fold
metrics_crossfold.csv (mean/std of the per-fold means).

Dice is computed directly (dimension-agnostic). HD95 aggregates per-2D-slice surface distances
(works for a 2D slice or a 3D volume), reusing helpers/metric_utils_2d. Metric conventions match
the existing 2D pipeline so 2D and 3D results are directly comparable.

Default label scheme (nnUNet 4-class): {1:WM, 2:GM, 3:lesion_WM, 4:lesion_GM}
Default regions: cord=[1,2,3,4], WM=[1,3], GM=[2,4], lesion=[3,4]
CLI:
  python -m helpers.eval --pred-dir P --gt-dir G [--labels 1,2,3,4] [--out-dir P]
  python -m helpers.eval --crossfold PARENT_DIR
"""
from __future__ import annotations
import os, sys, glob, argparse, json
import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.metric_utils_2d import compute_surface_distances_2d, calculate_hd95
from helpers.stats_utils import MetricTracker

DEFAULT_LABELS = {1: "WM", 2: "GM", 3: "lesion_WM", 4: "lesion_GM"}
DEFAULT_REGIONS = {"cord": [1, 2, 3, 4], "WM": [1, 3], "GM": [2, 4], "lesion": [3, 4]}


def _dice(p_bin, g_bin):
    ps, gs = p_bin.sum(), g_bin.sum()
    if gs == 0:
        return np.nan if ps == 0 else 0.0
    return 2.0 * np.logical_and(p_bin, g_bin).sum() / (ps + gs + 1e-8)


def _hd95(p_bin, g_bin, spacing):
    """Per-2D-slice surface distances aggregated -> HD95. Handles 2D (single slice) and 3D volumes."""
    if p_bin.ndim == 2:
        p_bin, g_bin = p_bin[None], g_bin[None]
    dists = []
    for z in range(p_bin.shape[0]):
        d = compute_surface_distances_2d(p_bin[z], g_bin[z], spacing)
        if d is None:                      # one empty, the other not on this slice
            continue
        dists.append(np.asarray(d))
    if not dists:
        return np.nan
    return calculate_hd95(np.concatenate(dists))


def score_case(pred, gt, labels=DEFAULT_LABELS, regions=DEFAULT_REGIONS, spacing=(1.0, 1.0)):
    """Return a dict of dice_<name> / hd95_<name> for each class and region."""
    m = {}
    for lab, name in labels.items():
        m[f"dice_{name}"] = _dice(pred == lab, gt == lab)
        m[f"hd95_{name}"] = _hd95(pred == lab, gt == lab, spacing)
    for name, labs in regions.items():
        pm, gm = np.isin(pred, labs), np.isin(gt, labs)
        m[f"dice_{name}"] = _dice(pm, gm)
        m[f"hd95_{name}"] = _hd95(pm, gm, spacing)
    return m


def evaluate(pred_dir, gt_dir, out_dir=None, labels=DEFAULT_LABELS, regions=DEFAULT_REGIONS):
    out_dir = out_dir or pred_dir
    os.makedirs(out_dir, exist_ok=True)
    tracker = MetricTracker()
    preds = sorted(glob.glob(os.path.join(pred_dir, "*.nii.gz")))
    n = 0
    for p in preds:
        cid = os.path.basename(p)[:-7]
        g = os.path.join(gt_dir, os.path.basename(p))
        if not os.path.exists(g):
            continue
        pi, gi = nib.load(p), nib.load(g)
        pa = np.asarray(pi.dataobj).astype(np.int16)
        ga = np.asarray(gi.dataobj).astype(np.int16)
        if pa.shape != ga.shape:
            print(f"  ! shape mismatch {cid}: pred {pa.shape} vs gt {ga.shape}, skipping"); continue
        spacing = tuple(float(s) for s in pi.header.get_zooms()[:3])
        row = {"case": cid}
        row.update(score_case(pa, ga, labels, regions, spacing))
        tracker.add_case_metric(row)
        tracker.update_counts(pa, ga, labels=list(labels))
        for name, labs in regions.items():
            tracker.update_counts_combo(pa, ga, labs, name)
        n += 1
    if n == 0:
        print(f"  no matched pred/gt pairs in {pred_dir} vs {gt_dir}"); return None
    cw = os.path.join(out_dir, "metrics_casewise.csv")
    tracker.save_casewise_csv(cw)
    # summary: per-metric mean/std + global dice
    df = pd.DataFrame(tracker.case_metrics).drop(columns=["case"])
    summ = df.agg(["mean", "std", "count"]).T.reset_index().rename(columns={"index": "metric"})
    name_by_lab = {**{l: nm for l, nm in labels.items()}, **{nm: nm for nm in regions}}
    gd = {f"global_dice_{name_by_lab.get(l, l)}": (2*c['TP'])/((2*c['TP'])+c['FP']+c['FN'])
          for l, c in tracker.global_counts.items() if ((2*c['TP'])+c['FP']+c['FN']) > 0}
    sp = os.path.join(out_dir, "metrics_summary.csv")
    summ.to_csv(sp, index=False)
    pd.Series(gd, name="global_dice").to_csv(os.path.join(out_dir, "metrics_global_dice.csv"))
    print(f"  evaluated {n} cases -> {cw} , {sp}")
    return summ


def compare(csv_a, csv_b, out=None, name_a="A", name_b="B"):
    """
    Paired statistical comparison of two methods from their metrics_casewise.csv files.

    Design: the unit of observation is the CASE/SUBJECT (cross-subject), not the fold. With
    subject-level k-fold CV each subject is held out exactly once, so each subject contributes one
    metric per method; we pair the two methods on the SAME cases and test across cases (N = #cases).
    For each shared metric column (dice_* / hd95_*) we report the paired Student t-test (parametric)
    AND the Wilcoxon signed-rank test (non-parametric, robust to the non-normal/bounded Dice & HD95).
    NaN pairs (e.g. empty-GT cases) are dropped per metric. Operates only on the CSVs -> no inference.
    """
    from scipy import stats
    a = pd.read_csv(csv_a).set_index("case")
    b = pd.read_csv(csv_b).set_index("case")
    common = a.index.intersection(b.index)
    metrics = [c for c in a.columns if c in b.columns and (c.startswith("dice_") or c.startswith("hd95_"))]
    rows = []
    for m in metrics:
        va, vb = a.loc[common, m], b.loc[common, m]
        mask = va.notna() & vb.notna()
        x, y = va[mask].to_numpy(float), vb[mask].to_numpy(float)
        r = {"metric": m, f"mean_{name_a}": x.mean() if len(x) else np.nan,
             f"mean_{name_b}": y.mean() if len(y) else np.nan,
             "mean_diff": (x - y).mean() if len(x) else np.nan, "n_pairs": len(x)}
        if len(x) >= 2:
            t, pt = stats.ttest_rel(x, y)
            r["t_stat"], r["p_ttest"] = t, pt
            if np.allclose(x, y):
                r["p_wilcoxon"] = 1.0
            else:
                try:
                    r["p_wilcoxon"] = stats.wilcoxon(x, y).pvalue
                except ValueError:
                    r["p_wilcoxon"] = np.nan
        rows.append(r)
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(out, index=False)
        print(f"  comparison ({name_a} vs {name_b}, n={int(df['n_pairs'].max()) if len(df) else 0} cases) -> {out}")
    return df


def crossfold(parent):
    rows = []
    for cw in sorted(glob.glob(os.path.join(parent, "fold_*", "metrics_casewise.csv"))):
        fold = os.path.basename(os.path.dirname(cw))
        df = pd.read_csv(cw)
        mean = df.drop(columns=["case"]).mean(numeric_only=True)
        mean["fold"] = fold
        rows.append(mean)
    if not rows:
        print(f"  no fold_*/metrics_casewise.csv under {parent}"); return
    allf = pd.DataFrame(rows).set_index("fold")
    out = allf.agg(["mean", "std"]).T
    p = os.path.join(parent, "metrics_crossfold.csv")
    out.to_csv(p)
    print(f"  cross-fold summary ({len(rows)} folds) -> {p}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir"); ap.add_argument("--gt-dir"); ap.add_argument("--out-dir")
    ap.add_argument("--labels", default="1,2,3,4")
    ap.add_argument("--crossfold", help="parent dir with fold_*/metrics_casewise.csv")
    ap.add_argument("--compare", nargs=2, metavar=("A_casewise.csv", "B_casewise.csv"),
                    help="paired per-case t-test + Wilcoxon between two methods")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.compare:
        df = compare(a.compare[0], a.compare[1], out=a.out or "comparison.csv", name_a=a.names[0], name_b=a.names[1])
        print(df.to_string(index=False))
    elif a.crossfold:
        crossfold(a.crossfold)
    else:
        labs = {int(x): DEFAULT_LABELS.get(int(x), f"class{x}") for x in a.labels.split(",")}
        evaluate(a.pred_dir, a.gt_dir, a.out_dir, labels=labs)
