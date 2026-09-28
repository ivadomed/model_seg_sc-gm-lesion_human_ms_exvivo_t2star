#!/usr/bin/env python3
"""
Score EVERY 3D model against the single manually-annotated 3D test volume (sub-TNU026),
build a leaderboard, and run the statistical tests.

    ============================================================================
    READ THIS FIRST -- what the statistics mean, and what they do NOT mean.
    ============================================================================
    We have ONE annotated test volume (N = 1 subject). The repo's normal
    comparison (helpers/eval.py::compare) pairs methods ACROSS SUBJECTS, so with
    one subject it is undefined -- a paired test on a single pair has no power.

    To get any statistics at all we change the UNIT OF OBSERVATION from
    "subject" to "2D slice" along one axis of the volume. That turns N = 1 volume
    into N ~= (number of slices) paired observations, which lets us:

      * put a bootstrap 95% CI around each model's mean Dice / HD95, and
      * run a paired slice-wise test (t-test + Wilcoxon) of every model against
        a reference model.

    >>> HARD CAVEAT, state it wherever these numbers are reported <<<
    Adjacent slices are spatially correlated (they are NOT independent samples),
    so the slice-wise p-values are ANTI-CONSERVATIVE: they OVERSTATE
    significance. These tests measure WITHIN-VOLUME consistency of the difference
    between two models on THIS specimen. They are NOT evidence of population-level
    generalization -- that would require more annotated subjects. Treat a
    "significant" slice-wise result as "the two models differ consistently across
    this volume", never as "model A is better in general".
    ============================================================================

Outputs (under --out):
  leaderboard.csv            one row per model: whole-volume Dice/HD95 per region,
                             + slice-mean and bootstrap 95% CI, sorted by --primary
  pairwise_vs_reference.csv  paired slice-wise t-test + Wilcoxon of each model vs the
                             reference, multiplicity-corrected (Holm + Benjamini-Hochberg)
  STATS_NOTES.md             the caveat above, written next to the numbers
and, inside each model's prediction folder:
  metrics_casewise.csv       whole-volume metrics (same schema as helpers/eval.py)
  metrics_slicewise.csv      one row per slice (the statistical unit)
  bootstrap_ci.csv           per-region mean + 95% CI from the slice bootstrap
"""
from __future__ import annotations
import os, sys, glob, json, argparse
import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.eval import score_case, DEFAULT_LABELS, DEFAULT_REGIONS, compare


# ---------------------------------------------------------------------------- #
#  per-model scoring: whole volume + slice-wise (the statistical unit)
# ---------------------------------------------------------------------------- #
def score_model(pred_path, gt_arr, spacing, axis, labels, regions, out_dir):
    """Whole-volume + slice-wise metrics for one prediction. Writes CSVs into out_dir."""
    pa = np.asarray(nib.load(pred_path).dataobj).astype(np.int16)
    if pa.shape != gt_arr.shape:
        print(f"  ! shape mismatch {os.path.basename(out_dir)}: {pa.shape} vs {gt_arr.shape}, skip")
        return None

    # in-plane spacing = the two axes that are NOT the slicing axis
    in_plane = tuple(s for i, s in enumerate(spacing) if i != axis)

    # whole volume (single row) -- identical schema to helpers/eval.py
    whole = {"case": os.path.basename(out_dir)}
    whole.update(score_case(pa, gt_arr, labels, regions, spacing=in_plane))
    pd.DataFrame([whole]).to_csv(os.path.join(out_dir, "metrics_casewise.csv"), index=False)

    # slice-wise: move slicing axis to front, one row per slice
    ps, gs = np.moveaxis(pa, axis, 0), np.moveaxis(gt_arr, axis, 0)
    rows = []
    for z in range(ps.shape[0]):
        r = {"case": f"slice_{z:04d}"}          # 'case' col so helpers.eval.compare() works verbatim
        r.update(score_case(ps[z], gs[z], labels, regions, spacing=in_plane))
        rows.append(r)
    sw = pd.DataFrame(rows)
    sw.to_csv(os.path.join(out_dir, "metrics_slicewise.csv"), index=False)
    return whole, sw


# ---------------------------------------------------------------------------- #
#  bootstrap CI over slices (percentile bootstrap of the mean)
# ---------------------------------------------------------------------------- #
def bootstrap_ci(slicewise_df, n_boot, seed, alpha=0.05):
    """Percentile bootstrap 95% CI of the slice-mean for every dice_/hd95_ column."""
    rng = np.random.default_rng(seed)
    metrics = [c for c in slicewise_df.columns if c.startswith(("dice_", "hd95_"))]
    n = len(slicewise_df)
    idx = rng.integers(0, n, size=(n_boot, n))          # shared resampling across metrics
    out = {}
    for m in metrics:
        v = slicewise_df[m].to_numpy(float)
        boot_means = np.nanmean(v[idx], axis=1)          # NaN = empty-on-both slices, ignored
        out[m] = {
            "slice_mean": np.nanmean(v),
            "ci_lo": np.nanpercentile(boot_means, 100 * alpha / 2),
            "ci_hi": np.nanpercentile(boot_means, 100 * (1 - alpha / 2)),
            "n_slices_scored": int(np.isfinite(v).sum()),
        }
    return pd.DataFrame(out).T.reset_index().rename(columns={"index": "metric"})


# ---------------------------------------------------------------------------- #
#  multiplicity correction
# ---------------------------------------------------------------------------- #
def _holm(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    order = np.argsort(p[ok])
    m = ok.sum()
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[ok][i])
        adj[i] = min(running, 1.0)
    out[ok] = adj
    return out


def _bh(pvals):
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    out = np.full_like(p, np.nan)
    pv = p[ok]; m = pv.size
    order = np.argsort(pv)
    adj = np.empty(m)
    running = 1.0
    for k in range(m - 1, -1, -1):
        i = order[k]
        running = min(running, pv[i] * m / (k + 1))
        adj[i] = running
    out[ok] = adj
    return out


# ---------------------------------------------------------------------------- #
STATS_NOTES = """# Statistics for the single-volume 3D test evaluation

**One annotated volume (sub-TNU026, N = 1 subject).** Cross-subject paired tests
(the repo default) are undefined on a single subject, so the statistical unit here
is the **2D slice** along axis `{axis}` of the volume ({n_slices} slices).

## What is reported
- **leaderboard.csv** - each model's whole-volume Dice/HD95 per region, plus the
  slice-mean and a **percentile bootstrap 95% CI** ({n_boot} resamples over slices).
  The CI reflects how much the score varies across slices of *this* volume.
- **pairwise_vs_reference.csv** - each model vs the reference model
  (`{reference}`): paired **Student t-test** and **Wilcoxon signed-rank** across the
  shared slices, per metric. p-values are corrected for the {n_models} model
  comparisons with **Holm** (`p_holm`) and **Benjamini-Hochberg** (`p_bh`).

## What these numbers mean -- and do NOT mean
Adjacent slices are spatially correlated, so they are **not independent samples**.
The slice-wise p-values are therefore **anti-conservative (they overstate
significance)**. Read them as: *"do the two models differ consistently across this
one volume?"* -- a within-specimen consistency check. They are **NOT** evidence of
population-level generalization; that needs more annotated subjects. Never report a
significant slice-wise result as "model A is better in general".
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", required=True,
                    help="root holding <family>/<exp>/<pred-tag>/ prediction folders")
    ap.add_argument("--pred-glob", default="*/*/predict_*",
                    help="glob under --pred-root matching each model's prediction folder")
    ap.add_argument("--gt", required=True, help="ground-truth 3D NIfTI")
    ap.add_argument("--axis", type=int, default=None,
                    help="slicing axis for the slice-wise unit (default: longest axis)")
    ap.add_argument("--primary", default="dice_cord", help="leaderboard sort key")
    ap.add_argument("--reference", default=None,
                    help="model id to compare all others against (default: best on --primary)")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    gi = nib.load(a.gt)
    gt = np.asarray(gi.dataobj).astype(np.int16)
    spacing = tuple(float(s) for s in gi.header.get_zooms()[:3])
    axis = a.axis if a.axis is not None else int(np.argmax(gt.shape))
    print(f"[stats] gt shape={gt.shape} spacing={spacing} -> slicing axis={axis} "
          f"({gt.shape[axis]} slices) as the statistical unit")

    # ---- discover predictions, score each model -------------------------------
    pred_dirs = sorted(glob.glob(os.path.join(a.pred_root, a.pred_glob)))
    whole_rows, slicewise = {}, {}
    for d in pred_dirs:
        preds = glob.glob(os.path.join(d, "*.nii.gz"))
        if not preds:
            continue
        mid = os.path.relpath(d, a.pred_root).replace(os.sep, "/")
        res = score_model(preds[0], gt, spacing, axis, DEFAULT_LABELS, DEFAULT_REGIONS, d)
        if res is None:
            continue
        whole, sw = res
        whole_rows[mid] = whole
        slicewise[mid] = sw
        ci = bootstrap_ci(sw, a.n_boot, a.seed)
        ci.to_csv(os.path.join(d, "bootstrap_ci.csv"), index=False)
        print(f"  scored {mid}: {a.primary}={whole.get(a.primary, float('nan')):.4f}")

    if not whole_rows:
        sys.exit(f"no scorable predictions under {a.pred_root}/{a.pred_glob}")

    # ---- leaderboard: whole-volume metrics + slice bootstrap CI on primary ----
    lb = pd.DataFrame(whole_rows).T
    lb.index.name = "model"
    for mid in lb.index:
        ci = bootstrap_ci(slicewise[mid], a.n_boot, a.seed).set_index("metric")
        if a.primary in ci.index:
            lb.loc[mid, f"{a.primary}__slice_mean"] = ci.loc[a.primary, "slice_mean"]
            lb.loc[mid, f"{a.primary}__ci_lo"] = ci.loc[a.primary, "ci_lo"]
            lb.loc[mid, f"{a.primary}__ci_hi"] = ci.loc[a.primary, "ci_hi"]
    lb = lb.sort_values(a.primary, ascending=False)
    lb.to_csv(os.path.join(a.out, "leaderboard.csv"))
    print(f"[stats] leaderboard -> {os.path.join(a.out, 'leaderboard.csv')}")

    # ---- pairwise slice-wise tests vs the reference model ---------------------
    reference = a.reference or lb.index[0]
    if reference not in slicewise:
        sys.exit(f"reference model {reference!r} not among scored models")
    ref_csv = os.path.join(a.pred_root, reference, "metrics_slicewise.csv")
    rows = []
    for mid in lb.index:
        if mid == reference:
            continue
        cmp = compare(os.path.join(a.pred_root, mid, "metrics_slicewise.csv"), ref_csv,
                      out=None, name_a=mid, name_b=reference)
        cmp.insert(0, "model", mid)
        rows.append(cmp)
    if rows:
        pair = pd.concat(rows, ignore_index=True)
        # correct across models, within each metric
        for col, fn, new in [("p_ttest", _holm, "p_ttest_holm"), ("p_ttest", _bh, "p_ttest_bh"),
                             ("p_wilcoxon", _holm, "p_wilcoxon_holm"), ("p_wilcoxon", _bh, "p_wilcoxon_bh")]:
            if col in pair:
                pair[new] = pair.groupby("metric")[col].transform(lambda s: fn(s.to_numpy()))
        pair.to_csv(os.path.join(a.out, "pairwise_vs_reference.csv"), index=False)
        print(f"[stats] pairwise vs {reference} -> {os.path.join(a.out, 'pairwise_vs_reference.csv')}")
    else:
        print(f"[stats] only one model scored ({reference}); no pairwise comparison written")

    with open(os.path.join(a.out, "STATS_NOTES.md"), "w") as f:
        f.write(STATS_NOTES.format(axis=axis, n_slices=gt.shape[axis], n_boot=a.n_boot,
                                   reference=reference, n_models=max(len(lb) - 1, 0)))
    print(f"[stats] read {os.path.join(a.out, 'STATS_NOTES.md')} before reporting these numbers")


if __name__ == "__main__":
    main()
