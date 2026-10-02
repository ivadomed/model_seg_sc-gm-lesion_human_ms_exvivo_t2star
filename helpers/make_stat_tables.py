#!/usr/bin/env python3
"""
Build the revision's subject-level tables (markdown) from the by_subject/ + stats/ CSVs produced by
reproduce/run_subject_stats_campaign_{2D,3D}.sh. Per config: per-subject mean +/- std and 95% CI (t, N=12),
and Delta vs baseline with paired Wilcoxon (Holm-corrected). Unit = subject (spinal cord).

  python -m helpers.make_stat_tables --root outputs/subject_stats_2D --baseline base --out TABLE.md
"""
from __future__ import annotations
import os, sys, glob, argparse
import numpy as np, pandas as pd
from scipy import stats

METRICS = ["dice_WM", "dice_GM", "dice_lesion_WM", "dice_lesion_GM"]


def ci95(x):
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    se = x.std(ddof=1) / np.sqrt(len(x))
    h = se * stats.t.ppf(0.975, len(x) - 1)
    return (x.mean() - h, x.mean() + h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bys = {os.path.basename(f)[len("metrics_by_subject__"):-4]: pd.read_csv(f)
           for f in glob.glob(f"{a.root}/by_subject/metrics_by_subject__*.csv")}
    if a.baseline not in bys:
        sys.exit(f"baseline {a.baseline} not found in {a.root}/by_subject")
    # gather Wilcoxon-Holm p per (config, metric) from stats/
    pv = {}
    for sf in glob.glob(f"{a.root}/stats/stats__*.csv"):
        s = pd.read_csv(sf)
        for _, r in s.iterrows():
            pv[(r["config"], r["metric"])] = r.get("p_wilcoxon_holm", np.nan)

    lines = [f"# Subject-level results ({os.path.basename(a.root)})",
             f"Unit = subject (N=12 spinal cords). Mean +/- std across subjects; 95% CI (t).",
             f"Delta vs **{a.baseline}**; p = paired Wilcoxon signed-rank, Holm-corrected within table. * p<0.05 ** p<0.01.",
             ""]
    order = [a.baseline] + [c for c in sorted(bys) if c != a.baseline]
    for m in METRICS:
        lines.append(f"\n### {m}")
        lines.append("| config | mean±std | 95% CI | Δ vs base | p (Holm) |")
        lines.append("|---|---|---|---|---|")
        bmean = bys[a.baseline][m].mean()
        for c in order:
            x = bys[c][m].to_numpy(float)
            lo, hi = ci95(x)
            d = "—" if c == a.baseline else f"{x[np.isfinite(x)].mean()-bmean:+.3f}"
            p = pv.get((c, m), np.nan)
            star = "**" if p < 0.01 else "*" if p < 0.05 else ""
            ps = "—" if c == a.baseline else (f"{p:.3f}{star}" if np.isfinite(p) else "n/a")
            lines.append(f"| {c} | {np.nanmean(x):.3f}±{np.nanstd(x,ddof=1):.3f} | [{lo:.3f}, {hi:.3f}] | {d} | {ps} |")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {a.out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
