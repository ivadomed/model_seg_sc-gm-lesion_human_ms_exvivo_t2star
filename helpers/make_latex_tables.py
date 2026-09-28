#!/usr/bin/env python3
"""
Generate manuscript-style ablation tables with the SUBJECT-LEVEL statistics (revision).
Reproduces the paper's exact table layout (Normal WM/GM, Lesion WM/GM, Total Average; Dice + HD95;
best-in-bold) but numbers are mean +/- std across the N=12 subjects, with significance vs the table
baseline marked (paired Wilcoxon, Holm-corrected within table): $^{*}$ p<0.05, $^{**}$ p<0.01.

Emits, per table, a change-log entry (reviewer / new LaTeX). Original LaTeX is spliced from main.tex separately.
  python -m helpers.make_latex_tables --root outputs/subject_stats_2D --dim 2D --out OUT.md
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np, pandas as pd

CLASSES = [("Normal WM", "WM"), ("Normal GM", "GM"), ("Lesion WM", "lesion_WM"), ("Lesion GM", "lesion_GM")]

# table spec: label -> (caption, baseline_config, [(manuscript_row_name, config_name), ...])
SPEC_2D = {
 "mag_phase_2D": ("Magnitude and Phase 2D Experiment", "base",
    [("2D Mag+Phase","base"),("2D Mag","mag")]),
 "prepro_2D": ("Preprocessing 2D Experiment", "base",
    [("2D Raw","base"),("2D Mag Prepro","mag_prepro"),("2D Phase Prepro","phase_prepro")]),
 "aug_2D": ("Data Augmentation 2D Experiment", "base",
    [("2D No Aug","no_aug"),("2D Aug1","base"),("2D Aug2","aug2"),("2D Aug3","aug3")]),
 "soft_2D": ("Soft Segmentation 2D Experiment", "base",
    [("2D Hard","base"),("2D Soft1","soft1"),("2D Soft2","soft2"),("2D Soft3","soft3")]),
 "opti_2D": ("Optimizer 2D Experiment", "base",
    [("2D SGD","sgd"),("2D AdamW","base")]),
}
SPEC_3D = {
 "mag_phase_3D": ("Magnitude and Phase 3D Experiment", "base3d",
    [("3D Mag+Phase","base3d"),("3D Mag","mag")]),
 "prepro_3D": ("Preprocessing 3D Experiment", "base3d",
    [("3D Raw","base3d"),("3D Phase Prepro","phase_prepro"),("3D Mag Prepro","mag_prepro")]),
 "aug_3D": ("Data Augmentation 3D Experiment", "base3d",
    [("3D No Aug","base3d"),("3D Aug1","aug1"),("3D Aug2","aug2"),("3D Aug3","aug3")]),
 "otsu_3D": ("Otsu Masking 3D Experiment", "base3d",
    [("3D No Otsu","base3d"),("3D Otsu","otsu")]),
 "soft_3D": ("Soft Segmentation 3D Experiment", "base3d",
    [("3D Hard","base3d"),("3D Soft1","soft1"),("3D Soft2","soft2"),("3D Soft3","soft3")]),
}


def load(root):
    bys, pv = {}, {}
    import glob
    for f in glob.glob(f"{root}/by_subject/metrics_by_subject__*.csv"):
        bys[os.path.basename(f)[len("metrics_by_subject__"):-4]] = pd.read_csv(f)
    for f in glob.glob(f"{root}/stats/stats__*.csv"):
        s = pd.read_csv(f)
        for _, r in s.iterrows():
            pv[(r["config"], r["metric"])] = r.get("p_wilcoxon_holm", np.nan)
    return bys, pv


def col_stats(bys, cfg):
    """Return dict metric-> (mean,std) incl. total-average over the 4 classes, Dice & HD95."""
    d = bys[cfg]; out = {}
    dice_cols, hd_cols = [], []
    for _, key in CLASSES:
        out[f"dice_{key}"] = (d[f"dice_{key}"].mean(), d[f"dice_{key}"].std(ddof=1)); dice_cols.append(f"dice_{key}")
        out[f"hd95_{key}"] = (d[f"hd95_{key}"].mean(), d[f"hd95_{key}"].std(ddof=1)); hd_cols.append(f"hd95_{key}")
    out["dice_Total"] = (d[dice_cols].mean(axis=1).mean(), d[dice_cols].mean(axis=1).std(ddof=1))
    out["hd95_Total"] = (d[hd_cols].mean(axis=1).mean(), d[hd_cols].mean(axis=1).std(ddof=1))
    return out


def gen_table(label, caption, baseline, rows, bys, pv):
    metrics = [f"dice_{k}" for _, k in CLASSES] + ["dice_Total"]
    hmetrics = [f"hd95_{k}" for _, k in CLASSES] + ["hd95_Total"]
    stats = {name: col_stats(bys, cfg) for name, cfg in rows if cfg in bys}
    # best per column: max dice, min hd95
    best = {}
    for m in metrics: best[m] = max(stats.values(), key=lambda s: (s[m][0] if not np.isnan(s[m][0]) else -9))[m][0]
    for m in hmetrics: best[m] = min(stats.values(), key=lambda s: (s[m][0] if not np.isnan(s[m][0]) else 9))[m][0]
    def cell(name, cfg, m, prec):
        mu, sd = stats[name][m]
        star = ""
        if m.startswith("dice_") and m != "dice_Total" and cfg != baseline:
            key = m.replace("dice_", "dice_" if False else "dice_")  # metric name in stats = dice_<key>
            p = pv.get((cfg, m), np.nan)
            star = "^{**}" if p < 0.01 else "^{*}" if p < 0.05 else ""
        val = f"{mu:.3f} \\pm {sd:.3f}" if prec == 3 else f"{mu:.2f} \\pm {sd:.2f}"
        body = f"\\mathbf{{{val}}}" if abs(mu - best[m]) < 1e-9 else val
        return f"${body}{star}$"
    hdr = " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{c}}}}}" for c, _ in CLASSES) + " & \\multicolumn{2}{c}{\\textbf{Total Average}}"
    cmid = " ".join(f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(5))
    base_name = next((n for n, c in rows if c == baseline), rows[0][0])
    lines = [f"\\begin{{table*}}[t]", "    \\centering",
             f"    \\caption{{{caption} -- subject-level Dice and HD95 (mean $\\pm$ std across $N{{=}}12$ subjects). "
             f"Best in bold; $^{{*}}$/$^{{**}}$ denote $p<0.05$/$p<0.01$ vs.\\ {base_name} (paired Wilcoxon, Holm-corrected).}}",
             f"    \\label{{tab:{label}}}", "    \\resizebox{\\textwidth}{!}{%", "        \\begin{tabular}{lcccccccccc}",
             "            \\toprule", f"            & {hdr} \\\\", f"            {cmid}",
             "            \\textbf{Config} & " + " & ".join(["\\textbf{Dice} & \\textbf{HD95}"]*5) + " \\\\",
             "            \\midrule"]
    for name, cfg in rows:
        if cfg not in bys: continue
        cells = []
        for dm, hm in zip(metrics, hmetrics):
            cells.append(cell(name, cfg, dm, 3)); cells.append(cell(name, cfg, hm, 2))
        lines.append(f"            {name} & " + " & ".join(cells) + " \\\\")
    lines += ["            \\bottomrule", "        \\end{tabular}%", "    }", "\\end{table*}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--dim", choices=["2D","3D"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bys, pv = load(a.root)
    spec = SPEC_2D if a.dim == "2D" else SPEC_3D
    blocks = []
    for label, (cap, base, rows) in spec.items():
        if not any(cfg in bys for _, cfg in rows): continue
        blocks.append(f"% ===== tab:{label} =====\n" + gen_table(label, cap, base, rows, bys, pv))
    open(a.out, "w").write("\n\n".join(blocks) + "\n")
    print(f"wrote {len(blocks)} tables -> {a.out}")


if __name__ == "__main__":
    main()
