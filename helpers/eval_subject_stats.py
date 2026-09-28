#!/usr/bin/env python3
"""
Subject-level statistics for the ablation tables (reviewer concern #2).

    ============================================================================
    WHY THIS EXISTS -- the statistical unit must be the SUBJECT (spinal cord).
    ============================================================================
    The paper reports GLOBAL (voxel-pooled) Dice: a single number per config, so
    it carries no per-subject distribution and cannot support a significance test.
    The saved per-slice CSVs are the wrong unit too -- slices from the same cord
    are correlated, exactly the non-independence Reviewer 4 called out.

    This module re-scores the held-out cross-validation predictions and pools
    voxels WITHIN each subject -> one robust Dice/HD95 per subject per class.
    Pooling within-subject keeps the stability that motivated the global score
    (a rare class absent on one slice is fine as long as the subject has it
    somewhere) while giving the SUBJECT as the independent unit. We then compare
    two configs with a PAIRED test across the shared subjects (N = #subjects):
    paired Student t-test + Wilcoxon signed-rank, with Holm & Benjamini-Hochberg
    correction across the variants in a table.

    Per-subject Dice is the correct micro-average: Dice_s = 2*TP_s/(2*TP_s+FP_s+FN_s),
    with TP/FP/FN summed over all of subject s's held-out slices. It is NOT the
    mean of per-slice Dice (that macro-average is what makes rare classes degenerate).
    ============================================================================

Recoverable configs (manifest present on disk):
  * 2D id-21 configs -> Dataset021_2D_MagPhase/inference_manifest.json  (dict: idx -> "sub-XXX/...slice-N")
  * 3D id-11 configs -> Dataset011_3D_MagPhase/manifest.json            (list of records w/ .subject)
id-22/23/24 (2D mag-only, merge_lesions, wm_gm) and id-12 (3D mag_only) have no
manifest on disk and cannot be subject-mapped without rebuilding those datasets.

CLI
  # score one config's held-out predictions into a per-subject CSV
  python -m helpers.eval_subject_stats score \
      --pred-root .../<exp>_FULL_EVAL --pred-glob 'fold_*/predictions_nifti/*.nii.gz' \
      --labels-dir nnUNet_data/nnUNet_raw/Dataset021_2D_MagPhase/labelsTr \
      --manifest  nnUNet_data/nnUNet_raw/Dataset021_2D_MagPhase/inference_manifest.json \
      --out OUT/metrics_by_subject__<exp>.csv

  # run a whole ablation table: score every config, pair each variant vs the baseline
  python -m helpers.eval_subject_stats table --spec table_spec.json --out OUT/
"""
from __future__ import annotations
import os, sys, glob, json, re, argparse
from collections import defaultdict
import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.eval import DEFAULT_LABELS, DEFAULT_REGIONS, compare
from helpers.eval_test_volume import _holm, _bh
from helpers.metric_utils_2d import compute_surface_distances_2d, calculate_hd95

_IDX = re.compile(r"(\d{4})(?=\.nii\.gz$)")


def _load_manifest_idx_to_subject(manifest_path):
    """Return {4-digit-index-str -> subject}. Handles the 2D dict manifest and the 3D list manifest."""
    m = json.load(open(manifest_path))
    out = {}
    if isinstance(m, dict):                       # 2D: {"2D_MagPhase_0007": "sub-XXX/..."}
        for k, v in m.items():
            idx = _IDX.search(k + ".nii.gz")
            if idx:
                out[idx.group(1)] = v.split("/")[0]
    else:                                         # 3D: [{"nnunet_id":"MagPhase_0007","subject":"sub-XXX",...}]
        for rec in m:
            idx = _IDX.search(str(rec.get("nnunet_id", "")) + ".nii.gz")
            if idx:
                out[idx.group(1)] = rec["subject"]
    return out


def score_config_by_subject(pred_glob, labels_dir, manifest_path, spacing_mm,
                            labels=DEFAULT_LABELS, regions=DEFAULT_REGIONS):
    """Pool voxels within subject over all matched prediction slices; return a per-subject DataFrame."""
    idx2subj = _load_manifest_idx_to_subject(manifest_path)
    label_prefix = None
    # infer the label filename prefix from the labels dir (e.g. '2D_MagPhase_')
    for f in os.listdir(labels_dir):
        mo = _IDX.search(f)
        if mo:
            label_prefix = f[:mo.start()]
            break
    if label_prefix is None:
        raise RuntimeError(f"no NNNN.nii.gz labels found in {labels_dir}")

    # per-subject accumulators
    counts = defaultdict(lambda: defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0}))
    dists = defaultdict(lambda: defaultdict(list))   # subject -> key -> list of surface-distance arrays
    seen_idx = defaultdict(set)                      # guard against double-counting an index

    n = 0
    for p in sorted(glob.glob(pred_glob)):
        mo = _IDX.search(os.path.basename(p))
        if not mo:
            continue
        idx = mo.group(1)
        subj = idx2subj.get(idx)
        if subj is None:
            continue
        if idx in seen_idx[subj]:                    # same held-out slice twice -> skip dup
            continue
        gt_path = os.path.join(labels_dir, f"{label_prefix}{idx}.nii.gz")
        if not os.path.exists(gt_path):
            continue
        pa = np.asarray(nib.load(p).dataobj).astype(np.int16)
        ga = np.asarray(nib.load(gt_path).dataobj).astype(np.int16)
        if pa.shape != ga.shape:
            print(f"  ! shape mismatch idx {idx}: {pa.shape} vs {ga.shape}, skip"); continue
        pa, ga = np.squeeze(pa), np.squeeze(ga)      # 2D slices may carry a singleton z

        def _accum(key, pm, gm):
            c = counts[subj][key]
            c["TP"] += int(np.logical_and(pm, gm).sum())
            c["FP"] += int(np.logical_and(pm, ~gm).sum())
            c["FN"] += int(np.logical_and(~pm, gm).sum())
            d = compute_surface_distances_2d(pm, gm, spacing_mm)   # None if exactly one is empty on this slice
            if d is not None:
                dists[subj][key].append(np.asarray(d))

        for lab, name in labels.items():
            _accum(name, pa == lab, ga == lab)
        for name, labs in regions.items():
            _accum(name, np.isin(pa, labs), np.isin(ga, labs))
        seen_idx[subj].add(idx)
        n += 1

    # build per-subject rows
    rows = []
    keys = list(labels.values()) + list(regions.keys())
    for subj in sorted(counts):
        row = {"case": subj, "n_slices": len(seen_idx[subj])}   # 'case' col so helpers.eval.compare() works verbatim
        for key in keys:
            c = counts[subj][key]
            gt = c["TP"] + c["FN"]                 # GT voxels of this class in the subject
            denom = 2 * c["TP"] + c["FP"] + c["FN"]
            # Dice undefined when the class is ABSENT from the subject's GT -> exclude (NaN),
            # regardless of false positives (those belong to a separate FP/specificity metric).
            row[f"dice_{key}"] = (2 * c["TP"] / denom) if gt > 0 else np.nan
            dd = dists[subj][key]
            row[f"hd95_{key}"] = calculate_hd95(np.concatenate(dd)) if dd else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"  scored {n} slices -> {len(df)} subjects")
    return df


def run_table(spec, out_dir, spacing_mm):
    """spec: {name, labels_dir, manifest, spacing?, baseline, configs:{cfg_name: pred_glob}} -> per-subject CSVs + stats."""
    os.makedirs(out_dir, exist_ok=True)
    labels_dir, manifest = spec["labels_dir"], spec["manifest"]
    sp = spec.get("spacing_mm", spacing_mm)
    by_subject = {}
    for name, pred_glob in spec["configs"].items():
        print(f"[score] {name}")
        df = score_config_by_subject(pred_glob, labels_dir, manifest, sp)
        path = os.path.join(out_dir, f"metrics_by_subject__{name}.csv")
        df.to_csv(path, index=False)
        by_subject[name] = path

    baseline = spec["baseline"]
    if baseline not in by_subject:
        sys.exit(f"baseline {baseline!r} not among configs")
    rows = []
    for name, path in by_subject.items():
        if name == baseline:
            continue
        cmp = compare(path, by_subject[baseline], out=None, name_a=name, name_b=baseline)
        cmp.insert(0, "config", name)
        rows.append(cmp)
    pair = pd.concat(rows, ignore_index=True)
    for col, fn, new in [("p_ttest", _holm, "p_ttest_holm"), ("p_ttest", _bh, "p_ttest_bh"),
                         ("p_wilcoxon", _holm, "p_wilcoxon_holm"), ("p_wilcoxon", _bh, "p_wilcoxon_bh")]:
        if col in pair:
            pair[new] = pair.groupby("metric")[col].transform(lambda s: fn(s.to_numpy()))
    out_csv = os.path.join(out_dir, f"stats__{spec['name']}.csv")
    pair.to_csv(out_csv, index=False)
    print(f"[table] {spec['name']}: paired vs {baseline} (N subjects, Holm+BH corrected) -> {out_csv}")
    return out_csv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("score")
    s.add_argument("--pred-glob", required=True, help="glob of held-out prediction niftis (quote it)")
    s.add_argument("--labels-dir", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--spacing-mm", type=float, default=0.075, help="in-plane voxel size (mm) for HD95")
    s.add_argument("--out", required=True)

    t = sub.add_parser("table")
    t.add_argument("--spec", required=True, help="JSON spec (see module docstring)")
    t.add_argument("--spacing-mm", type=float, default=0.075)
    t.add_argument("--out", required=True)

    c = sub.add_parser("compare")
    c.add_argument("--a", required=True); c.add_argument("--b", required=True)
    c.add_argument("--names", nargs=2, default=["A", "B"]); c.add_argument("--out")

    # per-table: pair every variant against one baseline, Holm+BH corrected across variants
    pt = sub.add_parser("paired")
    pt.add_argument("--baseline", required=True, help="name=path/to/metrics_by_subject__<name>.csv")
    pt.add_argument("--variants", nargs="+", required=True, help="name=path ... (paired vs baseline)")
    pt.add_argument("--out", required=True)

    a = ap.parse_args()
    if a.cmd == "score":
        df = score_config_by_subject(a.pred_glob, a.labels_dir, a.manifest, (a.spacing_mm, a.spacing_mm))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"-> {a.out}\n{df.to_string(index=False)}")
    elif a.cmd == "table":
        spec = json.load(open(a.spec))
        run_table(spec, a.out, (a.spacing_mm, a.spacing_mm))
    elif a.cmd == "compare":
        df = compare(a.a, a.b, out=a.out, name_a=a.names[0], name_b=a.names[1])
        print(df.to_string(index=False))
    elif a.cmd == "paired":
        bname, bpath = a.baseline.split("=", 1)
        rows = []
        for v in a.variants:
            vname, vpath = v.split("=", 1)
            cmp = compare(vpath, bpath, out=None, name_a=vname, name_b=bname)
            cmp.insert(0, "config", vname)
            rows.append(cmp)
        pair = pd.concat(rows, ignore_index=True)
        for col, fn, new in [("p_ttest", _holm, "p_ttest_holm"), ("p_ttest", _bh, "p_ttest_bh"),
                             ("p_wilcoxon", _holm, "p_wilcoxon_holm"), ("p_wilcoxon", _bh, "p_wilcoxon_bh")]:
            if col in pair:
                pair[new] = pair.groupby("metric")[col].transform(lambda s: fn(s.to_numpy()))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        pair.to_csv(a.out, index=False)
        print(f"paired vs {bname} (Holm+BH across {len(a.variants)} variants) -> {a.out}")


if __name__ == "__main__":
    main()
