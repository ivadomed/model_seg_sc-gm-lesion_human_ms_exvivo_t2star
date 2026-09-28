#!/usr/bin/env python3
"""
Materialize an nnU-Net case-level splits_final.json from the CANONICAL subject-level split.

The canonical split (splits/canonical_subject_split.json) is dataset-agnostic: it assigns each
SUBJECT (spinal cord) to a fold. This tool expands it to nnU-Net case IDs for a specific dataset
by reading that dataset's inference_manifest.json (case_id -> "sub-XXX/..."), so 2D (per-slice)
and 3D (per-volume) datasets both get a consistent, subject-respecting split.

This is the split the published models were trained with; using it keeps every re-run leakage-free
and reproducible (verified: reproduces the paper's cross-fold numbers).

Usage:
  # print/write the case-level split for a dataset
  python -m helpers.make_splits --dataset-dir nnUNet_data/nnUNet_raw/Dataset021_2D_MagPhase [--inject]
  # or a bare manifest -> splits file
  python -m helpers.make_splits --manifest <inference_manifest.json> --out splits_final.json
"""
from __future__ import annotations
import os, sys, json, argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = os.path.join(REPO, "splits", "subject_split_2D.json")


def _subject_of(manifest_value):
    # 2D dict manifest: value is "sub-XXX/..."; 3D list manifest handled by caller
    return manifest_value.split("/")[0]


def load_manifest(path):
    """Return {case_id -> subject} for either the 2D dict manifest or the 3D list manifest."""
    m = json.load(open(path))
    if isinstance(m, dict):
        return {k: _subject_of(v) for k, v in m.items()}
    return {r["nnunet_id"]: r["subject"] for r in m}


def make_splits(case_to_subject, canonical=CANON):
    folds = json.load(open(canonical))["folds"]
    subj_fold = {s: int(k) for k, subs in folds.items() for s in subs}
    n = len(folds)
    out = [{"train": [], "val": []} for _ in range(n)]
    missing = set()
    for case, subj in sorted(case_to_subject.items()):
        if subj not in subj_fold:
            missing.add(subj); continue
        vf = subj_fold[subj]
        for k in range(n):
            (out[k]["val"] if k == vf else out[k]["train"]).append(case)
    if missing:
        print(f"  ! subjects not in canonical split (skipped): {sorted(missing)}", file=sys.stderr)
    for k in range(n):
        out[k]["train"].sort(); out[k]["val"].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", help="nnU-Net raw dataset dir (uses its inference_manifest.json/manifest.json)")
    ap.add_argument("--manifest", help="explicit manifest path (alternative to --dataset-dir)")
    ap.add_argument("--canonical", default=CANON)
    ap.add_argument("--out", help="output splits_final.json path")
    ap.add_argument("--inject", action="store_true", help="write splits_final.json into the dataset dir")
    a = ap.parse_args()

    if a.dataset_dir:
        man = os.path.join(a.dataset_dir, "inference_manifest.json")
        if not os.path.exists(man):
            man = os.path.join(a.dataset_dir, "manifest.json")
    else:
        man = a.manifest
    if not man or not os.path.exists(man):
        sys.exit(f"manifest not found (looked for {man})")

    splits = make_splits(load_manifest(man), a.canonical)
    print(f"  val sizes per fold: {[len(f['val']) for f in splits]} (total {sum(len(f['val']) for f in splits)})")
    out = a.out or (os.path.join(a.dataset_dir, "splits_final.json") if a.inject else None)
    if out:
        json.dump(splits, open(out, "w"), indent=1)
        print(f"  wrote {out}")
    else:
        print(json.dumps(splits, indent=1)[:600] + " ...")


if __name__ == "__main__":
    main()
