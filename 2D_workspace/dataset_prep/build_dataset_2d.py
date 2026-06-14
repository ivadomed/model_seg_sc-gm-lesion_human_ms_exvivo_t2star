#!/usr/bin/env python3
"""
Build a 2D nnUNet_raw dataset from the clean ms-exvivo-nih source.

Pipeline (reuses the existing, clean-schema-compatible scripts unchanged):
  1) extract_slices_multichannel.py  --axis 2   (clean cord axis; axis-2 slices reproduce
     the legacy training-slice geometry: clean[:,:,k] == legacy[:,k,:])  -> annotated slices only
  2) (exclude held-out TEST subjects so there is no leakage)
  3) convert_bids_to_nnunet_multichannel.py  (derives WM/GM/lesion classes from SC/GM/lesion)
  4) generate subject-based 4-fold splits_final.json (KFold seed 12345)

Only annotated slices are materialized (small). Writes to nnUNet_data/nnUNet_raw/<Dataset>.

Usage:
  set_slot 3 .venv/bin/python build_dataset_2d.py \
      --clean-root ms-exvivo-nih --out-raw nnUNet_data/nnUNet_raw \
      --work nnUNet_data/work/sliced_2d \
      --dataset-id 21 --name 2D_MagPhase --label-mode all --channels mag_phase
"""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, sys, glob
from collections import defaultdict
from sklearn.model_selection import KFold

HERE = os.path.dirname(os.path.abspath(__file__))
PREP = HERE  # extract_slices / convert_bids deps live alongside this builder
DEFAULT_TEST = ["sub-TNU018", "sub-TNU025", "sub-TNU026"]


def run(cmd):
    print("  $", " ".join(cmd)); sys.stdout.flush()
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", default="ms-exvivo-nih")
    ap.add_argument("--out-raw", default="nnUNet_data/nnUNet_raw")
    ap.add_argument("--work", default="nnUNet_data/work/sliced_2d")
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--name", default="2D_MagPhase")
    ap.add_argument("--label-mode", default="all",
                    choices=["all", "tissues", "lesions", "merged_lesion", "sc_and_lesion"])
    ap.add_argument("--channels", choices=["mag_phase", "mag-one-channel"], default="mag_phase")
    ap.add_argument("--axis", type=int, default=2, help="cord axis for clean data (legacy was 1)")
    ap.add_argument("--test-subjects", nargs="*", default=DEFAULT_TEST)
    ap.add_argument("--keep-work", action="store_true")
    args = ap.parse_args()

    clean = os.path.abspath(args.clean_root)
    work = os.path.abspath(args.work)
    py = sys.executable
    if os.path.isdir(work):
        shutil.rmtree(work)

    # 1) slice annotated slices along the cord axis (axis 2 for clean)
    print("[1/4] extracting annotated 2D slices (axis %d)" % args.axis)
    run([py, os.path.join(PREP, "extract_slices_multichannel.py"),
         "--path-data", clean,
         "--label-folder", os.path.join(clean, "derivatives", "labels_2d"),
         "--labels", "SC", "GM", "lesion",
         "--axis", str(args.axis),
         "--path-out", work])

    # 2) drop held-out test subjects (no leakage)
    print("[2/4] excluding test subjects:", args.test_subjects)
    for sub in args.test_subjects:
        for p in (os.path.join(work, sub),
                  os.path.join(work, "derivatives", "labels", sub)):
            if os.path.isdir(p):
                shutil.rmtree(p); print("   removed", os.path.relpath(p, work))

    # 3) convert sliced BIDS -> nnUNet raw (label derivation happens here)
    print("[3/4] converting to nnUNet raw")
    run([py, os.path.join(PREP, "convert_bids_to_nnunet_multichannel.py"),
         "--path-data", work,
         "--path-out", os.path.abspath(args.out_raw),
         "--taskname", args.name,
         "--tasknumber", str(args.dataset_id),
         "--label-suffixes", "SC", "GM", "lesion",
         "--label-mode", args.label_mode,
         "--channel-config", args.channels])

    # 4) subject-based 4-fold split from the converter's inference_manifest.json
    print("[4/4] generating subject-based 4-fold split")
    ds_dir = os.path.join(os.path.abspath(args.out_raw), f"Dataset{args.dataset_id:03d}_{args.name}")
    manifest = json.load(open(os.path.join(ds_dir, "inference_manifest.json")))
    subj_to_cases = defaultdict(list)
    for sample_id, ref in manifest.items():     # ref = "sub-XXX/<key>"
        subj_to_cases[ref.split("/")[0]].append(sample_id)
    subjects = sorted(subj_to_cases)
    kf = KFold(n_splits=4, shuffle=True, random_state=12345)
    splits = []
    for tr, va in kf.split(subjects):
        splits.append({"train": [c for i in tr for c in subj_to_cases[subjects[i]]],
                       "val":   [c for i in va for c in subj_to_cases[subjects[i]]]})
    json.dump(splits, open(os.path.join(ds_dir, "splits_final.json"), "w"), indent=4)

    n_cases = sum(len(v) for v in subj_to_cases.values())
    print(f"\nBuilt Dataset{args.dataset_id:03d}_{args.name}: {n_cases} slice-cases / "
          f"{len(subjects)} subjects (mode={args.label_mode}, channels={args.channels})")
    print("  subjects:", ", ".join(subjects))
    print("  splits:", ", ".join(f"fold{i} val={len(s['val'])}" for i, s in enumerate(splits)))
    print("  ->", ds_dir)
    if not args.keep_work:
        shutil.rmtree(work); print("  (removed work dir)")


if __name__ == "__main__":
    main()
