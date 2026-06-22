#!/usr/bin/env python3
"""
Build a 3D nnUNet_raw dataset from the clean ms-exvivo-nih source WITHOUT duplicating
image bytes: imagesTr/labelsTr are SYMLINKS into ms-exvivo-nih.

The 12 training subjects' ``derivatives/labels_3d`` files are already combined multiclass
volumes {0:bg,1:WM,2:GM,3:lesionWM,4:lesionGM} in correct orientation (the weakly-supervised
pseudo-GT), so they are used directly as the nnUNet label -- no derivation, no copy. The 3
test subjects (separate SC/GM/lesion GT) are excluded from training.

Case IDs are assigned in sorted (subject, acquisition) order and the 4-fold subject split is
regenerated with KFold(seed=12345) -- identical partition to the historical pipeline, so the
existing splits stay reproducible.

Usage:
  set_slot 3 .venv/bin/python build_dataset_3d.py \
      --clean-root ms-exvivo-nih --out-raw nnUNet_data/nnUNet_raw \
      --dataset-id 11 --name 3D_MagPhase --channels mag_phase
"""
from __future__ import annotations
import argparse, json, os, glob, re
from sklearn.model_selection import KFold

DERIV = "derivatives/labels_3d"
LABELS_4CLASS = {"background": 0, "WM": 1, "GM": 2, "lesion_WM": 3, "lesion_GM": 4}
DEFAULT_TEST = ["sub-TNU018", "sub-TNU025", "sub-TNU026"]


def rel_symlink(src: str, dst: str):
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.relpath(os.path.realpath(src), os.path.dirname(dst)), dst)


def combined_label_files(clean_root: str, sub: str):
    """Combined multiclass pseudo-GT files for a training subject (no _label- suffix)."""
    files = glob.glob(os.path.join(clean_root, DERIV, sub, "**", "*_T2star.nii.gz"), recursive=True)
    return sorted(f for f in files if not re.search(r"_label-[A-Za-z]+_seg", os.path.basename(f)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-root", default="ms-exvivo-nih")
    ap.add_argument("--out-raw", default="nnUNet_data/nnUNet_raw")
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--name", default="3D_MagPhase")
    ap.add_argument("--channels", choices=["mag_phase", "mag"], default="mag_phase")
    ap.add_argument("--test-subjects", nargs="*", default=DEFAULT_TEST)
    args = ap.parse_args()

    clean = os.path.abspath(args.clean_root)
    task = f"Dataset{args.dataset_id:03d}_{args.name}"
    base = os.path.abspath(os.path.join(args.out_raw, task))
    imagesTr = os.path.join(base, "imagesTr")
    labelsTr = os.path.join(base, "labelsTr")
    os.makedirs(imagesTr, exist_ok=True)
    os.makedirs(labelsTr, exist_ok=True)

    test = set(args.test_subjects)
    subjects = []
    for d in sorted(glob.glob(os.path.join(clean, DERIV, "sub-*"))):
        sub = os.path.basename(d)
        if sub in test:
            continue
        if combined_label_files(clean, sub):   # training subjects only (combined pseudo-GT)
            subjects.append(sub)

    manifest, subject_to_cases = [], {s: [] for s in subjects}
    case = 0
    dual = args.channels == "mag_phase"
    for sub in subjects:
        for label_path in combined_label_files(clean, sub):
            fname = os.path.basename(label_path)                      # == raw mag basename
            mag = os.path.join(clean, sub, "anat", fname)
            phase = os.path.join(clean, sub, "anat", fname.replace("_part-mag", "_part-phase"))
            if not os.path.exists(mag):
                print(f"  ! skip {sub}/{fname}: missing mag {mag}"); continue
            if dual and not os.path.exists(phase):
                print(f"  ! skip {sub}/{fname}: missing phase (dual-channel)"); continue
            cid = f"MagPhase_{case:04d}"
            rel_symlink(mag, os.path.join(imagesTr, f"{cid}_0000.nii.gz"))
            if dual:
                rel_symlink(phase, os.path.join(imagesTr, f"{cid}_0001.nii.gz"))
            rel_symlink(label_path, os.path.join(labelsTr, f"{cid}.nii.gz"))
            subject_to_cases[sub].append(cid)
            manifest.append({"nnunet_id": cid, "subject": sub, "acq": fname,
                             "src_mag": os.path.relpath(mag, clean),
                             "src_phase": os.path.relpath(phase, clean) if dual else None,
                             "src_label": os.path.relpath(label_path, clean)})
            case += 1

    channel_names = {"0": "Magnitude", "1": "Phase"} if dual else {"0": "Magnitude"}
    with open(os.path.join(base, "dataset.json"), "w") as f:
        json.dump({"channel_names": channel_names, "labels": LABELS_4CLASS,
                   "numTraining": case, "file_ending": ".nii.gz", "name": task}, f, indent=4, sort_keys=True)
    with open(os.path.join(base, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # subject-based 4-fold split (same seed as historical preprocess_bids.py)
    kf = KFold(n_splits=4, shuffle=True, random_state=12345)
    splits = []
    for tr_idx, va_idx in kf.split(subjects):
        tr = [c for i in tr_idx for c in subject_to_cases[subjects[i]]]
        va = [c for i in va_idx for c in subject_to_cases[subjects[i]]]
        splits.append({"train": tr, "val": va})
    with open(os.path.join(base, "splits_final.json"), "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Built {task}: {case} cases / {len(subjects)} subjects (channels={args.channels})")
    print(f"  excluded test subjects: {sorted(test)}")
    print(f"  -> {base}")
    print(f"  splits: " + ", ".join(f"fold{i} val={len(s['val'])}" for i, s in enumerate(splits)))


if __name__ == "__main__":
    main()
