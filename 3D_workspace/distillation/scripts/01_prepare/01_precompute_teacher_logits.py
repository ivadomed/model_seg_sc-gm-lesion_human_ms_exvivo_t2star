#!/usr/bin/env python
"""Cache the frozen teacher's soft targets, once, before any student training.

For every preprocessed training case, runs the 4-fold teacher ensemble and saves the
**ensemble-averaged logits** (not probabilities) in the exact preprocessed voxel grid, so they
align voxel-for-voxel with the patches the student dataloader will crop. nnU-Net ensembles in
logit space (per-fold logits averaged, softmax applied once at the end), so the cached logits are
precisely the deployed teacher. Temperature is then free at train time via softmax(logits / T).

Output: one float16 `<case>.b2nd` per case, shape (num_classes, *spatial), under
`<preprocessed>/<dataset>/<out-subdir>/`.

GPU sliding-window inference -- run under set_slot.
"""
import argparse
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, check=True).stdout.strip()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(REPO_DIR))

os.environ.setdefault("nnUNet_raw", f"{PROJECT_ROOT}/nnUNet_data/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", f"{PROJECT_ROOT}/nnUNet_data/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", f"{PROJECT_ROOT}/nnUNet_data/nnUNet_results")

import numpy as np
import torch
import blosc2
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, join

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

DEFAULT_TEACHER = f"{PROJECT_ROOT}/nnUNet_data/nnUNet_results/paper_results/3D/ablations/base3d/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"
DEFAULT_PREPRO = f"{os.environ['nnUNet_preprocessed']}/Dataset011_3D_MagPhase/nnUNetPlans_3d_fullres"
DEFAULT_OUT_SUBDIR = "teacher_logits__patchsize_5_adamw"


def save_logits(arr_f16: np.ndarray, path: str):
    if os.path.exists(path):
        os.remove(path)
    blosc2.set_nthreads(1)
    blosc2.asarray(np.ascontiguousarray(arr_f16), urlpath=path, cparams={"codec": blosc2.Codec.ZSTD, "clevel": 8})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teacher", default=DEFAULT_TEACHER, help="teacher model folder (has fold_*, plans.json, dataset.json)")
    ap.add_argument("--preprocessed", default=DEFAULT_PREPRO, help="preprocessed config folder with <case>.b2nd")
    ap.add_argument("--out-subdir", default=DEFAULT_OUT_SUBDIR, help="output dir name, created beside the preprocessed dataset")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--checkpoint", default="checkpoint_best.pth")
    ap.add_argument("--cases", nargs="+", default=None, help="subset of case ids (default: all in preprocessed folder)")
    ap.add_argument("--tile-step-size", type=float, default=0.5)
    ap.add_argument("--no-mirroring", action="store_true", help="disable test-time mirroring (default: on, matches teacher inference)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = join(os.path.dirname(args.preprocessed.rstrip("/")), args.out_subdir)
    maybe_mkdir_p(out_dir)

    dataset = nnUNetDatasetBlosc2(args.preprocessed)
    case_ids = args.cases if args.cases is not None else sorted(dataset.identifiers)
    print(f"[precompute] {len(case_ids)} cases | teacher={args.teacher}")
    print(f"[precompute] folds={args.folds} ckpt={args.checkpoint} mirroring={not args.no_mirroring}")
    print(f"[precompute] out_dir={out_dir}")

    predictor = nnUNetPredictor(
        tile_step_size=args.tile_step_size, use_gaussian=True, use_mirroring=not args.no_mirroring,
        perform_everything_on_device=(args.device == "cuda"), device=torch.device(args.device),
        verbose=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(args.teacher, use_folds=tuple(args.folds), checkpoint_name=args.checkpoint)
    num_classes = predictor.label_manager.num_segmentation_heads
    print(f"[precompute] num_classes(out)={num_classes} patch={predictor.configuration_manager.patch_size}")

    t0 = time.time()
    for i, case_id in enumerate(case_ids):
        out_path = join(out_dir, case_id + ".b2nd")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  [{i+1}/{len(case_ids)}] {case_id}: exists, skip")
            continue
        data, _seg, _seg_prev, _properties = dataset.load_case(case_id)
        data = np.asarray(data[:], dtype=np.float32)  # (C_in, X, Y, Z), materialize blosc2 -> np
        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data))  # (C_out, X, Y, Z) cpu
        logits = logits.cpu().numpy().astype(np.float16)
        assert logits.shape[0] == num_classes, f"{case_id}: got {logits.shape[0]} heads, expected {num_classes}"
        assert logits.shape[1:] == data.shape[1:], f"{case_id}: logits {logits.shape} vs data {data.shape} spatial mismatch"
        save_logits(logits, out_path)
        dt = time.time() - t0
        print(f"  [{i+1}/{len(case_ids)}] {case_id}: {logits.shape} f16 saved "
              f"({dt:.0f}s elapsed, {dt/(i+1):.1f}s/case)", flush=True)

    print(f"[precompute] done in {time.time()-t0:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
