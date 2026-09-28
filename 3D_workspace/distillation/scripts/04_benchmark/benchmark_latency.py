#!/usr/bin/env python
"""Wall-clock inference latency: student vs teacher, single fold, no-TTA (optionally +TTA for
the teacher too). Times the sliding-window forward pass only (weights preloaded), first case
discarded as warmup. Run under set_slot (reserves the device).

Usage:
  python benchmark_latency.py --device cuda
  python benchmark_latency.py --device cpu
  python benchmark_latency.py --device cuda --with-tta
"""
import argparse
import os
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = subprocess.run(["git", "-C", HERE, "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True, check=True).stdout.strip()
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(REPO_DIR))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
ap.add_argument("--with-tta", action="store_true", help="also time the teacher with 8x mirroring TTA")
args = ap.parse_args()
if args.device == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hard guard: make sure nothing can touch the GPU

os.environ.setdefault("nnUNet_raw", f"{PROJECT_ROOT}/nnUNet_data/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", f"{PROJECT_ROOT}/nnUNet_data/nnUNet_preprocessed")
os.environ.setdefault("nnUNet_results", f"{PROJECT_ROOT}/nnUNet_data/nnUNet_results")

import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2

PP = f"{PROJECT_ROOT}/nnUNet_data/nnUNet_preprocessed/Dataset011_3D_MagPhase/nnUNetPlans_3d_fullres"
TEACHER = f"{PROJECT_ROOT}/nnUNet_data/nnUNet_results/paper_results/3D/ablations/base3d/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres"
STUDENT = f"{PROJECT_ROOT}/nnUNet_data/nnUNet_results/distillation/kd_T1/nnUNetPlansDistillS/Dataset011_3D_MagPhase/nnUnet3DDistillTrainer__nnUNetPlansDistillS__3d_fullres"
CASES = ["MagPhase_0038", "MagPhase_0039", "MagPhase_0040", "MagPhase_0041"]  # fold_1 val; 1st = warmup

dataset = nnUNetDatasetBlosc2(PP, identifiers=CASES)
volumes = {case: np.asarray(dataset.load_case(case)[0][:], np.float32) for case in CASES}


def bench(model_dir, checkpoint, mirror, tag):
    device = torch.device(args.device)
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=mirror,
                                 perform_everything_on_device=(device.type == "cuda"),
                                 device=device, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(model_dir, use_folds=(1,), checkpoint_name=checkpoint)
    net = predictor.network
    net.load_state_dict(predictor.list_of_parameters[0])
    net.to(device).eval()

    times = []
    with torch.no_grad():
        for i, case in enumerate(CASES):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            predictor.predict_sliding_window_return_logits(torch.from_numpy(volumes[case]))
            if device.type == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
            if i > 0:  # discard the first case as warmup
                times.append(dt)

    mean = float(np.mean(times))
    print(f"  {tag:32} {mean:6.2f} s/vol  (n={len(times)})", flush=True)
    return mean


device_label = torch.cuda.get_device_name(0) if args.device == "cuda" else f"CPU, {torch.get_num_threads()} threads"
print(f"Benchmarking (per-volume sliding-window, {args.device}): {device_label}")

student = bench(STUDENT, "checkpoint_final.pth", False, "student 1-model, no-TTA")
teacher = bench(TEACHER, "checkpoint_best.pth", False, "teacher 1-fold, no-TTA")

print("\nDerived deployment latencies (per volume):")
print(f"  student (deployed: 1 model, no-TTA)     {student:7.2f} s   [1.0x]")
print(f"  teacher 1-fold, no-TTA                  {teacher:7.2f} s   [{teacher/student:4.1f}x student]")

if args.with_tta:
    teacher_tta = bench(TEACHER, "checkpoint_best.pth", True, "teacher 1-fold, +TTA(8x)")
    print(f"  teacher 1-fold, +TTA                    {teacher_tta:7.2f} s   [{teacher_tta/student:4.1f}x student]")
