#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication-facing 3D nnU-Net inference entrypoint."
    )
    parser.add_argument("--input-dir", required=True, help="Folder with nnU-Net-format input images.")
    parser.add_argument("--output-dir", required=True, help="Folder where predictions are written.")
    parser.add_argument("--model-folder", required=True, help="Path to trained 3D model folder.")
    parser.add_argument(
        "--folds",
        nargs="+",
        default=["0", "1", "2", "3"],
        help="Folds to use. Example: --folds 0 1 2 3 or --folds all",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoint_best.pth",
        help="Checkpoint filename in each fold folder.",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU index (ignored on CPU).")
    parser.add_argument("--tile-step-size", type=float, default=0.5, help="nnU-Net tile step size.")
    parser.add_argument("--num-preproc-workers", type=int, default=1)
    parser.add_argument("--num-export-workers", type=int, default=1)
    parser.add_argument("--tta", action="store_true", help="Enable mirroring-based test-time augmentation.")
    parser.add_argument("--save-probabilities", action="store_true", help="Save softmax probabilities.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_folder = Path(args.model_folder)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder does not exist: {model_folder}")

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda", args.gpu_id) if torch.cuda.is_available() else torch.device("cpu")

    predictor = nnUNetPredictor(
        tile_step_size=args.tile_step_size,
        use_gaussian=True,
        use_mirroring=args.tta,
        perform_everything_on_device=torch.cuda.is_available(),
        device=device,
        verbose=True,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )

    parsed_folds = tuple(args.folds) if len(args.folds) == 1 and args.folds[0] == "all" else tuple(int(f) for f in args.folds)

    predictor.initialize_from_trained_model_folder(
        str(model_folder),
        use_folds=parsed_folds,
        checkpoint_name=args.checkpoint,
    )

    predictor.predict_from_files(
        str(input_dir),
        str(output_dir),
        save_probabilities=args.save_probabilities,
        overwrite=args.overwrite,
        num_processes_preprocessing=args.num_preproc_workers,
        num_processes_segmentation_export=args.num_export_workers,
        folder_with_segs_from_prev_stage=None,
        num_parts=1,
        part_id=0,
    )


if __name__ == "__main__":
    main()
