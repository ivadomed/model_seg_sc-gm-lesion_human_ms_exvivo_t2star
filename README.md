# model_seg_sc-gm-lesion_human_ms_exvivo_t2star

Segmentation models for spinal cord white matter (WM), gray matter (GM), and MS lesions from high-resolution ex vivo GRE magnitude + phase MRI.

## Status

- **Final publication model**: 3D nnU-Net model (recommended for reported results and external use).
- **Companion model**: 2D nnU-Net model (provided for comparison/ablation and legacy workflows).

## Citation (Placeholder)

If you use this repository, please cite:

```bibtex
@article{PLACEHOLDER_202X,
	title   = {PLACEHOLDER_TITLE},
	author  = {PLACEHOLDER_AUTHORS},
	journal = {PLACEHOLDER_JOURNAL},
	year    = {202X},
	doi     = {PLACEHOLDER_DOI}
}
```

## Data Access (Placeholder)

- **Dataset statement**: `PLACEHOLDER_DATASET_STATEMENT`
- **Access policy**: `PLACEHOLDER_ACCESS_POLICY`
- **Download link/request contact**: `PLACEHOLDER_LINK_OR_EMAIL`

## Environment and Reproducibility

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Expected input format for inference

Both publication inference scripts expect nnU-Net-style input folders:

- One folder containing NIfTI files in `imagesTs` style.
- Channel suffixes for multichannel inputs, e.g.:
	- `CASE001_0000.nii.gz` (magnitude)
	- `CASE001_0001.nii.gz` (phase)

## Publication Inference (new standalone scripts)

The publication scripts are in `inference_publication/` and are intentionally separate from tailored internal scripts.

### 3D inference (final model)

Edit placeholders in:

- `inference_publication/run_infer_3d_public.sh`

Then run:

```bash
bash inference_publication/run_infer_3d_public.sh
```

Direct Python usage:

```bash
python inference_publication/infer_3d_public.py \
	--input-dir /path/to/nnunet_inputs_3d/imagesTs \
	--output-dir /path/to/output/predictions_3d \
	--model-folder /path/to/nnUNet_results/DatasetXXXX_<TASK_NAME>/nnUnet3DCustomTrainer__nnUNetPlans__3d_fullres \
	--folds 0 1 2 3 \
	--checkpoint checkpoint_final.pth \
	--gpu-id 0 \
	--overwrite
```

### 2D inference (companion model)

Edit placeholders in:

- `inference_publication/run_infer_2d_public.sh`

Then run:

```bash
bash inference_publication/run_infer_2d_public.sh
```

Direct Python usage:

```bash
python inference_publication/infer_2d_public.py \
	--input-dir /path/to/nnunet_inputs_2d/imagesTs \
	--output-dir /path/to/output/predictions_2d \
	--model-folder /path/to/nnUNet_results/DatasetXXXX_<TASK_NAME>/nnUNetTrainerWandb__nnUNetPlans__2d \
	--folds 0 1 2 3 \
	--checkpoint checkpoint_final.pth \
	--gpu-id 0 \
	--overwrite
```

## Pretrained Weights (Placeholder)

- **Release page**: `PLACEHOLDER_RELEASE_URL`
- **3D final model folder name**: `PLACEHOLDER_3D_MODEL_FOLDER`
- **2D model folder name**: `PLACEHOLDER_2D_MODEL_FOLDER`

## Repository Layout (relevant to publication)

- `inference_publication/`: clean, standalone publication inference entrypoints.
- `3D_workspace/`: full 3D training/inference research workspace.
- `2D_workspace/`: full 2D training/inference research workspace.
- `helpers/`: shared utility modules.

## Notes

- Existing tailored inference scripts in `2D_workspace/` and `3D_workspace/` are preserved as-is.
- Publication users should prefer the scripts in `inference_publication/`.
