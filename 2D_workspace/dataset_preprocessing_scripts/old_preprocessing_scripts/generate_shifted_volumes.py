import nibabel as nib
import numpy as np
from pathlib import Path
import argparse

def generate_shifted_channels(volume, shifts=[0, -1, 1]):
    """
    Generate a multi-channel 3D input volume by shifting along the Y axis (slices).
    Ensures all outputs have the same shape using symmetric edge-padding.
    """
    shifted = []
    H, D, W = volume.shape
    for dz in shifts:
        if dz < 0:
            shifted_vol = np.pad(volume[:, :D+dz, :], ((0,0), (abs(dz),0), (0,0)), mode='edge')
        elif dz > 0:
            shifted_vol = np.pad(volume[:, dz:, :], ((0,0), (0,dz), (0,0)), mode='edge')
        else:
            shifted_vol = volume
        shifted.append(shifted_vol)
    return np.stack(shifted, axis=0)  # Shape: (C, H, D, W)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_nifti", required=True)
    parser.add_argument("--out", dest="output_dir", required=True)
    parser.add_argument("--id", dest="subject_id", required=True)
    parser.add_argument("--shifts", nargs="+", type=int, default=[0, -1, 1])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nii = nib.load(args.input_nifti)
    data = nii.get_fdata()  # (H, D, W)
    shifted = generate_shifted_channels(data, args.shifts)

    for i, vol in enumerate(shifted):
        out_path = output_dir / f"{args.subject_id}_{i:04d}.nii.gz"
        nib.save(nib.Nifti1Image(vol.astype(np.float32), nii.affine, nii.header), out_path)
        print(f"✅ Saved: {out_path}")

if __name__ == "__main__":
    main()
