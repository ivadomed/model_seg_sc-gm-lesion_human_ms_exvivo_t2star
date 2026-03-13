import os
import shutil
import nibabel as nib
import numpy as np

# ================= CONFIGURATION =================
INPUT_DIR = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/test_datasets/bids_test_dataset"
OUTPUT_DIR = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/test_datasets/bids_test_dataset_REORIENTED"

# Extensions to look for
EXTENSIONS = ('.nii.gz', '.nii')
# =================================================

def process_and_save(input_path, output_path):
    try:
        # 1. Load the image exactly as in your script
        img = nib.load(input_path)
        data = img.get_fdata()
        affine = img.affine

        # 2. Check dimensions
        orig_shape = data.shape
        if len(orig_shape) != 3:
            print(f"⚠️ Skipping reorientation for {os.path.basename(input_path)}: Not a 3D volume (shape: {orig_shape})")
            # Copy it as-is so the dataset remains intact
            shutil.copy2(input_path, output_path)
            return

        # 3. Swap Axis 1 (Y/Coronal) and Axis 2 (Z/Axial)
        new_data = np.swapaxes(data, 1, 2)
        
        # 4. Create new image using the NEW data but the OLD affine and header
        new_img = nib.Nifti1Image(new_data, affine, img.header)
        
        # 5. Update the header shape
        new_img.header.set_data_shape(new_data.shape)
        
        # 6. Save to the new destination
        nib.save(new_img, output_path)
        print(f"✅ Fixed & Saved: {os.path.basename(input_path)} | {orig_shape} -> {new_data.shape}")

    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")

def main():
    print(f"🚀 Starting out-of-place reorientation...")
    print(f"📂 Input: {INPUT_DIR}")
    print(f"📂 Output: {OUTPUT_DIR}\n")

    count = 0
    
    # Walk through the directory tree
    for root, dirs, files in os.walk(INPUT_DIR):
        
        # Calculate relative path to maintain folder structure
        rel_path = os.path.relpath(root, INPUT_DIR)
        target_dir = os.path.join(OUTPUT_DIR, rel_path)
        
        # Create the corresponding directory in the output path
        os.makedirs(target_dir, exist_ok=True)
        
        for file in files:
            input_path = os.path.join(root, file)
            output_path = os.path.join(target_dir, file)
            
            if file.endswith(EXTENSIONS):
                process_and_save(input_path, output_path)
                count += 1
            else:
                # Copy non-NIfTI files (JSON, TSV, etc.) directly to the new dataset
                shutil.copy2(input_path, output_path)

    print(f"\n✨ Processed {count} volumes and mirrored the dataset structure.")

if __name__ == "__main__":
    main()