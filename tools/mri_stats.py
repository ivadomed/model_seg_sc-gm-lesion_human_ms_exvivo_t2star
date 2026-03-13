import os
import glob
import nibabel as nib
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_axial_slices(nifti_path):
    """
    Loads a NIfTI file and returns the number of axial slices.
    """
    try:
        img = nib.load(nifti_path)
        shape = img.shape
        affine = img.affine
        orientation = nib.aff2axcodes(affine)
        
        # Determine which dimension corresponds to the "axial" axis as requested by user
        # User specified: "The axial axis (along which we count the slices) should be the longest dimension in a volume"
        # Additionally, filenames contain 'cor' suggestive of Coronal acquisition, where slice dim might be Anterior-Posterior (P)
        # Checking if P is the longest dimension confirms this hypothesis for the seen files.
        
        # We will count the longest dimension as the number of "slices".
        num_slices = max(shape)
        
        return num_slices
    except Exception as e:
        logging.error(f"Error processing {nifti_path}: {e}")
        return None

def get_labelled_slices(nifti_path):
    """
    Loads a segmentation NIfTI file and counts the number of slices 
    that have at least one non-zero pixel (labelled) along the axial axis.
    Utilizes numpy efficiency and avoids unnecessary float conversion.
    """
    try:
        img = nib.load(nifti_path)
        # Use dataobj to get the array without potentially casting to float64 via get_fdata
        data = np.asanyarray(img.dataobj)
        shape = data.shape
        
        # Determine axial axis (longest dimension, per user request)
        axial_dim = np.argmax(shape)
        
        # Check if data is empty
        if data.size == 0:
            return 0, 0

        # Construct axes tuple for summation (sum over all dims except axial_dim)
        # e.g., if shape=(300, 700, 200) and axial_dim=1, we sum over (0, 2)
        sum_axes = tuple([i for i in range(len(shape)) if i != axial_dim])
        
        # Check for non-zero values in each slice
        # np.any is faster than summing if we just want existence
        has_label = np.any(data, axis=sum_axes)
        
        labelled_slice_count = np.sum(has_label)
        
        return labelled_slice_count, shape[axial_dim]
    except Exception as e:
        logging.error(f"Error processing label file {nifti_path}: {e}")
        return 0, 0

def analyze_derivatives(dataset_name, derivatives_root):
    """
    Analyzes a derivatives folder to count labelled slices.
    """
    path = Path(derivatives_root)
    if not path.exists():
        logging.warning(f"Derivatives path not found for {dataset_name}: {derivatives_root}")
        return

    logging.info(f"--- Analyzing Labels for: {dataset_name} ---")
    
    # Identify search pattern based on dataset type.
    # 2D datasets use sparse labels split into classes (e.g. *SC_seg.nii.gz).
    # 3D datasets use a single multi-class label file which looks like the magnitude filename (part-mag.nii.gz)
    # inside the derivatives folder.
    
    if "3D" in dataset_name:
        # For 3D Dense: The label file is named e.g., ...part-mag.nii.gz
        pattern = "*part-mag.nii.gz"
        logging.info(f"Using 3D Dense pattern: {pattern}")
    else:
        # For 2D Sparse: Looking for SC specifically
        pattern = "*SC_seg.nii.gz"
        logging.info(f"Using 2D Sparse pattern: {pattern}")
        
    label_files = sorted(list(path.rglob(pattern)))
    
    if not label_files:
        logging.warning(f"No label files matching '{pattern}' found in {derivatives_root}")
        return

    logging.info(f"Found {len(label_files)} label files. Processing...")

    total_labelled_slices = 0
    total_volume_slices = 0
    file_data = []

    for i, f in enumerate(label_files):
        if i % 20 == 0:
            print(f"Processing {i}/{len(label_files)}...", end='\r', flush=True)

        l_count, v_count = get_labelled_slices(str(f))
        total_labelled_slices += l_count
        total_volume_slices += v_count
        file_data.append({
            'Dataset': dataset_name,
            'Filename': f.name,
            'LabelledSlices': l_count,
            'TotalSlices': v_count,
            'Ratio': l_count / v_count if v_count > 0 else 0
        })
    
    print(f"Processing {len(label_files)}/{len(label_files)} - Done.")
    logging.info(f"Total Labelled Slices: {total_labelled_slices}")
    
    return file_data

def analyze_bids_dataset(bids_root):
    """
    Analyzes the BIDS dataset to count axial slices.
    """
    bids_path = Path(bids_root)
    if not bids_path.exists():
        logging.error(f"BIDS root directory not found: {bids_root}")
        return

    # Find all magnitude nifti files
    logging.info(f"Searching for magnitude files in {bids_root}...")
    # Pattern: sub-*/anat/*part-mag.nii.gz
    files = sorted(list(bids_path.glob("sub-*/anat/*part-mag.nii.gz")))
    
    if not files:
        logging.warning("No magnitude files found matching pattern 'sub-*/anat/*part-mag.nii.gz'")
        return

    logging.info(f"Found {len(files)} files. Processing...")

    data = []

    for file_path in files:
        subject_id = file_path.parent.parent.name # sub-XXX
        filename = file_path.name
        
        # Simple heuristic for chunk ID if needed, e.g., first part of filename "S0"
        chunk_id = filename.split('_')[0] if '_' in filename else filename

        num_slices = get_axial_slices(str(file_path))

        if num_slices is not None:
            data.append({
                'Subject': subject_id,
                'Chunk': chunk_id,
                'Filename': filename,
                'AxialSlices': num_slices
            })

    if not data:
        logging.error("No valid data collected.")
        return

    df = pd.DataFrame(data)

    # --- Statistics Calculation ---
    
    # 1. Total number of axial slices
    total_slices = df['AxialSlices'].sum()
    
    # 2. Number of chunks per subject
    chunks_per_subject = df.groupby('Subject')['Chunk'].count()
    
    # 3. Number of axial slices per chunk (overall)
    slices_per_chunk = df['AxialSlices']

    # 4. Total axial slices per subject
    slices_per_subject_total = df.groupby('Subject')['AxialSlices'].sum()

    # 5. Average axial slices *per chunk* per subject (probably less useful if chunks vary wildly, but requested)
    slices_per_chunk_per_subject = df.groupby('Subject')['AxialSlices'].mean()

    print("\n" + "="*40)
    print("       MRI DATASET STATISTICS       ")
    print("="*40)
    print(f"Total Files Processed: {len(df)}")
    print(f"Total Subjects: {df['Subject'].nunique()}")
    print("-" * 40)
    
    print(f"\nTotal Axial Slices in Dataset: {total_slices}")

    print("\n--- Slices per Chunk (Single MRI file) ---")
    print(f"Mean: {slices_per_chunk.mean():.2f}")
    print(f"Std : {slices_per_chunk.std():.2f}")
    print(f"Min : {slices_per_chunk.min()}")
    print(f"Max : {slices_per_chunk.max()}")

    print("\n--- Chunks per Subject ---")
    print(f"Mean: {chunks_per_subject.mean():.2f}")
    print(f"Std : {chunks_per_subject.std():.2f}")
    print(f"Min : {chunks_per_subject.min()}")
    print(f"Max : {chunks_per_subject.max()}")

    print("\n--- Total Axial Slices per Subject (Sum of all chunks) ---")
    print(f"Mean: {slices_per_subject_total.mean():.2f}")
    print(f"Std : {slices_per_subject_total.std():.2f}")
    print(f"Min : {slices_per_subject_total.min()}")
    print(f"Max : {slices_per_subject_total.max()}")
    
    print("\n" + "="*40)
    # Optional: Save to CSV
    output_csv = "mri_slice_stats.csv"
    df.to_csv(output_csv, index=False)
    print(f"Detailed statistics saved to {output_csv}")
    print("="*40)

if __name__ == "__main__":
    # --- Part 1: MRI Volume Stats ---
    print("\n" + "="*40)
    print("       MRI VOLUME STATISTICS        ")
    print("="*40)
    BIDS_ROOT = "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/bids_dataset"
    analyze_bids_dataset(BIDS_ROOT)

    # --- Part 2: Labelled Data Stats ---
    label_datasets = {
        "2D Sparse Test": "/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/test_datasets/bids_test_dataset/derivatives",
        "2D Sparse Train/Val": "/home/ge.polymtl.ca/pahoa/nih_project/datasets/2D_datasets/train_datasets/bids_dataset/derivatives",
        "3D Dense Train/Val": "/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/train_datasets/bids_dataset/derivatives",
        # "3D Dense Test": "/home/ge.polymtl.ca/pahoa/nih_project/datasets/3D_datasets/test_datasets/bids_test_dataset/derivatives"
    }

    all_label_stats = []

    print("\n" + "="*40)
    print("       LABELLED DATA STATISTICS       ")
    print("="*40)

    for name, path in label_datasets.items():
        stats = analyze_derivatives(name, path)
        if stats:
            all_label_stats.extend(stats)
            df_curr = pd.DataFrame(stats)
            total_lbl = df_curr['LabelledSlices'].sum()
            print(f"\ndataset: {name}")
            print(f"Total Labelled Slices: {total_lbl}")
            print(f"Total Files: {len(df_curr)}")
            print(f"Avg Labelled Slices/File: {df_curr['LabelledSlices'].mean():.2f}")

    if all_label_stats:
        df_all = pd.DataFrame(all_label_stats)
        out_csv = "labelled_data_stats.csv"
        df_all.to_csv(out_csv, index=False)
        print(f"\nDetailed label statistics saved to {out_csv}")
