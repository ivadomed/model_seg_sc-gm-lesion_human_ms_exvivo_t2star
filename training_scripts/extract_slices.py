"""
Extract annotated slices from MRI image and segmentation masks

This python script extracts the segmentations masks slices which are annotated as well as the corresponding mri image slice.
It creates a new folder with the BIDS dataset format.

Example of run:

    $ python extract_slices.py --path-data /path/to/data --path-out /path/to/project/folder

Arguments:

    --path-data : Path to BIDS structured dataset.
    --path-out : Path to output directory.
    
Author: Julien Cohen-Adad (based on original script from Pierre-Louis Benveniste: https://github.com/ivadomed/model_seg_subject-sc_wm-gm_t1/blob/main/training_scripts/extract_slices.py)
"""

import os
import numpy as np
from nibabel import load, Nifti1Image, save
from tqdm import tqdm
import argparse
import pathlib
from pathlib import Path
from time import time


# Parse command line arguments
parser = argparse.ArgumentParser(description='Extract annotated slices from dataset.')
parser.add_argument('--path-data', required=True,
                    help='Path to BIDS structured dataset.')
parser.add_argument('--path-out', help='Path to output directory.', required=True)

args = parser.parse_args()

path_in_images = Path(args.path_data)
# TODO: do not assume the path to labels is always 'derivatives/manual_masks'
# This should be passed as an argument or determined dynamically.
# For now, we assume the labels are in a specific folder structure.
path_in_labels = Path(os.path.join(args.path_data, 'derivatives', 'labels'))
path_out_images = Path(os.path.join(args.path_out, 'data_extracted'))
path_out_labels = Path(os.path.join(args.path_out, 'data_extracted','derivatives', 'labels'))

if __name__ == '__main__':
    
    # Create output folders
    pathlib.Path(path_out_images).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_out_labels).mkdir(parents=True, exist_ok=True)

    folders_paths = [ f.path for f in os.scandir(path_in_images) if f.is_dir() and 'derivatives' not in f.path]
    
    # Iterate over folders
    total_count=0
    for folder in folders_paths:
        time0 = time()

        folder_name = folder.split('/')[-1]
        files_paths = [ f.path for f in os.scandir(Path(os.path.join(folder, 'anat'))) if '.nii.gz' in f.path]
        print("Processing folder: "+str(folder_name)+" ...")
        
        # Iterate over .nii.gz files
        for file in files_paths:
            file_name = file.split('/')[-1]
            file_name_split = file_name.split('.')[0]
            # Open the image file
            mri = load(file)
            mri_full = np.asarray(mri.dataobj)
            # Create MRI folder for that specific image
            mri_folder_out = Path(os.path.join(path_out_images, folder_name, 'anat'))
            pathlib.Path(mri_folder_out).mkdir(parents=True, exist_ok=True)
            # First save the entire nifti file
            img = Nifti1Image(mri_full, affine=mri.affine)
            mri_out_name = Path(os.path.join(mri_folder_out, file_name))
            save(img, str(mri_out_name))  
            # Access label file
            # TODO: do not assume there are these two labels (GM and WM), instead check for available labels dynamically, knowing that all labels are in the same folder.
            path_label_folder = Path(os.path.join(path_in_labels, folder_name, 'anat'))
            label_path_SC = [ f.path for f in os.scandir(path_label_folder) if '.nii.gz' in f.path and file_name_split in f.path and 'SC' in f.path]
            label_path_GM = [ f.path for f in os.scandir(path_label_folder) if '.nii.gz' in f.path and file_name_split in f.path and 'GM' in f.path]
            label_path_lesion = [ f.path for f in os.scandir(path_label_folder) if '.nii.gz' in f.path and file_name_split in f.path and 'lesion' in f.path]
            # Check if there is a label
            if len(label_path_SC)!=0:
                label_SC = load(label_path_SC[0])
                label_GM = load(label_path_GM[0])
                label_lesion = load(label_path_lesion[0])
                # Number of axial slices is asumed to be in the 2nd dimension
                nb_slices = np.asarray(label_SC.dataobj).shape[1]

                # Iterate over slices
                # TODO: can be improved with np.where(sum(label_GM_slice) not 0)
                for slice_i in range(nb_slices):
                    label_SC_slice = np.asarray(label_SC.dataobj)[:, slice_i, :]
                    label_GM_slice = np.asarray(label_GM.dataobj)[:, slice_i, :]
                    label_lesion_slice = np.asarray(label_lesion.dataobj)[:, slice_i, :]
                    # Find slices that are annotated (not empty) 
                    if np.sum(label_SC_slice) !=0:
                        total_count += 1
                        # Create the label folder for that specific image
                        label_folder_out = Path(os.path.join(path_out_labels, folder_name, 'anat'))
                        pathlib.Path(label_folder_out).mkdir(parents=True, exist_ok=True)
                        # Save the annotated MR slices
                        mri_extract = np.asarray(mri.dataobj)[:, slice_i, :]
                        img_extract = Nifti1Image(mri_extract, affine=mri.affine)
                        subject, chunk = file_name_split.split('_',2)[0],file_name_split.split('_',2)[1]
                        # TODO: remove hard-coded 'T2starw'
                        out_name = '{}_{}-slice-{}_{}.{}'.format(subject, chunk, slice_i, 'T2starw', 'nii.gz')
                        path_out_name = Path(os.path.join(mri_folder_out, out_name))
                        save(img_extract, str(path_out_name)) 
                        # Save the masks slices
                        SC_label_img = Nifti1Image(label_SC_slice, affine=label_SC.affine)
                        GM_label_img = Nifti1Image(label_GM_slice, affine=label_GM.affine)
                        lesion_label_img = Nifti1Image(label_lesion_slice, affine=label_lesion.affine)
                        # TODO: get file name from label file
                        SC_out_name = '{}_{}-slice-{}_{}.{}'.format(subject, chunk, slice_i, 'T2starw_label-SC_seg','nii.gz')
                        GM_out_name = '{}_{}-slice-{}_{}.{}'.format(subject, chunk, slice_i, 'T2starw_label-GM_seg','nii.gz')
                        lesion_out_name = '{}_{}-slice-{}_{}.{}'.format(subject, chunk, slice_i, 'T2starw_label-lesion_seg','nii.gz')
                        save(SC_label_img, os.path.join(label_folder_out, SC_out_name))
                        save(GM_label_img, os.path.join(label_folder_out, GM_out_name))
                        save(lesion_label_img, os.path.join(label_folder_out, lesion_out_name))
        print("Done ! It took " + str(time()-time0)+"sec")

    print('---  Finished: extracted '+ str(total_count)+' slices  ---')