#!/usr/bin/env python

#Inspired by FastSurfer's method at https://github.com/Deep-MI/FastSurfer
import time
import glob
import h5py
import numpy as np
import SimpleITK as sitk
from wmh.utilities import preprocessing

class Dataset:
    '''
    Class to load images and pre-process into hdf5 file
    '''

    def __init__(self, params):
        do = "something"


if __name__ == "__main__":

    import argparse

    # Training settings
    parser = argparse.ArgumentParser(description='HDF5-Creation')

    parser.add_argument('--hdf5_name', type=str, default="testsuite_2.hdf5",
                        help='path and name of hdf5-dataset (default: testsuite_2.hdf5)')
    parser.add_argument('--rows_standard', type=int, default=200, help='Height of input to network (Default 200)')
    parser.add_argument('--cols_standard', type=int, default=200, help='Width of input to network (Default 200)')
    parser.add_argument('--data_dir', type=str, default="/testsuite", help="Directory with images to load")
    parser.add_argument('--csv_file', type=str, default=None, help="Csv-file listing subjects to include in file")
    parser.add_argument('--pattern', type=str, help="Pattern to match files in directory.")
    parser.add_argument('--T1_name', type=str, default="T1/T1.nii.gz",
                        help="Default name of T1 images. (default T1/T1.nii.gz)")
    parser.add_argument('--FLAIR_name', type=str, default='T2_FLAIR/T2_FLAIR.nii.gz', help='Default name of T2FLAIR images. (default T2_FLAIR/T2_FLAIR)')
    parser.add_argument('--gt_name', type=str, default='T2_FLAIR/lesions/final_mask.nii.gz',help='Default name for ground truth segmentations (default T2_FLAIR/lesions/final_mask.nii.gz)')


    args = parser.parse_args()

    network_params = {"dataset_name": args.hdf5_name, "rows_standard": args.rows_standard, "cols_standard": args.cols_standard,
                      "data_path": args.data_dir, "csv_file": args.csv_file, "pattern": args.pattern, "T1_name": args.image_name,
                      "gt_name": args.gt_name, "FLAIR_name": args.FLAIR_name}

    stratified = PopulationDataset(network_params)
    stratified.create_hdf5_dataset(plane=args.plane)
