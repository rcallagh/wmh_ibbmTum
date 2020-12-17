#!/usr/bin/env python

#Inspired by FastSurfer's method at https://github.com/Deep-MI/FastSurfer
import time
import glob
import h5py
import numpy as np
import SimpleITK as sitk
from os.path import join
from wmh.utilities import preprocessing, ProcessingParams

class Dataset:
    '''
    Class to load images and pre-process into hdf5 file
    '''

    def __init__(self, data_args):
        self.proc_params = ProcessingParams()
        self.proc_params.updateFromArgs(data_args)
        self.dataset_name = data_args.hdf5_name
        self.data_path = data_args.data_dir
        self.csv_file = data_args.csv_file
        self.pattern = data_args.pattern
        self.T1_name = data_args.T1_name
        self.FLAIR_name = data_args.FLAIR_name
        self.gt_name = data_args.gt_name

        if self.csv_file is not None:
            with open(self.csv_file, "r") as s_dirs:
                self.subject_dirs = [line.strip() for line in s_dirs.readlines()]

        else:
            self.search_pattern = join(self.data_path, self.pattern)
            self.subject_dirs = glob.glob(self.search_pattern)

        self.data_set_size = len(self.subject_dirs)

    def create_hdf5_dataset(self):
        '''
        Function to read in images, preprocess and store with hdf5 compression
        '''
        start_d = time.time()

        # Arrays to store the data
        gt_dataset = np.ndarray(shape=(0, self.proc_params.rows_standard, self.proc_params.cols_standard, 1), dtype='float32')
        subjects = []
        if self.proc_params.two_modalities:
            image_dataset = np.ndarray(shape=(0, self.proc_params.rows_standard, self.proc_params.cols_standard, 2), dtype='float32')
        else:
            image_dataset = np.ndarray(shape=(0, self.proc_params.rows_standard, self.proc_params.cols_standard, 1), dtype='float32')

        for idx, current_subject in enumerate(self.subject_dirs):

            try:
                print("Volume Nr: {} Processing MRI Data from {}".format(idx, current_subject))
                start = time.time()
                FLAIR_image = sitk.ReadImage(join(current_subject, self.FLAIR_name), imageIO="NiftiImageIO")
                FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
                T1_image = []
                T1_array = []
                if self.proc_params.two_modalities:
                    T1_image = sitk.ReadImage(join(current_subject, self.T1_name), imageIO="NiftiImageIO")
                    T1_array = sitk.GetArrayFromImage(T1_image)
                gt_image = sitk.ReadImage(join(current_subject, self.gt_name), imageIO="NiftiImageIO")
                gt_array = sitk.GetArrayFromImage(gt_image)
                [images_preproc, self.proc_params] = preprocessing(FLAIR_array, T1_array, self.proc_params, gt_array)

                image_dataset = np.append(image_dataset, np.concatenate((images_preproc["FLAIR"], images_preproc["T1"]), axis=3), axis=0)
                gt_dataset = np.append(gt_dataset, images_preproc["gt"], axis = 0)
                print(image_dataset.shape)


                sub_name = current_subject.split("/")[-1]
                subjects.append(sub_name.encode("ascii", "ignore"))

                end = time.time() - start

                print("Volume: {} Finished Data Reading and Appending in {:.3f} seconds.".format(idx, end))
            except Exception as e:
                print("Volume: {} Failed Reading Data. Error: {}".format(idx, e))
                continue

            # Write the hdf5 file
        with h5py.File(self.dataset_name, "w") as hf:
            hf.create_dataset('image_dataset', data=image_dataset, compression='gzip')
            hf.create_dataset('gt_dataset', data=gt_dataset, compression='gzip')

            dt = h5py.special_dtype(vlen=str)
            hf.create_dataset("subject", data=subjects, dtype=dt, compression="gzip")

        end_d = time.time() - start_d
        print("Successfully written {} in {:.3f} seconds.".format(self.dataset_name, end_d))



if __name__ == "__main__":

    import argparse

    # Training settings
    parser = argparse.ArgumentParser(description='HDF5-Creation')

    parser.add_argument('--hdf5_name', type=str, default="testsuite_2.hdf5",
                        help='path and name of hdf5-dataset (default: testsuite_2.hdf5)')
    parser.add_argument('--rows_standard', type=int, default=200, help='Height of input to network (Default 200)')
    parser.add_argument('--cols_standard', type=int, default=200, help='Width of input to network (Default 200)')
    parser.add_argument('--data_dir', type=str, default="./testsuite", help="Directory with images to load")
    parser.add_argument('--csv_file', type=str, default=None, help="Csv-file listing subjects to include in file")
    parser.add_argument('--pattern', type=str, default="10*", help="Pattern to match files in directory.")
    parser.add_argument('--T1_name', type=str, default="T1/T1_brain.nii.gz",
                        help="Default name of T1 images. (default T1/T1_orig_defaced.nii.gz)")
    parser.add_argument('--FLAIR_name', type=str, default='T2_FLAIR/T2_FLAIR_brain.nii.gz', help='Default name of T2FLAIR images. (default T2_FLAIR/T2_FLAIR)')
    parser.add_argument('--gt_name', type=str, default='T2_FLAIR/lesions/final_mask.nii.gz',help='Default name for ground truth segmentations (default T2_FLAIR/lesions/final_mask.nii.gz)')

    args = parser.parse_args()

    network_params = {"dataset_name": args.hdf5_name, "rows_standard": args.rows_standard, "cols_standard": args.cols_standard,
                      "data_path": args.data_dir, "csv_file": args.csv_file, "pattern": args.pattern, "T1_name": args.T1_name,
                      "gt_name": args.gt_name, "FLAIR_name": args.FLAIR_name}

    stratified = Dataset(args)
    stratified.create_hdf5_dataset()
