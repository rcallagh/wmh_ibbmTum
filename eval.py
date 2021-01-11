#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf
import warnings
import h5py
import SimpleITK as sitk
import scipy.spatial
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from wmh.model import get_unet
from wmh.utilities import preprocessing, postprocessing, ProcessingParams
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD
from time import strftime

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

###################################
# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 1.0

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
###################################

class ModelEvaluator():
    def __init__(self, args):
        self.args = args
        self.i_start = args.num_unet_start

        self.models = []

        #Store a few useful things from the args
        self.T1_name = args.T1_name
        self.FLAIR_name = args.FLAIR_name
        self.gt_name = args.gt_name
        self.output_name = args.output_name

        self.FLAIR_only = args.FLAIR_only
        self.compute_metrics = args.compute_metrics

        #Set up image pre/post processing paramters
        self.proc_params = ProcessingParams()
        self.proc_params.updateFromArgs(args)

        self.imgs_test = []
        self.pred = []
        self.filename_resultImage

        #Set up subject directories
        if args.csv_file is not None:
            with open(args.csv_file, "r") as s_dirs:
                self.subject_dirs = [line.strip() for line in s_dirs.readlines()]
        else:
            search_pattern = join(self.data_path, self.pattern)
            self.subject_dirs = glob.glob(self.search_pattern)

    def load_model(self):
        for i_network in range(self.i_start, self.i_start + self.args.num_unet):
            if self.FLAIR_only:
                weight_str = os.path.join(self.args.model_dir, 'FLAIR_only', str(i_network))
                img_shape=(self.proc_params.rows_standard, self.proc_params.cols_standard, 1)
            else:
                weight_str = os.path.join(self.args.model_dir, 'FLAIR_T1', str(i_network))
                img_shape=(self.proc_params.rows_standard, self.proc_params.cols_standard, 2)

            weight_path = weight_str + '.h5'
            model = get_unet(img_shape, weight_path)
            self.models.append(model)

    def predict(self, i_subject):
        inputDir = self.subject_dirs[i_subject]
        if not self.FLAIR_only:
            FLAIR_image = sitk.ReadImage(os.path.join(inputDir, self.FLAIR_name), imageIO="NiftiImageIO")
            FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
            T1_image = sitk.ReadImage(os.path.join(inputDir, self.T1_name), imageIO="NiftiImageIO")
            T1_array = sitk.GetArrayFromImage(T1_image)
            # if self.compute_metrics:
                # gt_image = sitk.ReadImage(os.path.join(inputDir, self.gt_name), imageIO="NiftiImageIO")
                # gt_array = sitk.GetArrayFromImage(gt_image)
            # else:
                # gt_array = []
            [images_preproc, self.proc_params] = preprocessing(np.float32(FLAIR_array), np.float32(T1_array), self.proc_params)  # data preprocessing
            self.imgs_test = np.concatenate((images_preproc["FLAIR"], images_preproc["T1"]), axis=3)
        else:
            FLAIR_image = sitk.ReadImage(os.path.join(inputDir, self.FLAIR_name), imageIO="NiftiImageIO") #data preprocessing
            FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
            T1_array = []
            # if self.compute_metrics:
                # gt_image = sitk.ReadImage(os.path.join(inputDir, self.gt_name), imageIO='NiftiImageIO')
                # gt_array = sitk.GetArrayFromImage(gt_image)
            # else:
                # gt_array = []
            [images_preproc, self.proc_params] = preprocessing(np.float32(FLAIR_array), np.float32(T1_array), self.proc_params)
            self.imgs_test = images_preproc["FLAIR"]

        for i_network in range(args.num_unet):
            pred = models[i_network].predict(imgs_test, batch_size=args.batch_size, verbose=args.verbose)
            if i_network == 0:
                predictions = pred
            else:
                predictions = np.concatenate((predictions, pred), axis=3)

        self.pred = np.mean(predictions, axis=3)

        self.pred[pred > 0.45] = 1      #0.45 thresholding
        self.pred[pred <= 0.45] = 0

        self.pred = pred[..., np.newaxis]
        # import pdb; pdb.set_trace()
        self.pred = postprocessing(FLAIR_array, self.pred, proc_params) # get the original size to match

        self.filename_resultImage = os.path.join(inputDir, self.args.output_name)
        output_img = sitk.GetImageFromArray(self.pred)
        output_img.CopyInformation(FLAIR_image)
        sitk.WriteImage(output_img, self.filename_resultImage, imageIO="NiftiImageIO")

    def compute_metrics():
        pass



def main():
    import argparse
    parser = argparse.ArgumentParser(description='WMH training')

    parser.add_argument('--data_dir', type=str, default='./', help='Directory containing data. Will be overrided by --csv_file is supplied')
    parser.add_argument('--csv_file', type=str, default=None, help="Csv-file listing subjects to include in file")
    parser.add_argument('--pattern', type=str, default="10*", help="Pattern to match files in directory.")
    parser.add_argument('--T1_name', type=str, default="T1/T1_brain.nii.gz",
                        help="Default name of T1 images. (default T1/T1_brain.nii.gz)")
    parser.add_argument('--FLAIR_name', type=str, default='T2_FLAIR/T2_FLAIR_brain.nii.gz', help='Default name of T2FLAIR images. (default T2_FLAIR/T2_FLAIR)')
    parser.add_argument('--gt_name', type=str, default='T2_FLAIR/lesions/final_mask.nii.gz',help='Default name for ground truth segmentations (default T2_FLAIR/lesions/final_mask.nii.gz)')
    parser.add_argument('--output_name', type=str, default="wmh_seg.nii.gz", help='Name of ouput segmentation file. (default wmh_seg.nii.gz)')
    parser.add_argument('--rows_standard', type=int, default=200, help='Height of input to network (Default 200)')
    parser.add_argument('--cols_standard', type=int, default=200, help='Width of input to network (Default 200)')
    parser.add_argument('--batch_size', type=int, default=30, metavar='N', help='input batch size for training (default: 30)')
    parser.add_argument('--verbose', action='store_true', help='Flag to use verbose training')
    parser.add_argument('--model_dir', type=str, default='./wmh/weights/', help='path to store model weights to (also path containing starting weights for --resume) (default: ./wmh/weights)')
    parser.add_argument('--FLAIR_only', action='store_true', help='Flag whether to just use FLAIR (default (if flag not provided): use FLAIR and T1)')
    parser.add_argument('--num_unet', type=int, default=1, help='Number of networks to train (default: 1)')
    parser.add_argument('--num_unet_start', type=int, default=0, help='Number from which to start training networks (i.e. start from network 1 if network 0 is done) (default: 0)')
    parser.add_argument('--ignore_frac', type=float, default = 0.125, help='Fraction of slices from top and bottome to ignore (default: 0.125)')
    parser.add_argument('--compute_metrics', action='store_true', help='Flag whether to compute metrics after segmentation (requires ground truth)')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    # images = np.load('images_three_datasets_sorted.npy')
    # masks = np.load('masks_three_datasets_sorted.npy')
    modelEval = ModelEvaluator(args)
    modelEval.load_model()
    for i_subject in range(0, num_subject):
        modelEval.predict(i_subject)
        if args.compute_metrics:
            modelEval.compute_metrics()
    import pdb; pdb.set_trace()
    proc_params = ProcessingParams()
    proc_params.updateFromArgs(args)

    if args.csv_file is not None:
        with open(args.csv_file, "r") as s_dirs:
            subject_dirs = [line.strip() for line in s_dirs.readlines()]
    else:
        search_pattern = join(self.data_path, self.pattern)
        subject_dirs = glob.glob(self.search_pattern)


    i_start = args.num_unet_start
    models = []
    for i_network in range(i_start, i_start+args.num_unet):
        if args.FLAIR_only:
            weight_str = os.path.join(args.model_dir, 'FLAIR_only', str(i_network))
            img_shape=(args.rows_standard, args.cols_standard, 1)
        else:
            weight_str = os.path.join(args.model_dir, 'FLAIR_T1', str(i_network))
            img_shape=(args.rows_standard, args.cols_standard, 2)

        weight_path = weight_str + '.h5'
        model = get_unet(img_shape, weight_path)
        models.append(model)

    num_subject = len(subject_dirs)
    for i_subject in range(0, num_subject):
        inputDir = subject_dirs[i_subject]
        if not args.FLAIR_only:
            FLAIR_image = sitk.ReadImage(os.path.join(inputDir, args.FLAIR_name), imageIO="NiftiImageIO")
            FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
            T1_image = sitk.ReadImage(os.path.join(inputDir, args.T1_name), imageIO="NiftiImageIO")
            T1_array = sitk.GetArrayFromImage(T1_image)
            gt_image = sitk.ReadImage(os.path.join(inputDir, args.gt_name), imageIO="NiftiImageIO")
            gt_array = sitk.GetArrayFromImage(gt_image)
            [images_preproc, proc_params] = preprocessing(np.float32(FLAIR_array), np.float32(T1_array), proc_params, gt_array)  # data preprocessing
            imgs_test = np.concatenate((images_preproc["FLAIR"], images_preproc["T1"]), axis=3)
        else:
            FLAIR_image = sitk.ReadImage(os.path.join(inputDir, args.FLAIR_name), imageIO="NiftiImageIO") #data preprocessing
            FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
            T1_array = []
            gt_image = sitk.ReadImage(os.path.join(inputDir, args.gt_name), imageIO='NiftiImageIO')
            gt_array = sitk.GetArrayFromImage(gt_image)
            [images_preproc, proc_params] = preprocessing(np.float32(FLAIR_array), np.float32(T1_array), proc_params, gt_array)
            imgs_test = images_preproc["FLAIR"]

        for i_network in range(args.num_unet):
            pred = models[i_network].predict(imgs_test, batch_size=args.batch_size, verbose=args.verbose)
            if i_network == 0:
                predictions = pred
            else:
                predictions = np.concatenate((predictions, pred), axis=3)

        pred = np.mean(predictions, axis=3)

        pred[pred > 0.45] = 1      #0.45 thresholding
        pred[pred <= 0.45] = 0

        pred = pred[..., np.newaxis]
        import pdb; pdb.set_trace()
        pred = postprocessing(FLAIR_array, pred, proc_params) # get the original size to match

        filename_resultImage = os.path.join(inputDir, args.output_name)
        output_img = sitk.GetImageFromArray(pred)
        output_img.CopyInformation(FLAIR_image)
        sitk.WriteImage(output_img, filename_resultImage, imageIO="NiftiImageIO")


if __name__=='__main__':
    main()
