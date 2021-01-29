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
from wmh.evaluation import ModelEvaluator
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
    parser.add_argument('--verbose', '-v', action='count', help='Flag to use verbose training. A single flag will cause full verbosity. Double flag (e.g. -vv) will cause less verbosity (use -vv in non-interactive environments like cluster)')
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

    #Initialise model evaluation class
    modelEval = ModelEvaluator(args)
    #Load model parameters
    modelEval.load_model()

    #Loop over subjects and evaluate
    for i_subject in range(0, modelEval.num_subject):
        modelEval.predict(i_subject)
        if args.compute_metrics:
            modelEval.compute_metrics()
    modelEval.write_metrics()
    # import pdb; pdb.set_trace()

if __name__=='__main__':
    main()
