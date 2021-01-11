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
        self.filename_resultImage = ""

        #Set up subject directories
        if args.csv_file is not None:
            with open(args.csv_file, "r") as s_dirs:
                self.subject_dirs = [line.strip() for line in s_dirs.readlines()]
        else:
            search_pattern = join(self.data_path, self.pattern)
            self.subject_dirs = glob.glob(self.search_pattern)

        self.num_subject = len(self.subject_dirs)

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
        print('Predicting WMH on subject: ' + inputDir)
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

        for i_network in range(self.args.num_unet):
            pred = self.models[i_network].predict(self.imgs_test, batch_size=self.args.batch_size, verbose=self.args.verbose)
            if i_network == 0:
                predictions = pred
            else:
                predictions = np.concatenate((predictions, pred), axis=3)

        self.pred = np.mean(predictions, axis=3)

        self.pred[self.pred > 0.45] = 1      #0.45 thresholding
        self.pred[self.pred <= 0.45] = 0

        self.pred = self.pred[..., np.newaxis]
        # import pdb; pdb.set_trace()
        self.pred = postprocessing(FLAIR_array, self.pred, self.proc_params) # get the original size to match

        self.filename_resultImage = os.path.join(inputDir, self.args.output_name)
        output_img = sitk.GetImageFromArray(self.pred)
        output_img.CopyInformation(FLAIR_image)
        sitk.WriteImage(output_img, self.filename_resultImage, imageIO="NiftiImageIO")

    def compute_metrics():
        pass
