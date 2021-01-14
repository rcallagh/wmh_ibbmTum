#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf
import warnings
import h5py
import SimpleITK as sitk
import scipy.spatial
import scipy.io as sio
import difflib
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from wmh.model import get_unet
from wmh.utilities import preprocessing, postprocessing, ProcessingParams
import glob


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

        #Set up image pre/post processing paramters
        self.proc_params = ProcessingParams()
        self.proc_params.updateFromArgs(args)

        self.imgs_test = []
        self.pred = []
        self.filename_resultImage = ""

        #Arrays to store scores
        self.DSC = []
        self.Hausdorff = []
        self.recall = []
        self.f1 = []
        self.AVD = []

        #Set up subject directories
        if args.csv_file is not None:
            with open(args.csv_file, "r") as s_dirs:
                self.subject_dirs = [line.strip() for line in s_dirs.readlines()]
        else:
            search_pattern = os.path.join(self.data_path, self.pattern)
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
        self.last_subject = inputDir
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

    def compute_metrics(self, i_subject=None):
        if i_subject is None:
            subjectDir = self.last_subject
        else:
            subjectDir = self.subject_dirs[i_subject]

        gt_filename = os.path.join(subjectDir, self.gt_name)

        #Ground truth image, thresholded to remove the non-WMH labels
        gt_image = sitk.ReadImage(gt_filename, imageIO="NiftiImageIO")
        gt_image = sitk.BinaryThreshold(gt_image, 0.5, 1.5, 1, 0)

        #Read in the output image
        out_image = sitk.ReadImage(self.filename_resultImage, imageIO="NiftiImageIO")
        out_image.CopyInformation(gt_image)
        out_image = sitk.BinaryThreshold(out_image, 0.5, 1000, 1, 0)

        sio.savemat(os.path.join(subjectDir, 'gt_npy.mat'), {'gt':sitk.GetArrayFromImage(gt_image)})
        sio.savemat(os.path.join(subjectDir, 'out_npy.mat'), {'out':sitk.GetArrayFromImage(out_image)})

        import pdb; pdb.set_trace()
        DSC = getDSC(gt_image, out_image)
        # h95 = getHausdorff(gt_image, out_image) #Apparently a problem in python 3 with HD
        recall, f1 = getLesionDetection(gt_image, out_image)
        AVD = getAVD(gt_image, out_image)

        self.DSC.append(DSC)
        self.recall.append(recall)
        self.f1.append(f1)
        self.AVD.append(AVD)

        if self.args.verbose is not None:
            print('Dice Score : {:.3f}'.format(DSC))
            print('Recall     : {:.3f}'.format(recall))
            print('F1         : {:.3f}'.format(f1))
            print('Volume Diff: {:.3f}'.format(AVD))





#----------------------------------------------------
# Evaluation functions from evaluation.py in original sysu_media repo, proided by MICCAI WMH challenge
# ---------------------------------------------------
# def getImages(testFilename, resultFilename):
#     """Return the test and result images, thresholded and non-WMH masked."""
#     testImage   = sitk.ReadImage(testFilename)
#     resultImage = sitk.ReadImage(resultFilename)
#     assert testImage.GetSize() == resultImage.GetSize()
#     # Get meta data from the test-image, needed for some sitk methods that check this
#     resultImage.CopyInformation(testImage)

#     # Remove non-WMH from the test and result images, since we don't evaluate on that
#     maskedTestImage = sitk.BinaryThreshold(testImage, 0.5,  1.5, 1, 0) # WMH == 1
#     nonWMHImage     = sitk.BinaryThreshold(testImage, 1.5,  2.5, 0, 1) # non-WMH == 2
#     maskedResultImage = sitk.Mask(resultImage, nonWMHImage)

#     # Convert to binary mask
#     if 'integer' in maskedResultImage.GetPixelIDTypeAsString():
#         bResultImage = sitk.BinaryThreshold(maskedResultImage, 1, 1000, 1, 0)
#     else:
#         bResultImage = sitk.BinaryThreshold(maskedResultImage, 0.5, 1000, 1, 0)

#     return maskedTestImage, bResultImage


# def getResultFilename(participantDir):
#     """Find the filename of the result image.

#     This should be result.nii.gz or result.nii. If these files are not present,
#     it tries to find the closest filename."""
#     files = os.listdir(participantDir)

#     if not files:
#         raise Exception("No results in "+ participantDir)

#     resultFilename = None
#     if 'result.nii.gz' in files:
#         resultFilename = os.path.join(participantDir, 'result.nii.gz')
#     elif 'result.nii' in files:
#         resultFilename = os.path.join(participantDir, 'result.nii')
#     else:
#         # Find the filename that is closest to 'result.nii.gz'
#         maxRatio = -1
#         for f in files:
#             currentRatio = difflib.SequenceMatcher(a = f, b = 'result.nii.gz').ratio()

#             if currentRatio > maxRatio:
#                 resultFilename = os.path.join(participantDir, f)
#                 maxRatio = currentRatio

#     return resultFilename


def getDSC(testImage, resultImage):
    """Compute the Dice Similarity Coefficient."""
    testArray   = sitk.GetArrayFromImage(testImage).flatten()
    resultArray = sitk.GetArrayFromImage(resultImage).flatten()

    # similarity = 1.0 - dissimilarity
    return 1.0 - scipy.spatial.distance.dice(testArray, resultArray)


def getHausdorff(testImage, resultImage):
    """Compute the Hausdorff distance."""

    # Edge detection is done by ORIGINAL - ERODED, keeping the outer boundaries of lesions. Erosion is performed in 2D
    eTestImage   = sitk.BinaryErode(testImage, (1,1,0) )
    eResultImage = sitk.BinaryErode(resultImage, (1,1,0) )

    hTestImage   = sitk.Subtract(testImage, eTestImage)
    hResultImage = sitk.Subtract(resultImage, eResultImage)

    hTestArray   = sitk.GetArrayFromImage(hTestImage)
    hResultArray = sitk.GetArrayFromImage(hResultImage)

    # Convert voxel location to world coordinates. Use the coordinate system of the test image
    # np.nonzero   = elements of the boundary in numpy order (zyx)
    # np.flipud    = elements in xyz order
    # np.transpose = create tuples (x,y,z)
    # testImage.TransformIndexToPhysicalPoint converts (xyz) to world coordinates (in mm)
    testCoordinates   = np.apply_along_axis(testImage.TransformIndexToPhysicalPoint, 1, np.transpose( np.flipud( np.nonzero(hTestArray) )).astype(int) )
    resultCoordinates = np.apply_along_axis(testImage.TransformIndexToPhysicalPoint, 1, np.transpose( np.flipud( np.nonzero(hResultArray) )).astype(int) )

    # Use a kd-tree for fast spatial search
    def getDistancesFromAtoB(a, b):
        kdTree = scipy.spatial.KDTree(a, leafsize=100)
        return kdTree.query(b, k=1, eps=0, p=2)[0]

    # Compute distances from test to result; and result to test
    dTestToResult = getDistancesFromAtoB(testCoordinates, resultCoordinates)
    dResultToTest = getDistancesFromAtoB(resultCoordinates, testCoordinates)

    return max(np.percentile(dTestToResult, 95), np.percentile(dResultToTest, 95))


def getLesionDetection(testImage, resultImage):
    """Lesion detection metrics, both recall and F1."""

    # Connected components will give the background label 0, so subtract 1 from all results
    ccFilter = sitk.ConnectedComponentImageFilter()
    ccFilter.SetFullyConnected(True)

    # Connected components on the test image, to determine the number of true WMH.
    # And to get the overlap between detected voxels and true WMH
    ccTest = ccFilter.Execute(testImage)
    lResult = sitk.Multiply(ccTest, sitk.Cast(resultImage, sitk.sitkUInt32))

    ccTestArray = sitk.GetArrayFromImage(ccTest)
    lResultArray = sitk.GetArrayFromImage(lResult)

    # recall = (number of detected WMH) / (number of true WMH)
    recall = float(len(np.unique(lResultArray)) - 1) / (len(np.unique(ccTestArray)) - 1)

    # Connected components of results, to determine number of detected lesions
    ccResult = ccFilter.Execute(resultImage)
    ccResultArray = sitk.GetArrayFromImage(ccResult)

    # precision = (number of detected WMH) / (number of all detections)
    precision = float(len(np.unique(lResultArray)) - 1) / float(len(np.unique(ccResultArray)) - 1)

    f1 = 2.0 * (precision * recall) / (precision + recall)

    return recall, f1


def getAVD(testImage, resultImage):
    """Volume statistics."""
    # Compute statistics of both images
    testStatistics   = sitk.StatisticsImageFilter()
    resultStatistics = sitk.StatisticsImageFilter()

    testStatistics.Execute(testImage)
    resultStatistics.Execute(resultImage)

    return float(abs(testStatistics.GetSum() - resultStatistics.GetSum())) / float(testStatistics.GetSum()) * 100
