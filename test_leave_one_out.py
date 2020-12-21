import os
import time
import numpy as np
import warnings
import scipy
import SimpleITK as sitk
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, BatchNormalization, Activation
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import apply_transform, transform_matrix_offset_center
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages
from wmh.utilities import preprocessing, postprocessing
from wmh.model import get_unet
K.set_image_data_format('channels_last')

rows_standard = 200
cols_standard = 200
thresh_FLAIR = 70      #to mask the brain
thresh_T1 = 30
smooth=1.


def test_leave_one_out(patient=0, flair=True, t1=True, full=True, first5=True, aug=True, verbose=False):
    if patient < 20: dir = 'raw/Utrecht/'
    elif patient < 40: dir = 'raw/Singapore/'
    else: dir = 'raw/GE3T/'
    dirs = os.listdir(dir)
    dirs.sort()
    dir += dirs[patient%20]
    FLAIR_image = sitk.ReadImage(dir + '/pre/FLAIR.nii.gz')
    T1_image = sitk.ReadImage(dir + '/pre/T1.nii.gz')
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    T1_array = sitk.GetArrayFromImage(T1_image)
    imgs_test = preprocessing(FLAIR_array, T1_array)
    if not flair: imgs_test = imgs_test[..., 1:2].copy()
    if not t1: imgs_test = imgs_test[..., 0:1].copy()
    img_shape = (rows_standard, cols_standard, flair+t1)
    model = get_unet(img_shape, first5)
    model_path = 'models/'
#if you want to test three models ensemble, just do like this: pred = (pred_1+pred_2+pred_3)/3
    model.load_weights(model_path + str(patient) + '.h5')
    pred = model.predict(imgs_test, batch_size=1, verbose=verbose)
    pred[pred > 0.5] = 1.
    pred[pred <= 0.5] = 0.
    original_pred = postprocessing(FLAIR_array, pred)
    filename_resultImage = model_path + str(patient) + '.nii.gz'
    sitk.WriteImage(sitk.GetImageFromArray(original_pred), filename_resultImage )
    filename_testImage = os.path.join(dir + '/wmh.nii.gz')
    testImage, resultImage = getImages(filename_testImage, filename_resultImage)
    dsc = getDSC(testImage, resultImage)
    avd = getAVD(testImage, resultImage)
    h95 = getHausdorff(testImage, resultImage)
    recall, f1 = getLesionDetection(testImage, resultImage)
    return dsc, h95, avd, recall, f1

def main():
    result = np.ndarray((60,5), dtype = 'float32')
    for patient in range(60):
        dsc, h95, avd, recall, f1 = test_leave_one_out(patient, first5=True, verbose=True)#
        print('Result of patient ' + str(patient))
        print('Dice',                dsc,       '(higher is better, max=1)')
        print('HD',                  h95, 'mm',  '(lower is better, min=0)')
        print('AVD',                 avd,  '%',  '(lower is better, min=0)')
        print('Lesion detection', recall,       '(higher is better, max=1)')
        print('Lesion F1',            f1,       '(higher is better, max=1)')
        result[patient, 0] = dsc
        result[patient, 1] = h95
        result[patient, 2] = avd
        result[patient, 3] = recall
        result[patient, 4] = f1
    np.save('results.npy', result)

if __name__=='__main__':
    main()
