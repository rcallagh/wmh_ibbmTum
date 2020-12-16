#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import difflib
import SimpleITK as sitk

def preprocessing(FLAIR_image, T1_image, rowcol_info):
    #  start_slice = 10
    channel_num = 2
    print(np.shape(FLAIR_image))
    num_selected_slice = np.shape(FLAIR_image)[0]
    image_rows_Dataset = np.shape(FLAIR_image)[1]
    image_cols_Dataset = np.shape(FLAIR_image)[2]

    rows_to_shrink = image_rows_Dataset >= rowcol_info["rows_standard"]
    cols_to_shrink = image_cols_Dataset >= rowcol_info["cols_standard"]

    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    brain_mask_T1 = np.ndarray((np.shape(FLAIR_image)[0],image_rows_Dataset, image_cols_Dataset), dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, rows_standard, cols_standard,1), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)
    T1_image_suitable = np.ndarray((num_selected_slice, rows_standard, cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >= rowcol_info["thresh"]] = 1
    brain_mask_FLAIR[FLAIR_image < rowcol_info["thresh"]] = 0
    for iii in range(np.shape(FLAIR_image)[0]):

        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain

    #------Extract brain CoM
    brain_CoM = scipy.ndimage.center_of_mass(brain_mask_FLAIR)
    brain_CoM_row = int(brain_CoM[1])
    brain_CoM_col = int(brain_CoM[2])
def postprocessing(FLAIR_array, pred, rowcol_info):