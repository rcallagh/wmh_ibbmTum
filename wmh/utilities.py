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
def postprocessing(FLAIR_array, pred, rowcol_info):