#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import difflib
import SimpleITK as sitk
import scipy

class ProcessingParams:
    def __init__(self):
        self.rows_standard = 200
        self.cols_standard = 200
        self.thresh_T1 = 70
        self.thresh_FLAIR = 30
        self.rhs_row_min = 0
        self.rhs_row_max = 0
        self.rhs_col_min = 0
        self.rhs_col_max = 0
        self.lhs_row_min = 0
        self.lhs_row_max = 0
        self.lhs_col_min = 0
        self.lhs_col_max = 0
        self.brain_CoM = []
        self.two_modalities = True

    def updateFromArgs(self, args):
        self.rows_standard = args.rows_standard
        self.cols_standard = args.cols_standard

def preprocessing(FLAIR_image, T1_image, proc_params, gt_image = None):
    #  start_slice = 10
    channel_num = 2
    print(np.shape(FLAIR_image))
    num_selected_slice = FLAIR_image.shape[0]
    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)
    if gt_image is not None:
        gt_image = np.float32(gt_image)

    brain_mask_FLAIR = np.ndarray((np.shape(FLAIR_image)[0],FLAIR_image.shape[1], FLAIR_image.shape[2]), dtype=np.float32)
    brain_mask_T1 = np.ndarray(T1_image.shape, dtype=np.float32)
    imgs_two_channels = np.ndarray((num_selected_slice, proc_params.rows_standard, proc_params.cols_standard, channel_num), dtype=np.float32)
    imgs_mask_two_channels = np.ndarray((num_selected_slice, proc_params.rows_standard, proc_params.cols_standard,1), dtype=np.float32)
    FLAIR_image_suitable = np.ndarray((num_selected_slice, proc_params.rows_standard, proc_params.cols_standard), dtype=np.float32)
    T1_image_suitable = np.ndarray((num_selected_slice, proc_params.rows_standard, proc_params.cols_standard), dtype=np.float32)
    if gt_image is not None:
        gt_image_suitable = np.ndarray((num_selected_slice, proc_params.rows_standard, proc_params.cols_standard), dtype=np.float32)

    # FLAIR --------------------------------------------
    brain_mask_FLAIR[FLAIR_image >= proc_params.thresh_FLAIR] = 1
    brain_mask_FLAIR[FLAIR_image < proc_params.thresh_FLAIR] = 0
    for iii in range(np.shape(FLAIR_image)[0]):

        brain_mask_FLAIR[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_FLAIR[iii,:,:])  #fill the holes inside brain


    #------Extract brain CoM
    brain_CoM = scipy.ndimage.center_of_mass(brain_mask_FLAIR)
    brain_CoM_row = int(brain_CoM[1])
    brain_CoM_col = int(brain_CoM[2])
    proc_params.brain_CoM = brain_CoM

    #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])

    proc_params = setCroppingParams(proc_params, FLAIR_image.shape)
    rhs_row_min = proc_params.rhs_row_min
    rhs_row_max = proc_params.rhs_row_max
    rhs_col_min = proc_params.rhs_col_min
    rhs_col_max = proc_params.rhs_col_max
    lhs_row_min = proc_params.lhs_row_min
    lhs_row_max = proc_params.lhs_row_max
    lhs_col_min = proc_params.lhs_col_min
    lhs_col_max = proc_params.lhs_col_max

#    import pdb; pdb.set_trace()
    FLAIR_image_suitable[...] = np.min(FLAIR_image)

#    import pdb; pdb.set_trace()
    FLAIR_image_suitable[:, lhs_row_min:lhs_row_max, lhs_col_min:lhs_col_max] = FLAIR_image[:, rhs_row_min:rhs_row_max, rhs_col_min:rhs_col_max]
    # filename_resultImage = os.path.join(outputDir,'FLAIR_crop.nii.gz')
    # sitk.WriteImage(sitk.GetImageFromArray(FLAIR_image_suitable), filename_resultImage )
    #   # T1 -----------------------------------------------
    if proc_params.two_modalities:
        brain_mask_T1[T1_image >= proc_params.thresh_T1] = 1
        brain_mask_T1[T1_image <  proc_params.thresh_T1] = 0
        for iii in range(np.shape(T1_image)[0]):
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
        T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
        T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:, lhs_row_min:lhs_row_max, lhs_col_min:lhs_col_max] = T1_image[:, rhs_row_min:rhs_row_max, rhs_col_min:rhs_col_max]
        # filename_resultImage = os.path.join(outputDir,'T1_crop.nii.gz')
        # sitk.WriteImage(sitk.GetImageFromArray(T1_image_suitable), filename_resultImage, imageIO="NiftiImageIO")
    #---------------------------------------------------
    if gt_image is not None:
        gt_image_suitable[:, lhs_row_min:lhs_row_max, lhs_col_min:lhs_col_max] = gt_image[:, rhs_row_min:rhs_row_max, rhs_col_min:rhs_col_max]
        gt_image_suitable = gt_image_suitable[..., np.newaxis]

    FLAIR_image_suitable  = FLAIR_image_suitable[..., np.newaxis]
    T1_image_suitable  = T1_image_suitable[..., np.newaxis]

    imgs_out = {"FLAIR":None, "T1" :None, "gt":None}
    imgs_out['FLAIR'] = FLAIR_image_suitable
    if proc_params.two_modalities:
        imgs_out['T1'] = T1_image_suitable

    if gt_image is not None:
        imgs_out['gt'] = gt_image_suitable

    return imgs_out, proc_params

def postprocessing(FLAIR_array, pred, proc_params):
    start_slice = int(np.shape(FLAIR_array)[0]*per)
    num_o = np.shape(FLAIR_array)[1]  # original size
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    original_pred = np.zeros(np.shape(FLAIR_array), dtype=np.float32)
#    import pdb; pdb.set_trace()

    rhs_row_min = proc_params.rhs_row_min
    rhs_row_max = proc_params.rhs_row_max
    rhs_col_min = proc_params.rhs_col_min
    rhs_col_max = proc_params.rhs_col_max
    lhs_row_min = proc_params.lhs_row_min
    lhs_row_max = proc_params.lhs_row_max
    lhs_col_min = proc_params.lhs_col_min
    lhs_col_max = proc_params.lhs_col_max

    original_pred[:, rhs_row_min:rhs_row_max, rhs_col_min:rhs_col_max] = pred[:,lhs_row_min:lhs_row_max,lhs_col_min:lhs_col_max,0]
    original_pred[0: start_slice, ...] = 0
    original_pred[(num_o-start_slice):num_o, ...] = 0
    return original_pred

def setCroppingParams(proc_params, image_shape):

    num_selected_slice = image_shape[0]
    image_rows_Dataset = image_shape[1]
    image_cols_Dataset = image_shape[2]

    rows_standard = proc_params.rows_standard
    cols_standard = proc_params.cols_standard

    rows_to_shrink = image_rows_Dataset >= rows_standard
    cols_to_shrink = image_cols_Dataset >= cols_standard

    brain_CoM_row = int(proc_params.brain_CoM[1])
    brain_CoM_col = int(proc_params.brain_CoM[2])


    row_offset = int(rows_standard/2 - brain_CoM_row)
    col_offset = int(cols_standard/2 - brain_CoM_col)
    row_offset = 0
    col_offset = 0

    if rows_to_shrink:
        lhs_row_min = 0;
        lhs_row_max = rows_standard;

        rhs_row_min = int(0.5 * (image_rows_Dataset - rows_standard)) - row_offset
        rhs_row_max = int(0.5 * (image_rows_Dataset + rows_standard)) - row_offset

        if rhs_row_min < 0:
            overstep = -rhs_row_min
            rhs_row_min = 0
            lhs_row_max -= overstep

    else:
        lhs_row_min = int(0.5 * (rows_standard - image_rows_Dataset)) + row_offset
        lhs_row_max = int(0.5 * (rows_standard + image_rows_Dataset)) + row_offset
        rhs_row_min = 0
        rhs_row_max = image_rows_Dataset

        if lhs_row_max > rows_standard:
            overstep = lhs_row_max - rows_standard
            lhs_row_max = rows_standard
            rhs_row_min += overstep


    if cols_to_shrink:
        lhs_col_min = 0
        lhs_col_max = cols_standard

        rhs_col_min = int(0.5 * (images_cols_Dataset - cols_standard)) - col_offset
        rhs_col_max = int(0.5 * (images_cols_Dataset + cols_standard)) - col_offset

        if rhs_col_min < 0:
            overstep = -rhs_col_min
            rhs_col_min = 0
            lhs_col_max -= ovestep


    else:
        lhs_col_min = int(0.5 * (cols_standard - image_cols_Dataset)) + col_offset
        lhs_col_max = int(0.5 * (cols_standard + image_cols_Dataset)) + col_offset
        rhs_col_min = 0
        rhs_col_max = image_cols_Dataset

        if lhs_col_max > cols_standard:
            overstep = lhs_col_max - cols_standard
            lhs_col_max = cols_standard
            rhs_col_min += overstep

    proc_params.rhs_row_min = rhs_row_min
    proc_params.rhs_row_max = rhs_row_max
    proc_params.rhs_col_min = rhs_col_min
    proc_params.rhs_col_max = rhs_col_max
    proc_params.lhs_row_min = lhs_row_min
    proc_params.lhs_row_max = lhs_row_max
    proc_params.lhs_col_min = lhs_col_min
    proc_params.lhs_col_max = lhs_col_max

    return proc_params
