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
        self.ignore_frac = 0

    def updateFromArgs(self, args):
        self.rows_standard = args.rows_standard
        self.cols_standard = args.cols_standard
        self.two_modalities = not args.FLAIR_only
        self.ignore_frac = args.ignore_frac

def preprocessing(FLAIR_image, T1_image, proc_params, gt_image = None):
    #  start_slice = 10
    channel_num = 2
    print(np.shape(FLAIR_image))
    FLAIR_image = np.float32(FLAIR_image)
    T1_image = np.float32(T1_image)
    if gt_image is not None:
        gt_image = np.float32(gt_image)

    FLAIR_image, T1_image, gt_image, zero_slice = strip_empty_slices(FLAIR_image, proc_params, T1_image, gt_image)
    proc_params.zero_slice = zero_slice
    num_selected_slice = FLAIR_image.shape[0]

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
        gt_image_suitable[gt_image_suitable != 1] = 0;
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
    # start_slice = int(np.shape(FLAIR_array)[0]*proc_params.ignore_frac)
    # num_o = np.shape(FLAIR_array)[1]  # original size
    # rows_o = np.shape(FLAIR_array)[1]
    # cols_o = np.shape(FLAIR_array)[2]
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

    zero_slice = proc_params.zero_slice
    original_pred[~zero_slice, rhs_row_min:rhs_row_max, rhs_col_min:rhs_col_max] = pred[:,lhs_row_min:lhs_row_max,lhs_col_min:lhs_col_max,0]
    # original_pred[0: start_slice, ...] = 0
    # original_pred[(num_o-start_slice):num_o, ...] = 0
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

        rhs_col_min = int(0.5 * (image_cols_Dataset - cols_standard)) - col_offset
        rhs_col_max = int(0.5 * (image_cols_Dataset + cols_standard)) - col_offset

        if rhs_col_min < 0:
            overstep = -rhs_col_min
            rhs_col_min = 0
            lhs_col_max -= overstep


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

def strip_empty_slices(FLAIR_array, proc_params, T1_array=None, gt_array=None):
    # FLAIR_min = np.min(FLAIR_array)

    # Find zero mean FLAIR slices
    FLAIR_slice_mean = np.mean(FLAIR_array, axis=(1,2))
    FLAIR_zero_slice = FLAIR_slice_mean == 0

    # Find zero mean T1 slices
    T1_zero_slice = np.zeros(FLAIR_zero_slice.shape, dtype=bool)
    if (T1_array is not None) and (len(T1_array) > 0):
        T1_slice_mean = np.mean(T1_array, axis=(1,2))
        T1_zero_slice = T1_slice_mean == 0

    #Combine zeros
    zero_slice = np.logical_or(FLAIR_zero_slice, T1_zero_slice)

    if proc_params.ignore_frac > 0:
        start_slice = int(FLAIR_array.shape[0] * proc_params.ignore_frac)
        zero_slice[0:start_slice] = True
        zero_slice[-start_slice:] = True

    #Resample arrays
    FLAIR_array = FLAIR_array[~zero_slice, :, :]
    if (T1_array is not None) and (len(T1_array) > 0):
        T1_array = T1_array[~zero_slice, :, :]

    if (gt_array is not None) and (len(gt_array) > 0):
        gt_array = gt_array[~zero_slice, :, :]

    return FLAIR_array, T1_array, gt_array, zero_slice

def augmentation(x_0, x_1, y):
    theta = (np.random.uniform(-15, 15) * np.pi) / 180.
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    shear = np.random.uniform(-.1, .1)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    zx, zy = np.random.uniform(.9, 1.1, 2)
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    augmentation_matrix = np.dot(np.dot(rotation_matrix, shear_matrix), zoom_matrix)
    # import pdb; pdb.set_trace()
    transform_matrix = transform_matrix_offset_center(augmentation_matrix, x_0.shape[0], x_0.shape[1])
    x_0 = apply_transform(x_0[..., np.newaxis], transform_matrix, channel_axis=2)
    x_1 = apply_transform(x_1[..., np.newaxis], transform_matrix, channel_axis=2)
    y = apply_transform(y[..., np.newaxis], transform_matrix, channel_axis=2)
    return x_0[..., 0], x_1[..., 0], y[..., 0]

def transform_matrix_offset_center(matrix, x, y):
    '''Taken from Keras 2.1.5 https://github.com/keras-team/keras/blob/2.1.5/keras/preprocessing/image.py'''
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix

def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    From Keras 2.1.5 https://github.com/keras-team/keras/blob/2.1.5/keras/preprocessing/image.py
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [scipy.ndimage.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=1,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x
