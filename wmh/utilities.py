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

    #------Gaussion Normalization
    FLAIR_image -=np.mean(FLAIR_image[brain_mask_FLAIR == 1])      #Gaussion Normalization
    FLAIR_image /=np.std(FLAIR_image[brain_mask_FLAIR == 1])



#    import pdb; pdb.set_trace()
    FLAIR_image_suitable[...] = np.min(FLAIR_image)
    if rows_to_shrink:
        lhs_row_min = 0;
        lhs_row_max = rows_standard;

        row_offset = int(brain_CoM_row - rows_standard/2)
        if (row_offset + rows_standard/2) > image_rows_Dataset/2:
            row_offset -= int(row_offset + rows_standard/2 - image_rows_Dataset/2)

        rhs_row_min = int(image_rows_Dataset/2-rows_standard/2 + row_offset);
        rhs_row_max = int(image_rows_Dataset/2+rows_standard/2 + row_offset);

    else:
#        import pdb; pdb.set_trace()
        lhs_row_min = int(rows_standard/2 - image_rows_Dataset/2)
        lhs_row_max = int(rows_standard/2 + image_rows_Dataset/2)
        rhs_row_min = 0
        rhs_row_max = image_rows_Dataset

    if cols_to_shrink:
        lhs_col_min = 0
        lhs_col_max = cols_standard

        col_offset = int(brain_CoM_col - cols_standard/2)
        if (col_offset + cols_standard/2) > image_cols_Dataset/2:
            col_offset -= int(col_offset + cols_standard/2 - image_cols_Dataset/2)

        rhs_col_min = int(image_cols_Dataset/2-cols_standard/2 + col_offset)
        rhs_col_max = int(image_cols_Dataset/2+cols_standard/2 + col_offset)

    else:
        lhs_col_min = int(cols_standard/2-image_cols_Dataset/2)
        lhs_col_max = int(cols_standard/2+image_cols_Dataset/2)
        rhs_col_min = 0
        rhs_col_max = image_cols_Dataset

#    import pdb; pdb.set_trace()
    FLAIR_image_suitable[:, lhs_row_min:lhs_row_max, lhs_col_min:lhs_col_max] = FLAIR_image[:, rhs_row_min:rhs_row_max, rhs_col_min:rhs_col_max]
    filename_resultImage = os.path.join(outputDir,'FLAIR_crop.nii.gz')
    sitk.WriteImage(sitk.GetImageFromArray(FLAIR_image_suitable), filename_resultImage )
    #   # T1 -----------------------------------------------
    if rowcol_info["two_modalities"]:
        brain_mask_T1[T1_image >= thresh] = 1
        brain_mask_T1[T1_image <  thresh] = 0
        for iii in range(np.shape(T1_image)[0]):

            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside brain
        #------Gaussion Normalization
        T1_image -=np.mean(T1_image[brain_mask_T1 == 1])      #Gaussion Normalization
        T1_image /=np.std(T1_image[brain_mask_T1 == 1])

        T1_image_suitable[...] = np.min(T1_image)
        T1_image_suitable[:, lhs_row_min:lhs_row_max, lhs_col_min:lhs_col_max] = T1_image[:, rhs_row_min:rhs_row_max, rhs_col_min:rhs_col_max]
        filename_resultImage = os.path.join(outputDir,'T1_crop.nii.gz')
        sitk.WriteImage(sitk.GetImageFromArray(T1_image_suitable), filename_resultImage, imageIO="NiftiImageIO")
def postprocessing(FLAIR_array, pred, rowcol_info):