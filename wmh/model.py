#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf
import difflib
import SimpleITK as sitk
import scipy.spatial
from keras.models import Model, load_model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.optimizers import Adam
from evaluation import getDSC, getHausdorff, getLesionDetection, getAVD, getImages  #please download evaluation.py from the WMH website
from keras import backend as K
import h5py

from wmh.losses import * #Get all the loss functions


def get_crop_shape(target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)

### ----define U-net architecture--------------
def get_unet(img_shape = None, f_weight=None, args=None):

    #If we have a file name provided, check if it has the full model
    full_model = False
    if f_weight is not None:
        f = h5py.File(f_weight)
        if 'optimizer_weights' in list(f.keys()):
            full_model = True
        f.close()

    if full_model:
        print("Loading model from {}".format(f_weight))
        model = load_model(f_weight, custom_objects=all_losses)
        if args is not None:
            lossfunc = get_loss
            metricfuncs = get_metrics(args)
            recompile = False
            if lossfunc != model.loss:
                recompile = True
            for metric in metricfuncs:
                if any(model.metrics) == metric:
                    continue
                else:
                    recompile = True
                    break

            if recompile:
                print('Recompiling model with new loss/metrics')
                model.compile(optimizer=model.optimizer, loss=lossfunc, metrics=metricfuncs)

        return model

    if args is not None:
        lr = args.lr
    else:
        lr = 1e-2

    dim_ordering = 'tf'
    inputs = Input(shape = img_shape)
    concat_axis = -1
    ### the size of convolutional kernels is defined here
    conv1 = Conv2D(64, 5, 5, activation='relu', border_mode='same', dim_ordering=dim_ordering, name='conv1_1')(inputs)
    conv1 = Conv2D(64, 5, 5, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv1)
    conv2 = Conv2D(96, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool1)
    conv2 = Conv2D(96, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv2)

    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool2)
    conv3 = Conv2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv3)

    conv4 = Conv2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool3)
    conv4 = Conv2D(256, 4, 4, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering)(conv4)

    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(pool4)
    conv5 = Conv2D(512, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv5)

    up_conv5 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up6)
    conv6 = Conv2D(256, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv6)

    up_conv6 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up7)
    conv7 = Conv2D(128, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv7)

    up_conv7 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = Conv2D(96, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up8)
    conv8 = Conv2D(96, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv8)

    up_conv8 = UpSampling2D(size=(2, 2), dim_ordering=dim_ordering)(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch,cw), dim_ordering=dim_ordering)(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = Conv2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(up9)
    conv9 = Conv2D(64, 3, 3, activation='relu', border_mode='same', dim_ordering=dim_ordering)(conv9)

    ch, cw = get_crop_shape(inputs, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw), dim_ordering=dim_ordering)(conv9)
    conv10 = Conv2D(1, 1, 1, activation='sigmoid', dim_ordering=dim_ordering)(conv9)
    model = Model(input=inputs, output=conv10)

    if args is not None:
        lossfunc = get_loss(args.loss)
        metricfuncs = get_metrics(args)
    else:
        lossfunc = dice_coef_loss
        metricfuncs = [dice_coef_loss]

    model.compile(optimizer=Adam(lr=lr), loss=lossfunc, metrics=metricfuncs)

    if f_weight is not None:
        print("Previous checkpoint only contains weights. Loading in previous weights from {}, but initial training may be poor due to lack of trained optimizer.".format(f_weight))
        model.load_weights(f_weight)

    return model

def get_loss(loss_str):

    if loss_str == 'dice':
        return dice_coef_loss
    elif loss_str == 'dsc':
        return dice_coef_for_training
    elif loss_str == 'jaccard':
        return jaccard_distance_loss
    elif loss_str == 'tversky':
        return tversky_loss
    elif loss_str == 'focal-tversky':
        return focal_tversky

    else:
        print('No loss known for {}'.format(loss_str))
        exit(1)

def get_metrics(args):
    metrics = []
    for metric_str in args.metrics:
        metrics.append(get_loss(metric_str))

    return metrics
