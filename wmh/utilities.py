#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import difflib
import SimpleITK as sitk

def preprocessing(FLAIR_image, T1_image, rowcol_info):
    #  start_slice = 10
def postprocessing(FLAIR_array, pred, rowcol_info):