#!/usr/bin/env python
import os
import numpy as np
import tensorflow as tf
import difflib
import SimpleITK as sitk

def preprocessing(FLAIR_image, T1_image):
def postprocessing(FLAIR_array, pred, rowcol_minmax):