#!/usr/bin/env python

import os
import numpy as np
import tensorflow as tf
import warnings
import h5py
import SimpleITK as sitk
import scipy.spatial
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from wmh.model import get_unet
from wmh.utilities import preprocessing, postprocessing, ProcessingParams
from time import strftime

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

###################################
# TensorFlow wizardry
config = tf.compat.v1.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True

# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 1.0

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
###################################


def train(args, i_network):
    #Load in training dataset
    f = h5py.File(args.hdf5_name_train)
    images = np.array(f['image_dataset'])
    masks = np.array(f['gt_dataset'])
    # subject = f['subject']

    #Set up on the fly augmentation
    if args.no_aug:
        img_gen = ImageDataGenerator(
            validation_split=0.2
        )
    else:
        img_gen = ImageDataGenerator(
            rotation_range=15,
            zoom_range=0.1,
            shear_range=18,
            validation_split=0.2
        )

    #If resuming, set up path to load in previous state
    weight_path = None
    if args.resume:
        #Back up previous checkpoint/weights
        if args.FLAIR_only:
            weight_str = os.path.join(args.model_dir, 'FLAIR_only', str(i_network))
        else:
            weight_str = os.path.join(args.model_dir, 'FLAIR_T1', str(i_network))
        # weight_str = os.path.join(args.model_dir,str(i_network))
        os.popen('cp {}.h5 {}_orig_{}.h5'.format(weight_str, weight_str, strftime('%d-%m-%y_%H%M')))

        weight_path = weight_str + '.h5'


    num_channel = 2
    if args.FLAIR_only:
        num_channel = 1

    samples_num = images.shape[0]
    row = images.shape[1]
    col = images.shape[2]
    img_shape = (row, col, num_channel)

    # augmen, augment = augmentation(images[0,...,0], images[0,...,1], masks[0,...])

    #Get the unet. If weight path provided this will load in previous state
    model = get_unet(img_shape, weight_path, args.lr)
    current_epoch = 1
    bs = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    train_gen = img_gen.flow(images, masks, batch_size=bs, shuffle=True, subset='training')
    validation_gen = img_gen.flow(images, masks, batch_size=bs, shuffle=True, subset='validation')

    if args.output_test_aug:
        aug_test_img = img_gen.flow(images, batch_size=1, seed=1234, subset='training',save_to_dir=args.model_dir,save_prefix='img', save_format='png')
        total = 0
        for image in aug_test_img:
            total+=1
            if total > 10:
                break

        aug_test_mask = img_gen.flow(masks, batch_size=1, seed=1234, subset='training',save_to_dir=args.model_dir,save_prefix='masks', save_format='png')
        total = 0
        for image in aug_test_mask:
            total+=1
            if total > 10:
                break


    if args.FLAIR_only:
        model_path = os.path.join(args.model_dir, 'FLAIR_only', (str(i_network) + '.h5'))
    else:
        model_path = os.path.join(args.model_dir, 'FLAIR_T1', (str(i_network) + '.h5'))
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = model.fit(
        images,
        masks,
        batch_size = bs,
        validation_split=0.2,
        epochs=epochs,
        verbose=verbose,
        shuffle=True,
        callbacks = callbacks_list
    )

    # model_path = args.model_dir
    # if not os.path.exists(model_path):
    #     os.mkdir(model_path)
    # import pdb; pdb.set_trace()



    # model_path += str(i_network) + '.h5'
    # model.save_weights(model_path)
    # model.save(model_path)
    # print('Model saved to ', model_path)

    f.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='WMH training')

    parser.add_argument('--data_dir', type=str, default='./', help='Directory containing data. Will be overrided by --csv_file is supplied')
    parser.add_argument('--csv_file', type=str, default=None, help="Csv-file listing subjects to include in file")
    parser.add_argument('--pattern', type=str, default="10*", help="Pattern to match files in directory.")
    parser.add_argument('--T1_name', type=str, default="T1/T1_brain.nii.gz",
                        help="Default name of T1 images. (default T1/T1_brain.nii.gz)")
    parser.add_argument('--FLAIR_name', type=str, default='T2_FLAIR/T2_FLAIR_brain.nii.gz', help='Default name of T2FLAIR images. (default T2_FLAIR/T2_FLAIR)')
    parser.add_argument('--gt_name', type=str, default='T2_FLAIR/lesions/final_mask.nii.gz',help='Default name for ground truth segmentations (default T2_FLAIR/lesions/final_mask.nii.gz)')
    parser.add_argument('--output_name', type=str, default="wmh_seg.nii.gz", help='Name of ouput segmentation file. (default wmh_seg.nii.gz)')
    parser.add_argument('--rows_standard', type=int, default=200, help='Height of input to network (Default 200)')
    parser.add_argument('--cols_standard', type=int, default=200, help='Width of input to network (Default 200)')
    parser.add_argument('--batch_size', type=int, default=30, metavar='N', help='input batch size for training (default: 30)')
    parser.add_argument('--verbose', action='store_true', help='Flag to use verbose training')
    parser.add_argument('--model_dir', type=str, default='./wmh/weights/', help='path to store model weights to (also path containing starting weights for --resume) (default: ./wmh/weights)')
    parser.add_argument('--FLAIR_only', action='store_true', help='Flag whether to just use FLAIR (default (if flag not provided): use FLAIR and T1)')
    parser.add_argument('--num_unet', type=int, default=1, help='Number of networks to train (default: 1)')
    parser.add_argument('--num_unet_start', type=int, default=0, help='Number from which to start training networks (i.e. start from network 1 if network 0 is done) (default: 0)')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    # images = np.load('images_three_datasets_sorted.npy')
    # masks = np.load('masks_three_datasets_sorted.npy')

    proc_params = ProcessingParams()
    proc_params.updateFromArgs(args)

    if args.csv_file is not None:
        with open(args.csv_file, "r") as s_dirs:
            subject_dirs = [line.strip() for line in s_dirs.readlines()]
    else:
        search_pattern = join(self.data_path, self.pattern)
        subject_dirs = glob.glob(self.search_pattern)


    i_start = args.num_unet_start
    models = []
    for i_network in range(i_start, i_start+args.num_unet):
        if args.FLAIR_only:
            weight_str = os.path.join(args.model_dir, 'FLAIR_only', str(i_network))
            img_shape=(args.rows_standard, args.cols_standard, 1)
        else:
            weight_str = os.path.join(args.model_dir, 'FLAIR_T1', str(i_network))
            img_shape=(args.rows_standard, args.cols_standard, 2)

        weight_path = weight_str + '.h5'
        model = get_unet(img_shape, weight_path)
        models.append(model)

    num_subject = len(subject_dirs)
    import pdb; pdb.set_trace()
    for i_subject in range(0, num_subject):
        inputDir = subject_dirs[i_subject]
        if not args.FLAIR_only:
            FLAIR_image = sitk.ReadImage(os.path.join(inputDir, args.FLAIR_name), imageIO="NiftiImageIO")
            FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
            T1_image = sitk.ReadImage(os.path.join(inputDir, args.T1_name), imageIO="NiftiImageIO")
            T1_array = sitk.GetArrayFromImage(T1_image)
            gt_image = sitk.ReadImage(os.path.join(inputDir, args.gt_name), imageIO="NiftiImageIO")
            gt_array = sitk.getArrayFromImage(gt_array)
            [images_preproc, proc_params] = preprocessing(np.float32(FLAIR_array), np.float32(T1_array), proc_params, gt_array)  # data preprocessing
            imgs_test = np.concatenate((images_preproc["FLAIR"], images_preproc["T1"]), axis=3)
        else:
            FLAIR_image = sitk.ReadImage(os.path.join(inputDir, args.FLAIR_name), imageIO="NiftiImageIO") #data preprocessing
            FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
            T1_array = []
            gt_image = sitk.ReadImage(os.path.join(inputDir, args.gt_name), imageIO='NiftiImageIO')
            gt_array = sitk.getArrayFromImage(gt_array)
            [images_preproc, proc_params] = preprocessing(np.float32(FLAIR_array), np.float32(T1_array), proc_params, gt_array)
            imgs_test = images_preproc["FLAIR"]

        predictions = []
        for i_network in range(0, args.num_unet):
            pred = model[i_network].fit(imgs_test, batch_size=args.batch_size, verbose=args.verbose)
            predictions.append(pred)
            import pdb; pdb.set_trace()



if __name__=='__main__':
    main()
