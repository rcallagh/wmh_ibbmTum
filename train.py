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
    images = f['image_dataset']
    masks = f['gt_dataset']
    subject = f['subject']

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
        weight_str = os.path.join(args.model_dir,str(i_network))
        os.popen('cp {}.h5 {}_orig_{}.h5'.format(weight_str, weight_str, strftime('%d-%m-%y_%H%M')))

        weight_path = os.path.join(args.model_dir,str(i_network)) + '.h5'


    num_channel = 2
    if args.flair_only:
        num_channel = 1

    samples_num = images.shape[0]
    row = images.shape[1]
    col = images.shape[2]
    img_shape = (row, col, num_channel)

    #Get the unet. If weight path provided this will load in previous state
    model = get_unet(img_shape, weight_path, args.lr)
    current_epoch = 1
    bs = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    train_gen = img_gen.flow(images, masks, batch_size=bs, shuffle=True, subset='training')
    validation_gen = img_gen.flow(images, masks, batch_size=bs, shuffle=True, subset='validation')

    model_path = os.path.join(args.model_dir, (str(i_network) + '.h5'))
    checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.n / bs,
        validation_data = validation_gen,
        validation_steps = validation_gen.n / bs,
        epochs=epochs,
        verbose=verbose,
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

    parser.add_argument('--hdf5_name_train', type=str, default="test_train.hdf5", help='path and name of hdf5-dataset for training (default: test_train.hdf5)')
    parser.add_argument('--hdf5_name_test', type=str, default="test_test.hdf5", help='path and name of hdf5-dataset for testing (default: test_test.hdf5)')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Fraction of data for validation. Will be overridden by hdf5_name_test for explicit validation set. (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=30, metavar='N', help='input batch size for training (default: 30)')
    parser.add_argument('--validation_batch_size', type=int, default=30, metavar='N',help='input batch size for validation (default: 30)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--verbose', action='store_true', help='Flag to use verbose training')
    parser.add_argument('--model_dir', type=str, default='./wmh/weights', help='path to store model weights to (also path containing starting weights for --resume) (default: ./wmh/weights)')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from checkpoints.')
    parser.add_argument('--flair_only', action='store_true', help='Flag whether to just use FLAIR (default (if flag not provided): use FLAIR and T1)')
    parser.add_argument('--no_aug', action='store_true', help="Flag to not do any augmentation")
    parser.add_argument('--num_unet', type=int, default=1, help='Number of networks to train (default: 1)')
    parser.add_argument('--num_unet_start', type=int, default=0, help='Number from which to start training networks (i.e. start from network 1 if network 0 is done) (default: 0)')
    parser.add_argument('--test_ensemble', action='store_true', help='Flag to test the overall ensemble performance once all networks are trained')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 2e-4)')

    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    # images = np.load('images_three_datasets_sorted.npy')
    # masks = np.load('masks_three_datasets_sorted.npy')
    i_start = args.num_unet_start
    for i_net in range(i_start, i_start + args.num_unet):
        train(args, i_net)

if __name__=='__main__':
    main()