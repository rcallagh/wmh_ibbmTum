#!/usr/bin/env python

import os
import numpy as np
from numpy.random import default_rng
import tensorflow as tf
import warnings
import h5py
import SimpleITK as sitk
import scipy.spatial
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from wmh.model import get_unet
from wmh.utilities import augmentation
from wmh.evaluation import ModelEvaluator
from time import strftime

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

print('Doing the tensor flow stuff')
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
print('Done the tensoflow stuff')

def train(args, i_network):
    #Load in training dataset
    print('Loading data')
    f = h5py.File(args.hdf5_name_train)
    images = np.array(f['image_dataset'])
    masks = np.array(f['gt_dataset'])
    # subject = f['subject']
    f.close()
    print('Loaded data')


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
    print('loaded model')

    num_channel = 2
    if args.FLAIR_only:
        num_channel = 1

    samples_num = images.shape[0]
    row = images.shape[1]
    col = images.shape[2]
    img_shape = (row, col, num_channel)

    #Augmentation
    if not args.no_aug:
        num_aug_sample = int(samples_num * args.aug_factor)
        rng = default_rng()
        samples = rng.integers(0, samples_num-1, (num_aug_sample,1))
        # import pdb; pdb.set_trace()
        images_aug = np.zeros((num_aug_sample, row, col, num_channel), dtype=np.float32)
        masks_aug = np.zeros((num_aug_sample, row, col, num_channel), dtype=np.float32)
        for i in range(len(samples)):
            images_aug[i, ..., 0], images_aug[i, ..., 1], masks_aug[i, ..., 0] = augmentation(images[int(samples[i]), ..., 0], images[int(samples[i]), ..., 1], masks[int(samples[i]), ..., 0])
            if args.output_test_aug:
                if i < 10:
                    sitk.WriteImage(sitk.GetImageFromArray(images_aug[i, ..., 0]), '/SAN/medic/camino_2point0/Ross/test{}.png'.format(i))
        exit(1)
        images = np.concatenate((images, images_aug), axis=0)
        masks = np.concatenate((masks, masks_aug), axis=0)
    # augmen, augment = augmentation(images[0,...,0], images[0,...,1], masks[0,...])

    #Get the unet. If weight path provided this will load in previous state
    model = get_unet(img_shape, weight_path, args.lr)
    current_epoch = 1
    bs = args.batch_size
    epochs = args.epochs
    verbose = args.verbose

    '''
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
    '''

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

    parser.add_argument('--hdf5_name_train', type=str, default="test_train.hdf5", help='path and name of hdf5-dataset for training (default: test_train.hdf5)')
    parser.add_argument('--hdf5_name_test', type=str, default="test_test.hdf5", help='path and name of hdf5-dataset for testing (default: test_test.hdf5)')
    parser.add_argument('--validation_split', type=float, default=0.2, help='Fraction of data for validation. Will be overridden by hdf5_name_test for explicit validation set. (default: 0.2)')
    parser.add_argument('--batch_size', type=int, default=30, metavar='N', help='input batch size for training (default: 30)')
    parser.add_argument('--validation_batch_size', type=int, default=30, metavar='N',help='input batch size for validation (default: 30)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Flag to use verbose training output. -v will have progress bar per epoch, -vv will print one line per epoch (use this in non-interactive runs e.g. cluster)')
    parser.add_argument('--model_dir', type=str, default='./wmh/weights/', help='path to store model weights to (also path containing starting weights for --resume) (default: ./wmh/weights)')
    parser.add_argument('--resume', action='store_true', help='Flag to resume training from checkpoints.')
    parser.add_argument('--FLAIR_only', action='store_true', help='Flag whether to just use FLAIR (default (if flag not provided): use FLAIR and T1)')
    parser.add_argument('--no_aug', action='store_true', help="Flag to not do any augmentation")
    parser.add_argument('--aug_factor', type=float, default=1, help="Factor by which to increase dataset by using augmentation. i.e. the dataset will be x times bigger after augmentation (default: 1 (results in twice as big a dataset))")
    parser.add_argument('--num_unet', type=int, default=1, help='Number of networks to train (default: 1)')
    parser.add_argument('--num_unet_start', type=int, default=0, help='Number from which to start training networks (i.e. start from network 1 if network 0 is done) (default: 0)')
    parser.add_argument('--test_ensemble', action='store_true', help='Flag to test the overall ensemble performance once all networks are trained')
    parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate (default: 2e-4)')
    parser.add_argument('--output_test_aug', action='store_true', help='Flag to save 10 test images from augmentation generator')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    # images = np.load('images_three_datasets_sorted.npy')
    # masks = np.load('masks_three_datasets_sorted.npy')
    i_start = args.num_unet_start
    for i_net in range(i_start, i_start + args.num_unet):
        print('Training net {}'.format(i_net))
        train(args, i_net)

if __name__=='__main__':
    main()
