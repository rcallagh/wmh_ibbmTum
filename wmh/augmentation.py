#!/usr/bin/env python3
import numpy as np
from keras.utils import Sequence
from wmh.utilities import augmentation

class DataGenerator(Sequence):
    def __init__(self, x, y=None, aug_params=None, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(np.shape(x)[0])
        self.x = x
        self.y = y
        if aug_params is None:
            self.aug_params = {'theta': 0, 'shear': 0, 'scale': 0}
        else:
            self.aug_params = aug_params
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        batch = self.indices[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(np.shape(self.x)[0])
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __get_data(self, batch):
        X = self.x[batch, ...]
        y = self.y[batch, ...]
        # print('in DataGenerator.__get_data')
        # print(np.shape(X))

        for i, id in enumerate(batch):
            X[i, ..., 0], X[i, ..., 1], y[i, ..., 0] = augmentation(X[i, ..., 0], X[i, ..., 1], y[i, ..., 0], aug_params=self.aug_params)

        return X, y
