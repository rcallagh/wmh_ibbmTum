#!/usr/bin/env python3
import numpy as np
from keras.utils import Sequence
from wmh.utilities import augmentation

class DataGenerator(Sequence):
    def __init__(self, x, y=None, batch_size=32, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(np.shape(x)[0])
        self.x = x
        self.y = y
        self.on_epoch_end()

    def __len__(self):
        return len(self.indices) // self.batch_size

    def __getitem__(self, index):
        batch = self.index[index * self.batch_size:(index + 1) * self.batch_size]

        X, y = self.__get_data(batch)
        return X, y

    def on_epoch_end(self):
        self.indices = np.arange(np.shape(x)[0])
        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def __get_data(self, batch):
        X = x[batch, ...]
        y = y[batch, ...]


        for i, id in enumerate(batch):
            X[..., 0], X[..., 1], y[..., 0] = augmentation(X[..., 0], X[..., 1], y[..., 0])

        return X, y
