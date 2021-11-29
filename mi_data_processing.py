from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

from tensorflow.keras import layers
import numpy as np

# a wrapper for tensorflow.dataset
class dataflow(object):
    def __init__(self, x, buffersize, batchsize, y=None):
        self.x = x
        self.y = y
        self.buffersize = buffersize
        self.batchsize = batchsize

        if y is not None:
            dx = tf.data.Dataset.from_tensor_slices(x)
            dy = tf.data.Dataset.from_tensor_slices(y)
            self.dataset = tf.data.Dataset.zip((dx, dy))
        else:
            self.dataset = tf.data.Dataset.from_tensor_slices(x)

        self.batched_dataset = self.dataset.batch(batchsize)
        self.shuffled_batched_dataset = self.dataset.shuffle(buffersize).batch(batchsize)

    def get_shuffled_batched_dataset(self):
        return self.shuffled_batched_dataset

    def get_batched_dataset(self):
        return self.batched_dataset

    def update_shuffled_batched_dataset(self):
        self.shuffled_batched_dataset = self.dataset.shuffle(self.buffersize).batch(self.batchsize)
        return self.shuffled_batched_dataset

    def get_n_batch_from_shuffled_batched_dataset(self, n):
        it = iter(self.shuffled_batched_dataset)
        xs = []
        for i in range(n):
            x = next(it)
            if isinstance(x, tuple):
              xs.append(x[0])
            else:
              xs.append(x)
        x = tf.concat(xs, 0)

        return x

    def get_n_batch_from_batched_dataset(self, n):
        it = iter(self.batched_dataset)
        xs = []
        for i in range(n):
            x = next(it)
            if isinstance(x, tuple):
              xs.append(x[0])
            else:
              xs.append(x)
        x = tf.concat(xs, 0)

        return x