from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras.datasets import mnist


def data_generator(permute, batch_size):
    """
    Args:
        permute: Use permuted MNIST
        batch_size: Size of batch
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28 * 28, 1).astype('float32') / 255.
    x_test = x_test.reshape(-1, 28 * 28, 1).astype('float32') / 255.

    if permute:
        shuffle_index = np.random.permutation(28 * 28)
        x_train, x_test = x_train[:, shuffle_index, :], x_test[:, shuffle_index, :]

    return tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size), \
           tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
