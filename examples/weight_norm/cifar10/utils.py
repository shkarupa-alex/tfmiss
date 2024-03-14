from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_keras.datasets import cifar10
from tf_keras_preprocessing.image import ImageDataGenerator


def data_generator(batch_size):
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    X_train = X_train.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255

    datagen = ImageDataGenerator(featurewise_center=True, zca_whitening=True)
    datagen.fit(X_train)

    X_train = datagen.standardize(X_train)
    X_test = datagen.standardize(X_test)

    return tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(batch_size), \
           tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(batch_size)
