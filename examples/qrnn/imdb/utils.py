from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


def data_generator(batch_size):
    """
    Args:
        dataset: Dataset name
        seq_length: Length of sequence
        batch_size: Size of batch
    """
    vocab_size = 20000
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)
    x_train, y_train, x_test, y_test = tf.ragged.constant(x_train), tf.ragged.constant(y_train), \
                                       tf.ragged.constant(x_test), tf.ragged.constant(y_test)

    def _flat_labels(x, y):
        return x, y.to_tensor(0)

    # Shuffle only train dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
        .shuffle(batch_size * 100) \
        .batch(batch_size)\
        # \
        # .map(_flat_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
        .batch(batch_size)\
        # \
        # .map(_flat_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return train_dataset, test_dataset, vocab_size
