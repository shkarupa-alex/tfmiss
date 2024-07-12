import numpy as np
import tensorflow as tf


def data_generator(N, seq_length, batch_size):
    """
    Args:
        N: # of data in the set
        seq_length: Length of the adding problem data
        batch_size: Size of batch
    """
    X_num = np.random.rand(N, 1, seq_length)
    X_mask = np.zeros((N, 1, seq_length))
    Y = np.zeros((N, 1))
    for i in range(N):
        positions = np.random.choice(seq_length, size=2, replace=False)
        X_mask[i, 0, positions[0]] = 1
        X_mask[i, 0, positions[1]] = 1
        Y[i, 0] = X_num[i, 0, positions[0]] + X_num[i, 0, positions[1]]
    X = np.concatenate((X_num, X_mask), axis=1)
    X = np.transpose(X, (0, 2, 1))

    return tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)
