import numpy as np
import tensorflow as tf


def data_generator(T, mem_length, N, batch_size):
    """
    Args:
        T: The total blank time length
        mem_length: The length of the memory to be recalled
        N: Number of examples
        batch_size: Size of batch
    """
    seq = np.array(np.random.randint(1, 9, size=(N, mem_length)), dtype=float)
    zeros = np.zeros((N, T))
    marker = 9 * np.ones((N, mem_length + 1))
    placeholders = np.zeros((N, mem_length))

    x = np.array(np.concatenate((seq, zeros[:, :-1], marker), 1), dtype=int)
    y = np.array(np.concatenate((placeholders, zeros, seq), 1), dtype=int)

    x = np.expand_dims(x, axis=2)
    y = np.expand_dims(y, axis=2)

    return tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
