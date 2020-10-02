from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.ops import tfmiss_ops


def fo_pool(inputs, forget, initial_state=None, time_major=False, name=None):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        inputs: Tensor, input values in [Batch, Time, Channels] format or [Time, Batch, Channels] if time_major
        forget: Tensor, input values in [Batch, Time, Channels] format or [Time, Batch, Channels] if time_major.
            Usually in the range 0-1.
        initial_state: Tensor, initial hidden state values in [Batch, Channels] format.
        time_major: boolean, indicates if time dimension is on axis 0.
        name: A name for the operation (optional).
    Returns:
        Tensor: fo_pooled output, [Batch, Time, Channels] format or [Time, Batch, Channels] if time_major.
    """
    with tf.name_scope(name or 'fo_pool'):
        inputs = tf.convert_to_tensor(inputs)
        forget = tf.convert_to_tensor(forget, dtype=inputs.dtype)

        if initial_state is None:
            inputs_shape = tf.shape(inputs)
            initial_state = tf.zeros(
                (inputs_shape[1] if time_major else inputs_shape[0], inputs_shape[2]), dtype=inputs.dtype)
        else:
            initial_state = tf.convert_to_tensor(initial_state, dtype=inputs.dtype)

        if time_major:
            return tfmiss_ops.miss_time_major_fo_pool(inputs, forget, initial_state)[1:]
        else:
            return tfmiss_ops.miss_batch_major_fo_pool(inputs, forget, initial_state)[:, 1:]
