import tensorflow as tf
from tfmiss.ops import tfmiss_ops


def fo_pool(inputs, forget, initial_state=None, name=None):
    """Applies a single layer Quasi-Recurrent Neural Network (QRNN) to an input sequence.

    Args:
        inputs: Tensor, input values in [Batch, Time, Channels] format.
        forget: Tensor, input values in [Batch, Time, Channels] format. Usually in the range 0-1.
        initial_state: Tensor, initial hidden state values in [Batch, Channels] format.
        name: A name for the operation (optional).
    Returns:
        Tensor: fo_pooled output, [Batch, Time, Channels] format.
    """
    with tf.name_scope(name or 'fo_pool'):
        inputs = tf.convert_to_tensor(inputs)
        forget = tf.convert_to_tensor(forget, dtype=inputs.dtype)

        if initial_state is None:
            inputs_shape = tf.shape(inputs)
            initial_state = tf.zeros((inputs_shape[0], inputs_shape[2]), dtype=inputs.dtype)
        else:
            initial_state = tf.convert_to_tensor(initial_state, dtype=inputs.dtype)

        return tfmiss_ops.miss_fo_pool(input=inputs, forget=forget, init=initial_state)[:, 1:]
