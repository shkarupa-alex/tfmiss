import tensorflow as tf
from tfmiss.ops import tfmiss_ops


def connected_components(source, normalize=True, name=None):
    """Labels the connected components in a batch of images.

    Based on "A New Algorithm for Parallel Connected-Component Labelling on GPUs"
    https://ieeexplore.ieee.org/document/8274991

    Args:
        source: 4D images `Tensor` (integer, floating point and boolean types are supported).
        normalize: A boolean flag to enable labels reordering.
        name: A name for the operation (optional).
    Returns:
        `Tensor` with `int64` dtype and same shape as input.
    """
    with tf.name_scope(name or 'connected_components'):
        source = tf.convert_to_tensor(source, name='source')

        components = tfmiss_ops.miss_connected_components(input=source, normalize=normalize)

        return components
