from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.ops import tfmiss_ops


def euclidean_distance(source, dtype=tf.float32, name=None):
    """Applies euclidean distance transform to the images.

    Based on "Distance Transforms of Sampled Functions"
    http://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf

    Args:
        source: 4D images `Tensor` (integer, floating point and boolean types are supported).
        dtype: The DType of the resulting `Tensor`.
        name: A name for the operation (optional).
    Returns:
        `Tensor` with the same shape as input.
    """
    with tf.name_scope(name or 'euclidean_distance'):
        source = tf.convert_to_tensor(source, name='source')
        dtype = tf.dtypes.as_dtype(dtype)

        return tfmiss_ops.miss_euclidean_distance(input=source, dtype=dtype)
