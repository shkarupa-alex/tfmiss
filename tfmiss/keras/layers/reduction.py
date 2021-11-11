from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='Miss')
class Reduction(layers.Layer):
    """Performs an optionally-weighted reduction.
    This layer performs a reduction across one axis of its input data. This data may optionally be weighted by passing
    in an identical float tensor.
    Args:
      reduction: The type of reduction to perform. Can be one of the following: "max", "mean", "min", "prod", or "sum".
        This layer uses the Tensorflow reduce op which corresponds to that reduction (so, for "mean", we use
        "reduce_mean").
      axis: The axis to reduce along. Defaults to '-2', which is usually the axis that contains embeddings (but is not
        within the embedding itself).
    Input shape:
      A tensor of 2 or more dimensions of any numeric dtype.
    Output:
      A tensor of 1 less dimension than the input tensor, of the same dtype.
    Call arguments:
      inputs: The data to reduce.
      weights: An optional tensor or constant of the same shape as inputs that will weight the input data before it is
        reduced.
    """

    reduction_ops = {
        'max': tf.reduce_max,
        'mean': tf.reduce_mean,
        'min': tf.reduce_min,
        'prod': tf.reduce_prod,
        'sum': tf.reduce_sum
    }

    def __init__(self, reduction, axis=-2, **kwargs):
        super(Reduction, self).__init__(**kwargs)
        self._supports_ragged_inputs = True

        if 'sqrtn' != reduction and reduction not in self.reduction_ops:
            raise ValueError('Unknown reduction: {}'.format(reduction))

        self.reduction = reduction
        self.axis = axis

    def call(self, inputs, weights=None, **kwargs):
        if 'sqrtn' == self.reduction:
            input_sum = tf.reduce_sum(inputs, axis=self.axis)
            weights_sum = tf.reduce_sum(tf.ones_like(inputs), axis=self.axis)
            sqrt_weights_sum = tf.sqrt(weights_sum)

            return tf.divide(input_sum, sqrt_weights_sum)

        return self.reduction_ops[self.reduction](inputs, axis=self.axis)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if -1 == self.axis:
            return input_shape[:self.axis]

        return input_shape[:self.axis] + input_shape[self.axis + 1:]

    def get_config(self):
        config = super(Reduction, self).get_config()
        config.update({
            'reduction': self.reduction,
            'axis': self.axis,
        })

        return config
