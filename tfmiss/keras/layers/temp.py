from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.layers.preprocessing.reduction import Reduction as _Reduction


@tf.keras.utils.register_keras_serializable(package='Miss')
class Reduction(_Reduction):
    def __init__(self, *args, **kwargs):
        super(Reduction, self).__init__(*args, **kwargs)
        self._supports_ragged_inputs = True

    @tf_utils.shape_type_conversion
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
