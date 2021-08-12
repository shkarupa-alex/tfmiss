import tensorflow as tf
from keras.layers import Dropout, InputSpec
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='Miss')
class TimestepDropout(Dropout):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        noise_shape = (input_shape[0], input_shape[1], 1)

        return noise_shape
