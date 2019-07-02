from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.python.keras import backend as K


class PositionalEncoding(keras.layers.Layer):
    """Positional encoding layer.

    Calculates the position encodings as a mix of sine and cosine functions with
    geometrically increasing wavelengths. Each channel of the input Tensor is
    incremented by exactly one of these sinusoids.

    Defined and formulized in "Attention is All You Need", section 3.5.
    """

    def __init__(self, max_length, *args, **kwargs):
        """Initialize PositionalEncoding.
        Args:
          max_length: int, length of precomputed timing signal sequence
        """
        super(PositionalEncoding, self).__init__(*args, **kwargs)
        self.input_spec = keras.layers.InputSpec(ndim=3)

        self.max_length = max_length
        self.encodings = None

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        # Positional encodings have the same channel dimension as the embeddings
        num_channels = input_shape[-1]
        if num_channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = keras.layers.InputSpec(ndim=3, axes={-1: num_channels})

        # Compute the positional encodings once
        positions = K.math_ops.range(self.max_length, dtype='float32')
        positions = K.expand_dims(positions, axis=-1)

        channels = K.math_ops.range(num_channels, dtype='float32')
        channels = K.expand_dims(channels, axis=0)

        angles = positions / K.math_ops.pow(10000, 2 * (channels // 2) / num_channels)

        sines = K.math_ops.sin(angles[:, 0::2])  # Apply sin to even index in the array
        cosines = K.math_ops.cos(angles[:, 1::2])  # Apply cos to odd index in the array

        encodings = K.array_ops.concat([sines, cosines], axis=-1)
        encodings = K.expand_dims(encodings, axis=0)
        self.encodings = K.cast(encodings, K.floatx())

        super(PositionalEncoding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        length = K.shape(inputs)[1]

        length_check = K.control_flow_ops.Assert(
            K.less_equal(length, self.max_length),
            ['Inputs length should be less then reserved `max_length` '
             '({}) for positional encodings'.format(self.max_length),
             length]
        )
        positions = K.control_flow_ops.with_dependencies(
            [length_check], self.encodings[:, :length, :])

        return inputs + positions

    def get_config(self):
        config = {'max_length': self.max_length}
        base_config = super(PositionalEncoding, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
