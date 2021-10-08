import math
import tensorflow as tf
from keras import layers
from keras.initializers.initializers_v2 import RandomUniform
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.conv_utils import normalize_tuple, normalize_padding
from keras.utils.tf_utils import shape_type_conversion
from tfmiss.nn import modulated_deformable_column


@register_keras_serializable(package='SegMe')
class DCNv2(layers.Layer):
    def __init__(self, filters, kernel_size, stride=(1, 1), padding='valid', dilation_rate=(1, 1), deformable_groups=1,
                 use_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=4)

        self.filters = filters
        self.kernel_size = normalize_tuple(kernel_size, 2, 'kernel_size')
        self.stride = normalize_tuple(stride, 2, 'stride')
        self.padding = padding
        self.dilation_rate = normalize_tuple(dilation_rate, 2, 'dilation_rate')
        self.deformable_groups = deformable_groups
        self.use_bias = use_bias

        if 'valid' == str(self.padding).lower():
            self._padding = (0, 0, 0, 0)
        elif 'same' == str(self.padding).lower():
            kh, kw = self.kernel_size
            self._padding = (kh // 2 + kh % 2 - 1, kh // 2, kw // 2 + kw % 2 - 1, kw // 2)
        else:
            raise ValueError('The `padding` argument must be one of "valid" or "same". Received: {}'.format(padding))

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1] if not self.extra_offset_mask else input_shape[0][-1]
        if channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')
        if channels < self.deformable_groups:
            raise ValueError('Number of deformable groups should be less or equals to channel dimension size')
        self.input_spec = layers.InputSpec(ndim=4, axes={-1: channels})

        kernel_shape = (self.kernel_size[0] * self.kernel_size[1] * channels, self.filters)
        kernel_stdv = 1.0 / math.sqrt(math.prod((channels,) + self.kernel_size))
        kernel_init = RandomUniform(-kernel_stdv, kernel_stdv)
        self.kernel = self.add_weight(
            name='kernel',
            shape=kernel_shape,
            initializer=kernel_init,
            trainable=True,
            dtype=self.dtype)
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(1, 1, 1, self.filters),
                initializer='zeros',
                trainable=True,
                dtype=self.dtype)

        self.offset_size = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        self.offset_mask = layers.Conv2D(
            self.offset_size * 3 // 2,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            kernel_initializer='zeros')

        self.sigmoid = layers.Activation('sigmoid')

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        offset_mask = self.offset_mask(inputs)

        offset, mask = offset_mask[..., : self.offset_size], offset_mask[..., self.offset_size:]
        mask = self.sigmoid(mask) * 2.  # (0.; 2.) with mean == 1.

        columns = modulated_deformable_column(
            inputs, offset, mask,
            kernel_size=self.kernel,
            strides=self.stride,
            padding=self._padding,
            dilation_rate=self.dilation,
            deformable_groups=self.deformable_groups)

        outputs = tf.matmul(columns, self.kernel)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        height = None
        if input_shape[1]:
            height = input_shape[1] + 2 * sum(self.pad[0]) - self.dilation[0] * (self.kernel[0] - 1) - 1
            height = int(height / self.stride[0] + 1)

        width = None
        if input_shape[2]:
            width = input_shape[2] + 2 * sum(self.pad[1]) - self.dilation[1] * (self.kernel[1] - 1) - 1
            width = int(width / self.stride[1] + 1)

        return input_shape[0], height, width, self.filters

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'stride': self.stride,
            'padding': self.padding,
            'dilation_rate': self.dilation_rate,
            'deformable_groups': self.deformable_groups,
            'use_bias': self.use_bias
        })

        return config
