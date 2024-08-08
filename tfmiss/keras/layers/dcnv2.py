import numpy as np
import tensorflow as tf
from keras.src import initializers
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable
from keras.src.utils.argument_validation import standardize_tuple

from tfmiss.nn import modulated_deformable_column


@register_keras_serializable(package="SegMe")
class DCNv2(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        dilation_rate=(1, 1),
        deformable_groups=1,
        use_bias=True,
        custom_alignment=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.input_spec = InputSpec(ndim=4)  # inputs
        if custom_alignment:
            self.input_spec = [
                InputSpec(ndim=4),  # inputs
                InputSpec(ndim=4),  # alignments
            ]

        self.filters = filters
        self.kernel_size = standardize_tuple(kernel_size, 2, "kernel_size")
        self.strides = standardize_tuple(strides, 2, "strides")
        self.padding = padding
        self.dilation_rate = standardize_tuple(
            dilation_rate, 2, "dilation_rate"
        )
        self.deformable_groups = deformable_groups
        self.use_bias = use_bias
        self.custom_alignment = custom_alignment

        if "valid" == str(self.padding).lower():
            self._padding = (0, 0, 0, 0)
        elif "same" == str(self.padding).lower():
            pad_h = self.dilation_rate[0] * (self.kernel_size[0] - 1)
            pad_w = self.dilation_rate[1] * (self.kernel_size[1] - 1)
            self._padding = (
                pad_h // 2,
                pad_h - pad_h // 2,
                pad_w // 2,
                pad_w - pad_w // 2,
            )
        else:
            raise ValueError(
                f"The `padding` argument must be one of `valid` or `same`. "
                f"Received: {padding}"
            )

    def build(self, input_shape):
        channels = input_shape[-1]
        if self.custom_alignment:
            channels = input_shape[0][-1]

        if channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )
        if channels < self.deformable_groups:
            raise ValueError(
                "Number of deformable groups should be less or "
                "equals to channel dimension size"
            )

        kernel_shape = (
            self.kernel_size[0] * self.kernel_size[1] * channels,
            self.filters,
        )
        kernel_stdv = 1.0 / np.sqrt(np.prod((channels,) + self.kernel_size))
        kernel_init = initializers.RandomUniform(-kernel_stdv, kernel_stdv)
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=kernel_init,
            trainable=True,
            dtype=self.dtype,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
                dtype=self.dtype,
            )

        self.offset_size = (
            self.deformable_groups
            * 2
            * self.kernel_size[0]
            * self.kernel_size[1]
        )
        self.offset_mask = layers.Conv2D(
            self.offset_size * 3 // 2,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            kernel_initializer="zeros",
            dtype=self.dtype_policy,
        )
        if self.custom_alignment:
            self.offset_mask.build(input_shape[1])
        else:
            self.offset_mask.build(input_shape)

        self.sigmoid = layers.Activation("sigmoid", dtype=self.dtype_policy)

        super().build(input_shape)

    def call(self, inputs, **kwargs):
        alignments = inputs
        if self.custom_alignment:
            inputs, alignments = inputs
            alignments = tf.cast(alignments, inputs.dtype)

        offset_mask = self.offset_mask(alignments)

        offset, mask = (
            offset_mask[..., : self.offset_size],
            offset_mask[..., self.offset_size :],
        )
        mask = self.sigmoid(mask) * 2.0  # (0.; 2.) with mean == 1.

        columns = modulated_deformable_column(
            inputs,
            offset,
            mask,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self._padding,
            dilation_rate=self.dilation_rate,
            deformable_groups=self.deformable_groups,
        )

        outputs = tf.matmul(columns, self.kernel)
        out_shape = tf.concat(
            [tf.shape(offset_mask)[:-1], [self.filters]], axis=-1
        )

        outputs = tf.reshape(outputs, out_shape)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        if not tf.executing_eagerly():
            # Infer the static output shape
            if self.custom_alignment:
                source_shape = [inputs.shape, alignments.shape]
            else:
                source_shape = inputs.shape
            out_shape = self.compute_output_shape(source_shape)
            outputs.set_shape(out_shape)

        return outputs

    def compute_output_shape(self, input_shape):
        source_shape = input_shape
        if self.custom_alignment:
            source_shape = input_shape[1]

        offset_mask_shape = self.offset_mask.compute_output_shape(source_shape)

        return offset_mask_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "deformable_groups": self.deformable_groups,
                "use_bias": self.use_bias,
                "custom_alignment": self.custom_alignment,
            }
        )

        return config
