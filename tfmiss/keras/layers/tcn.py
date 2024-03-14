from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tf_keras import activations, backend, constraints, initializers, layers, models, regularizers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion
from tfmiss.keras.layers.wrappers import WeightNorm


@register_keras_serializable(package='Miss')
class TemporalBlock(layers.Layer):
    """Residual block for Temporal Convolutional Network.
    Reference: https://arxiv.org/abs/1803.01271
    An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
    Shaojie Bai, J. Zico Kolter, Vladlen Koltun (2018)
    """

    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
    # Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    _STRIDES = 1

    def __init__(self,
                 filters,
                 kernel_size,
                 dilation,
                 dropout,
                 padding='causal',
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(TemporalBlock, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)
        self.supports_masking = True

        if padding not in {'causal', 'same'}:
            raise ValueError('Only "causal" and "same" padding are compatible with this layer.')

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.padding = padding

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.conv1d1 = None
        self.conv1d2 = None
        self.dropout1 = None
        self.dropout2 = None
        self.downsample = None
        self.add = None
        self.act = None

    @shape_type_conversion
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        num_channels = input_shape[-1]
        if num_channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = layers.InputSpec(ndim=3, axes={-1: num_channels})

        self.conv1d1 = WeightNorm(layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self._STRIDES,
            padding=self.padding,
            dilation_rate=self.dilation,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        ))
        self.conv1d2 = WeightNorm(layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self._STRIDES,
            padding=self.padding,
            dilation_rate=self.dilation,
            activation=self.activation,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint,
        ))

        self.dropout1 = layers.SpatialDropout1D(rate=self.dropout)
        self.dropout2 = layers.SpatialDropout1D(rate=self.dropout)

        if num_channels != self.filters:
            self.downsample = layers.Conv1D(
                self.filters,
                kernel_size=1,
                padding='valid',
                activation=None,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
            )

        self.add = layers.Add()
        self.act = layers.Activation(activation=self.activation)

        super(TemporalBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()

        out = self.conv1d1(inputs)
        out = self.dropout1(out, training=training)

        out = self.conv1d2(out)
        out = self.dropout2(out, training=training)

        res = inputs if self.downsample is None else self.downsample(inputs)

        out = self.add([out, res])
        out = self.act(out)

        return out

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = super(TemporalBlock, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation': self.dilation,
            'dropout': self.dropout,
            'padding': self.padding,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })

        return config


@register_keras_serializable(package='Miss')
class TemporalConvNet(layers.Layer):
    """Temporal Convolutional Network layer.
    Reference: https://arxiv.org/abs/1803.01271
    An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
    Shaojie Bai, J. Zico Kolter, Vladlen Koltun (2018)
    """

    def __init__(self,
                 filters,
                 kernel_size=2,
                 dropout=0.2,
                 padding='causal',
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(TemporalConvNet, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)
        self.supports_masking = True

        if not isinstance(filters, (list, tuple)) or not len(filters):
            raise ValueError('Number of residual layers could not be zero.')

        if padding not in {'causal', 'same'}:
            raise ValueError('Only "causal" and "same" padding are compatible with this layer.')

        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.padding = padding

        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    @shape_type_conversion
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        num_channels = input_shape[-1]
        if num_channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = layers.InputSpec(ndim=3, axes={-1: num_channels})

        self.blocks = models.Sequential()
        for i in range(len(self.filters)):
            self.blocks.add(TemporalBlock(
                filters=self.filters[i],
                kernel_size=self.kernel_size,
                dilation=2 ** i,
                dropout=self.dropout,
                padding=self.padding,
                activation=self.activation,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
                dtype=self.dtype,
            ))

        super(TemporalConvNet, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = backend.learning_phase()

        return self.blocks(inputs, training=training)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        return input_shape[:-1] + (self.filters[-1],)

    def get_config(self):
        config = super(TemporalConvNet, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'padding': self.padding,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
        })

        return config
