from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfmiss.keras.layers.wrappers import WeightNorm
from tensorflow.python.keras.backend import convert_inputs_if_ragged, maybe_convert_to_ragged
from tensorflow.python.keras.utils import tf_utils


@tf.keras.utils.register_keras_serializable(package='Miss')
class TemporalBlock(tf.keras.layers.Layer):
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
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(TemporalBlock, self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

        if padding not in {'causal', 'same'}:
            raise ValueError('Only "causal" and "same" padding are compatible with this layer.')

        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = dropout
        self.padding = padding

        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.conv1d1 = None
        self.conv1d2 = None
        self.dropout1 = None
        self.dropout2 = None
        self.downsample = None
        self.add = None
        self.act = None

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        num_channels = input_shape[-1]
        if num_channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={-1: num_channels})

        self.conv1d1 = WeightNorm(tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self._STRIDES,
            padding=self.padding,
            data_format='channels_last',
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
        self.conv1d2 = WeightNorm(tf.keras.layers.Conv1D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            strides=self._STRIDES,
            padding=self.padding,
            data_format='channels_last',
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

        self.dropout1 = tf.keras.layers.SpatialDropout1D(rate=self.dropout)
        self.dropout2 = tf.keras.layers.SpatialDropout1D(rate=self.dropout)

        if num_channels != self.filters:
            self.downsample = tf.keras.layers.Conv1D(
                self.filters,
                kernel_size=1,
                padding='valid',
                data_format='channels_last',
                activation=None,
                use_bias=self.use_bias,
                kernel_initializer=self.kernel_initializer,
                bias_initializer=self.bias_initializer,
                kernel_regularizer=self.kernel_regularizer,
                bias_regularizer=self.bias_regularizer,
                kernel_constraint=self.kernel_constraint,
                bias_constraint=self.bias_constraint,
            )

        self.add = tf.keras.layers.Add()
        self.act = tf.keras.layers.Activation(activation=self.activation)

        super(TemporalBlock, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        out = self.conv1d1(inputs)
        out = self.dropout1(out, training=training)

        out = self.conv1d2(out)
        out = self.dropout2(out, training=training)

        res = inputs if self.downsample is None else self.downsample(inputs)

        out = self.add([out, res])
        out = self.act(out)

        return out

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.filters,)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dilation': self.dilation,
            'dropout': self.dropout,
            'padding': self.padding,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(TemporalBlock, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='Miss')
class TemporalConvNet(tf.keras.layers.Layer):
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
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(TemporalConvNet, self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)
        self._supports_ragged_inputs = True

        if not isinstance(filters, (list, tuple)) or not len(filters):
            raise ValueError('Number of residual layers could not be zero.')

        if padding not in {'causal', 'same'}:
            raise ValueError('Only "causal" and "same" padding are compatible with this layer.')

        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.padding = padding

        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        num_channels = input_shape[-1]
        if num_channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={-1: num_channels})

        self.layers = []
        num_levels = len(self.filters)
        for i in range(num_levels):
            temporal_block = TemporalBlock(
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
            )
            self.layers.append(temporal_block)
            setattr(self, 'temporal_block_{}'.format(i), temporal_block)

        super(TemporalConvNet, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        inputs, row_lengths = convert_inputs_if_ragged(inputs)
        is_ragged_input = (row_lengths is not None)

        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs, training=training)

        outputs = maybe_convert_to_ragged(is_ragged_input, outputs, row_lengths)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        return input_shape[:-1] + (self.filters[-1],)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
            'padding': self.padding,
            'activation': tf.keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(TemporalConvNet, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
