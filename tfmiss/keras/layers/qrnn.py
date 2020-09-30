from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tfmiss.nn import fo_pool


@tf.keras.utils.register_keras_serializable(package='Miss')
class QRNN(tf.keras.layers.Layer):
    """Residual block for Temporal Convolutional Network.
    Reference: https://arxiv.org/abs/1803.01271
    An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
    Shaojie Bai, J. Zico Kolter, Vladlen Koltun (2018)
    """

    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
    # Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    _STRIDES = 1

    def __init__(self,
                 units,
                 window,
                 zoneout=0.0,
                 output_gate=True,
                 activation='tanh',
                 gate_activation='sigmoid',
                 time_major=False,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(QRNN, self).__init__(**kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)
        self.supports_masking = True

        self.units = units
        self.window = window
        self.zoneout = zoneout
        self.output_gate = output_gate
        self.activation = tf.keras.activations.get(activation)
        self.gate_activation = tf.keras.activations.get(gate_activation)
        self.time_major = time_major

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

        conv1d_channels = self.units * (3 if self.output_gate else 2)
        self.conv1d = tf.keras.layers.Conv1D(
            filters=conv1d_channels,
            kernel_size=self.window,
            padding='same',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)

        if self.zoneout > 0.:
            self.drop = tf.keras.layers.Dropout(self.zoneout)

        self.act = tf.keras.layers.Activation(activation=self.activation)
        self.gate_act = tf.keras.layers.Activation(activation=self.gate_activation)

        super(QRNN, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        inputs_batch_major = inputs
        if self.time_major:
            # go to batch_major for convolution if needed
            inputs_batch_major = tf.transpose(inputs, (1, 0, 2), name='to_batch_major')

        gate_values = self.conv1d(inputs_batch_major)
        if self.time_major:
            # return to time_major if needed
            gate_values = tf.transpose(gate_values, (1, 0, 2), name='to_time_major')

        gate_values = tf.split(gate_values, 3 if self.output_gate else 2, axis=-1)
        if self.output_gate:
            x, forget, output = gate_values
        else:
            x, forget = gate_values

        x = self.act(x)
        forget = self.gate_act(forget)

        if self.zoneout > 0.:
            forget = tf_utils.smart_cond(
                training,
                # multiply by (1. - self.zoneout) due to dropout scales preserved items
                lambda: 1. - self.drop(1. - forget) * (1. - self.zoneout),
                lambda: tf.identity(forget)
            )

        c = fo_pool(x, forget, time_major=self.time_major)
        h = self.gate_act(output) * c if self.output_gate else c

        return h

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.units,)

    def get_config(self):
        config = super(QRNN, self).get_config()
        config.update({
            'units': self.units,
            'window': self.window,
            'zoneout': self.zoneout,
            'output_gate': self.output_gate,
            'activation': tf.keras.activations.serialize(self.activation),
            'gate_activation': tf.keras.activations.serialize(self.gate_activation),
            'time_major': self.time_major,
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
        })

        return config
