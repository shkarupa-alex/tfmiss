from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import activations, backend, constraints, initializers, layers, regularizers
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion
from tfmiss.nn import fo_pool


@register_keras_serializable(package='Miss')
class QRNN(layers.Layer):
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
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 **kwargs):
        super(QRNN, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)
        self.supports_masking = return_sequences

        self.units = units
        self.window = window
        self.zoneout = zoneout
        self.output_gate = output_gate
        self.activation = activations.get(activation)
        self.gate_activation = activations.get(gate_activation)

        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards

    @shape_type_conversion
    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        num_channels = input_shape[-1]
        if num_channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = layers.InputSpec(ndim=3, axes={-1: num_channels})

        conv1d_channels = self.units * (3 if self.output_gate else 2)
        self.conv1d = layers.Conv1D(
            filters=conv1d_channels,
            kernel_size=self.window,
            padding='causal',
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint)

        if self.zoneout > 0.:
            self.drop = layers.Dropout(self.zoneout)

        self.act = layers.Activation(activation=self.activation)
        self.gate_act = layers.Activation(activation=self.gate_activation)

        super(QRNN, self).build(input_shape)

    def call(self, inputs, training=None, initial_state=None):
        if training is None:
            training = backend.learning_phase()

        if self.go_backwards:
            inputs = tf.reverse(inputs, [1])

        gate_values = self.conv1d(inputs)
        gate_values = tf.split(gate_values, 3 if self.output_gate else 2, axis=-1)
        if self.output_gate:
            z, f, o = gate_values
        else:
            z, f = gate_values

        z = self.act(z)
        f = self.gate_act(f)

        if self.zoneout > 0.:
            f = smart_cond(
                training,
                # multiply by (1. - self.zoneout) due to dropout scales preserved items
                lambda: self.drop(f) * (1. - self.zoneout),
                lambda: f * (1. - self.zoneout)
            )

        c = fo_pool(z, f, initial_state=initial_state)
        h = self.gate_act(o) * c if self.output_gate else c

        if not self.return_sequences:
            h = h[:, -1, :]
        elif self.go_backwards:
            h = tf.reverse(h, [1])

        if self.return_state:
            last_state = c[:, -1, :]

            return h, last_state

        return h

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        h_shape = input_shape[:-1] + (self.units,)
        c_shape = (h_shape[0], h_shape[2])

        if not self.return_sequences:
            h_shape = c_shape

        if self.return_state:
            return h_shape, c_shape
        else:
            return h_shape

    def compute_mask(self, inputs, mask=None):
        if not self.return_sequences:
            return None

        return mask

    def get_config(self):
        config = super(QRNN, self).get_config()
        config.update({
            'units': self.units,
            'window': self.window,
            'zoneout': self.zoneout,
            'output_gate': self.output_gate,
            'activation': activations.serialize(self.activation),
            'gate_activation': activations.serialize(self.gate_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'go_backwards': self.go_backwards
        })

        return config
