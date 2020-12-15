from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.utils import control_flow_util, tf_utils
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
                 time_major=False,
                 **kwargs):
        super(QRNN, self).__init__(**kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)
        self.supports_masking = return_sequences

        self.units = units
        self.window = window
        self.zoneout = zoneout
        self.output_gate = output_gate
        self.activation = tf.keras.activations.get(activation)
        self.gate_activation = tf.keras.activations.get(gate_activation)

        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

        self.return_sequences = return_sequences
        self.return_state = return_state
        self.go_backwards = go_backwards
        self.time_major = time_major

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
            padding='causal',
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

    def call(self, inputs, training=None, initial_state=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        reverse_sim = [0] if self.time_major else [1]
        if self.go_backwards:
            inputs = tf.reverse(inputs, reverse_sim)

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
            hidden, forget, output = gate_values
        else:
            hidden, forget = gate_values

        hidden = self.act(hidden)
        forget = self.gate_act(forget)

        if self.zoneout > 0.:
            forget = control_flow_util.smart_cond(
                training,
                # multiply by (1. - self.zoneout) due to dropout scales preserved items
                lambda: 1. - self.drop(1. - forget) * (1. - self.zoneout),
                lambda: tf.identity(forget)
            )

        c = fo_pool(hidden, forget, initial_state=initial_state, time_major=self.time_major)
        h = self.gate_act(output) * c if self.output_gate else c
        # TODO: https://github.com/JonathanRaiman/tensorflow_qrnn/blob/master/qrnn.py#L161
        # h = gate_activation_fn(c) if output_gate else c

        if not self.return_sequences:
            h = h[:, -1, :] if not self.time_major else h[-1, :, :]
        elif self.go_backwards:
            h = tf.reverse(h, reverse_sim)

        if self.return_state:
            last_state = c[:, -1, :] if not self.time_major else c[-1, :, :]

            return h, last_state

        return h

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        h_shape = input_shape[:-1] + (self.units,)
        c_shape = (h_shape[0], h_shape[2]) if not self.time_major else (h_shape[1], h_shape[2])

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
            'activation': tf.keras.activations.serialize(self.activation),
            'gate_activation': tf.keras.activations.serialize(self.gate_activation),
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'return_sequences': self.return_sequences,
            'return_state': self.return_state,
            'go_backwards': self.go_backwards,
            'time_major': self.time_major,
        })

        return config
