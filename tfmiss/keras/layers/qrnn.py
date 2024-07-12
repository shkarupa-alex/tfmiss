import tensorflow as tf
from keras import activations, backend, constraints, initializers, layers, regularizers
from keras.saving import register_keras_serializable
from keras.src.utils.control_flow_util import smart_cond
from keras.src.utils.tf_utils import shape_type_conversion
from tfmiss.nn import fo_pool


@register_keras_serializable(package='Miss')
class QRNN(layers.Layer):
    """Residual block for Temporal Convolutional Network.
    Reference: https://arxiv.org/abs/1803.01271
    An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
    Shaojie Bai, J. Zico Kolter, Vladlen Koltun (2018)
    """

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
                 zero_output_for_mask=False,
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

        self.zero_output_for_mask = zero_output_for_mask

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

        self.initial_state = self.add_weight(
            name='initial_state',
            shape=(1, self.units),
            initializer='zeros',
            trainable=True,
            dtype=self.dtype)

        self.act = layers.Activation(activation=self.activation)
        self.gate_act = layers.Activation(activation=self.gate_activation)

        super(QRNN, self).build(input_shape)

    def call(self, inputs, training=None, mask=None, initial_state=None):
        shape = tf.shape(inputs)

        if self.go_backwards:
            inputs = tf.reverse(inputs, [1])

        if training is None:
            training = backend.learning_phase()

        if mask is not None:
            mask = mask[..., None]
            if self.go_backwards:
                mask = tf.reverse(mask, [1])

        if initial_state is None:
            initial_state = tf.repeat(self.initial_state, shape[0], axis=0)

        gates = self.conv1d(inputs)
        gates = tf.split(gates, 3 if self.output_gate else 2, axis=-1)

        if self.output_gate:
            sequence, forget, output = gates
            output = self.gate_act(output)
        else:
            sequence, forget = gates
            output = None

        sequence = self.act(sequence)
        forget = self.gate_act(forget)

        if self.zoneout > 0.:
            forget = smart_cond(
                training,
                lambda: self.drop(forget) * (1. - self.zoneout),  # tf.nn.dropout upscales preserved values, revert
                lambda: tf.identity(forget))

        if mask is not None:
            forget = tf.where(mask, forget, 0.)

        recurrent = fo_pool(sequence, forget, initial_state=initial_state)
        hidden = recurrent if not self.output_gate else recurrent * output

        if self.return_sequences:
            if mask is not None and self.zero_output_for_mask:
                hidden = tf.where(mask, hidden, 0.)

            if self.go_backwards:
                hidden = tf.reverse(hidden, [1])
        else:
            if mask is None or not self.output_gate:
                hidden = hidden[:, -1, :]
            else:
                last_idx = shape[1] - 1 - tf.argmax(tf.reverse(mask, [1]), axis=1, output_type='int32')
                hidden = tf.gather(hidden, last_idx[..., 0], batch_dims=1)

        if self.return_state:
            return hidden, recurrent[:, -1, :]

        return hidden

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        hidden_shape = input_shape[:-1] + (self.units,)
        state_shape = (hidden_shape[0], hidden_shape[2])

        if not self.return_sequences:
            hidden_shape = state_shape

        if self.return_state:
            return hidden_shape, state_shape

        return hidden_shape

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
            'go_backwards': self.go_backwards,
            'zero_output_for_mask': self.zero_output_for_mask
        })

        return config
