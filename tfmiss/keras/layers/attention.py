from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import backend, constraints, initializers, layers, regularizers
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='Miss')
class SelfAttentionWithContext(layers.Layer):
    """Self-attention layer for temporal data.

    Reference: https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf
    Hierarchical Attention Networks for Document Classification
    Yang et al.
    """

    def __init__(
            self, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None,
            bias_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
        self.input_spec = layers.InputSpec(ndim=3)
        self.supports_masking = True

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(SelfAttentionWithContext, self).__init__(**kwargs)

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        self.representation = layers.Dense(
            channels,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )
        self.importance = layers.Dense(
            channels,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
        )

        super(SelfAttentionWithContext, self).build(input_shape)

    def call(self, inputs, mask=None):
        uit = self.representation(inputs)
        uit = tf.tanh(uit)
        ait = self.importance(uit)
        a = tf.exp(ait)

        # Apply mask after the exp. Will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= tf.cast(mask, a.dtype)

        # In some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        epsilon = backend.epsilon()
        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + epsilon, a.dtype)

        weighted = inputs * a
        outputs = tf.reduce_sum(weighted, axis=1)

        return outputs

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super(SelfAttentionWithContext, self).get_config()
        config.update({
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })
        return config


@register_keras_serializable(package='Miss')
class MultiplicativeSelfAttention(layers.Attention):
    """Multiplicative (Luong) self-attention layer for temporal data."""

    def __init__(self, **kwargs):
        super(MultiplicativeSelfAttention, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        self.make_query = layers.Dense(channels, use_bias=False)
        self.att_bias = self.add_weight(
            'bias',
            shape=(1,),
            initializer='zeros',
            dtype=self.dtype,
            trainable=True)

        super(MultiplicativeSelfAttention, self).build([input_shape, input_shape, input_shape])

    def call(self, inputs, mask=None, training=None, return_attention_scores=False):
        if training is None:
            training = backend.learning_phase()

        query = self.make_query(inputs)
        value = inputs
        key = inputs

        outputs = super(MultiplicativeSelfAttention, self).call(
            [query, value, key],
            mask=None if mask is None else [mask, mask],
            training=training,
            return_attention_scores=return_attention_scores
        )

        return outputs

    def _calculate_scores(self, query, key):
        scores = tf.matmul(query, key, transpose_b=True)
        scores += self.att_bias  # This stage missed in parent implementation
        if self.scale is not None:
            scores *= self.scale
        return scores

    def compute_mask(self, inputs, mask=None):
        return super(MultiplicativeSelfAttention, self).compute_mask(
            [inputs, inputs, inputs],
            mask=None if mask is None else [mask, mask])

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


@register_keras_serializable(package='Miss')
class AdditiveSelfAttention(layers.AdditiveAttention):
    """Additive (Bahdanau) self-attention layer for temporal data."""

    def __init__(self, units, **kwargs):
        super(AdditiveSelfAttention, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        self.units = units

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        self.make_query = layers.Dense(self.units, use_bias=False)
        self.make_key = layers.Dense(self.units)
        self.make_score = layers.Dense(1, activation='sigmoid')

        super(AdditiveSelfAttention, self).build([input_shape, input_shape, input_shape])

    def call(self, inputs, mask=None, training=None, return_attention_scores=False):
        if training is None:
            training = backend.learning_phase()

        query = self.make_query(inputs)
        value = inputs
        key = self.make_key(inputs)

        outputs = super(AdditiveSelfAttention, self).call(
            [query, value, key],
            mask=None if mask is None else [mask, mask],
            training=training,
            return_attention_scores=return_attention_scores
        )

        return outputs

    def _calculate_scores(self, query, key):
        q_reshaped = tf.expand_dims(query, axis=-2)
        k_reshaped = tf.expand_dims(key, axis=-3)
        if self.use_scale:
            scale = self.scale
        else:
            scale = 1.

        scores = tf.tanh(q_reshaped + k_reshaped)
        scores = self.make_score(scores)  # This stage missed in parent implementation

        return tf.reduce_sum(scale * scores, axis=-1)

    def compute_mask(self, inputs, mask=None):
        return super(AdditiveSelfAttention, self).compute_mask(
            [inputs, inputs, inputs],
            mask=None if mask is None else [mask, mask])

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(AdditiveSelfAttention, self).get_config()
        config.update({'units': self.units})

        return config
