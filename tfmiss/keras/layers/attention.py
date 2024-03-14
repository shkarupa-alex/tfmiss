from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_keras import backend, constraints, initializers, layers, regularizers
from tf_keras.saving import register_keras_serializable
from tf_keras.src.utils.tf_utils import shape_type_conversion


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
            activation='tanh',
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

    def call(self, inputs, mask=None, *args, **kwargs):
        uit = self.representation(inputs)

        ait = self.importance(uit)
        if mask is not None:
            ait -= 100 * tf.cast(mask, ait.dtype)

        att = tf.nn.softmax(ait, axis=1)

        weighted = inputs * att
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

    def __init__(self, units=None, **kwargs):
        super(MultiplicativeSelfAttention, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        self.units = units

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        units = channels if self.units is None else self.units
        self.proj = layers.Dense(units * 3, use_bias=False)

        proj_shape = input_shape[:-1] + (units,)
        super(MultiplicativeSelfAttention, self).build([proj_shape, proj_shape, proj_shape])

    def call(self, inputs, mask=None, training=None, return_attention_scores=False, use_causal_mask=False):
        if training is None:
            training = backend.learning_phase()

        query, key, value = tf.split(self.proj(inputs), 3, axis=-1)
        outputs = super(MultiplicativeSelfAttention, self).call(
            [query, value, key],
            mask=None if mask is None else [mask, mask],
            training=training,
            return_attention_scores=return_attention_scores,
            use_causal_mask=use_causal_mask
        )

        return outputs

    def compute_mask(self, inputs, mask=None):
        return super(MultiplicativeSelfAttention, self).compute_mask(
            [inputs, inputs, inputs],
            mask=None if mask is None else [mask, mask])

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        units = input_shape[-1] if self.units is None else self.units

        return input_shape[:-1] + (units,)

    def get_config(self):
        config = super(MultiplicativeSelfAttention, self).get_config()
        config.update({'units': self.units})

        return config


@register_keras_serializable(package='Miss')
class AdditiveSelfAttention(layers.AdditiveAttention):
    """Additive (Bahdanau) self-attention layer for temporal data."""

    def __init__(self, units=None, **kwargs):
        super(AdditiveSelfAttention, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

        self.units = units

    @shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        units = channels if self.units is None else self.units
        self.proj = layers.Dense(units * 3, use_bias=False)

        proj_shape = input_shape[:-1] + (units,)
        super(AdditiveSelfAttention, self).build([proj_shape, proj_shape, proj_shape])

    def call(self, inputs, mask=None, training=None, return_attention_scores=False):
        if training is None:
            training = backend.learning_phase()

        query, key, value = tf.split(self.proj(inputs), 3, axis=-1)
        outputs = super(AdditiveSelfAttention, self).call(
            [query, value, key],
            mask=None if mask is None else [mask, mask],
            training=training,
            return_attention_scores=return_attention_scores
        )

        return outputs

    def compute_mask(self, inputs, mask=None):
        return super(AdditiveSelfAttention, self).compute_mask(
            [inputs, inputs, inputs],
            mask=None if mask is None else [mask, mask])

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        units = input_shape[-1] if self.units is None else self.units

        return input_shape[:-1] + (units,)

    def get_config(self):
        config = super(AdditiveSelfAttention, self).get_config()
        config.update({'units': self.units})

        return config
