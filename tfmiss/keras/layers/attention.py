from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import constraints, initializers, layers, regularizers, utils
from tensorflow.python.keras.utils import tf_utils


@utils.register_keras_serializable(package='Miss')
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

    @tf_utils.shape_type_conversion
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
        epsilon = tf.keras.backend.epsilon()
        a /= tf.cast(tf.reduce_sum(a, axis=1, keepdims=True) + epsilon, a.dtype)

        weighted = inputs * a
        outputs = tf.reduce_sum(weighted, axis=1)

        return outputs

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

    @tf_utils.shape_type_conversion
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


@utils.register_keras_serializable(package='Miss')
class MultiplicativeSelfAttention(layers.Attention):
    """Multiplicative (Luong) self-attention layer for temporal data.

    This version of Self-Attention refers to Transformers
    """

    def __init__(self, **kwargs):
        super(MultiplicativeSelfAttention, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        self.make_query = layers.Dense(channels, use_bias=False)
        self.make_value = layers.Dense(channels, use_bias=False)
        self.make_key = layers.Dense(channels, use_bias=False)

        super(MultiplicativeSelfAttention, self).build([input_shape, input_shape, input_shape])

    def call(self, inputs, mask=None, training=None, return_attention_scores=False):
        if training is None:
            training = tf.keras.backend.learning_phase()

        query = self.make_query(inputs)
        value = self.make_value(inputs)
        key = self.make_key(inputs)

        floatx = tf.keras.backend.floatx()  # Hack for wrong cast in _apply_scores
        tf.keras.backend.set_floatx(self.compute_dtype)

        outputs = super(MultiplicativeSelfAttention, self).call(
            [query, value, key],
            mask=None if mask is None else [mask, mask],
            training=training,
            return_attention_scores=return_attention_scores
        )

        tf.keras.backend.set_floatx(floatx)

        return outputs

    def compute_mask(self, inputs, mask=None):
        return super(MultiplicativeSelfAttention, self).compute_mask(
            [inputs, inputs, inputs],
            mask=None if mask is None else [mask, mask])

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape


@utils.register_keras_serializable(package='Miss')
class AdditiveSelfAttention(layers.AdditiveAttention):
    """Additive (Bahdanau) self-attention layer for temporal data.

    Reference: https://arxiv.org/pdf/1806.01264.pdf
    """

    def __init__(self, **kwargs):
        super(AdditiveSelfAttention, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(ndim=3)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        channels = input_shape[-1]
        if channels is None:
            raise ValueError('Channel dimension of inputs should be defined. Found `None`.')
        self.input_spec = layers.InputSpec(ndim=3, axes={-1: channels})

        self.make_query = layers.Dense(channels, use_bias=False)
        self.make_key = layers.Dense(channels)
        self.make_score = layers.Dense(1, activation='sigmoid')

        super(AdditiveSelfAttention, self).build([input_shape, input_shape, input_shape])

    def call(self, inputs, mask=None, training=None, return_attention_scores=False):
        if training is None:
            training = tf.keras.backend.learning_phase()

        query = self.make_query(inputs)
        value = inputs
        key = self.make_key(inputs)

        floatx = tf.keras.backend.floatx()  # Hack for wrong cast in _apply_scores
        tf.keras.backend.set_floatx(self.compute_dtype)

        outputs = super(AdditiveSelfAttention, self).call(
            [query, value, key],
            mask=None if mask is None else [mask, mask],
            training=training,
            return_attention_scores=return_attention_scores
        )

        tf.keras.backend.set_floatx(floatx)

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

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
