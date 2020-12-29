from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import constraints, initializers, layers, regularizers, utils
from tensorflow.python.keras.utils import tf_utils


@utils.register_keras_serializable(package='Miss')
class AttentionWithContext(layers.Layer):
    """Attention layer, with a context/query vector, for temporal data.

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

        super(AttentionWithContext, self).__init__(**kwargs)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.channels = input_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of predictions should be defined. Found `None`.')

        self.representation = layers.Dense(
            self.channels,
            kernel_initializer=self.kernel_initializer,
            bias_initializer=self.bias_initializer,
            kernel_regularizer=self.kernel_regularizer,
            bias_regularizer=self.bias_regularizer,
            kernel_constraint=self.kernel_constraint,
            bias_constraint=self.bias_constraint
        )
        self.importance = layers.Dense(
            self.channels,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
        )

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # do not pass the mask to the next layers
        return None

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

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = super(AttentionWithContext, self).get_config()
        config.update({
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        })
        return config
