from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package='Miss')
class L2Scale(tf.keras.layers.Layer):
    """L2-constrained and scaled layer.
    Reference: https://arxiv.org/pdf/1703.09507.pdf
    L2-constrained Softmax Loss for Discriminative Face Verification
    Rajeev Ranjan, Carlos D. Castillo and Rama Chellappa (2017)

    Notes
        As mentioned in paper, theoretically good `alpha` can be estimated as `log(p * (C - 2) / (1 - p))` where `p`
        is the average softmax probability p for correctly classifying a feature and C is a number of classes.
        Usually good `alpha` will be in range [10; 30].
    """

    def __init__(self, alpha=20., **kwargs):
        super(L2Scale, self).__init__(**kwargs)
        self.supports_masking = True
        self._supports_ragged_inputs = True

        self.alpha = alpha

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tf.RaggedTensor):
            normalized = tf.ragged.map_flat_values(tf.math.l2_normalize, inputs, axis=-1)
        else:
            normalized = tf.math.l2_normalize(inputs, axis=-1)

        alpha = tf.cast(self.alpha, self.dtype)

        return normalized * alpha

    def get_config(self):
        config = {'alpha': self.alpha}
        base_config = super(L2Scale, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
