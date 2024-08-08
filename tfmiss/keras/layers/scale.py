import tensorflow as tf
from keras.src import layers
from keras.src.layers.input_spec import InputSpec
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="Miss")
class L2Scale(layers.Layer):
    """L2-constrained and scaled layer.
    Reference: https://arxiv.org/pdf/1703.09507.pdf
    L2-constrained Softmax Loss for Discriminative Face Verification
    Rajeev Ranjan, Carlos D. Castillo and Rama Chellappa (2017)

    Notes
        As mentioned in paper, theoretically good `alpha` can be estimated as
          `log(p * (C - 2) / (1 - p))` where `p` is the average softmax
          probability p for correctly classifying a feature and C is a number
          of classes.
        Usually good `alpha` will be in range [10; 30].
    """

    def __init__(self, alpha=20.0, **kwargs):
        super(L2Scale, self).__init__(**kwargs)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

        self.alpha = alpha

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError("Shape {} must have rank >= 2".format(input_shape))

        num_channels = input_shape[-1]
        if num_channels is None:
            raise ValueError(
                "Channel dimension of the inputs should be defined. "
                "Found `None`."
            )

        self.input_spec = InputSpec(
            ndim=len(input_shape), axes={-1: num_channels}
        )

        super(L2Scale, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return tf.math.l2_normalize(inputs, axis=-1) * self.alpha

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(L2Scale, self).get_config()
        config.update({"alpha": self.alpha})

        return config
