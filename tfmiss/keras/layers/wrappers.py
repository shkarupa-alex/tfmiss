import inspect

import tensorflow as tf
from keras.src import layers
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="Miss")
class WithRagged(layers.Wrapper):
    """Passes ragged tensor to layer that accepts only dense one.

    Arguments:
      layer: The `Layer` instance to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        super(WithRagged, self).__init__(layer, **kwargs)
        self.input_spec = layer.input_spec
        self.supports_masking = layer.supports_masking
        self._supports_ragged_inputs = True

        if not isinstance(layer, layers.Layer):
            raise ValueError(
                "Please initialize `WithRagged` layer with a "
                "`Layer` instance. You passed: {input}".format(input=layer)
            )

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)

        self.input_spec = self.layer.input_spec

        zero = "" if self.layer.dtype == tf.string else 0
        self.masking_layer = layers.Masking(mask_value=zero)

        super(WithRagged, self).build()

    def call(self, inputs, **kwargs):
        layer_kwargs = {}
        layer_call = inspect.unwrap(self.layer.call)
        layer_spec = inspect.getfullargspec(layer_call)
        for key in kwargs.keys():
            if key in layer_spec.args or key in layer_spec.kwonlyargs:
                layer_kwargs[key] = kwargs[key]

        if isinstance(inputs, tf.RaggedTensor):
            row_lengths = inputs.nested_row_lengths()
            inputs_dense = inputs.to_tensor()
        else:
            row_lengths = None
            inputs_dense = inputs

        inputs_dense = self.masking_layer(inputs_dense)
        outputs_dense = self.layer.call(inputs_dense, **layer_kwargs)

        if row_lengths is not None:
            outputs = tf.RaggedTensor.from_tensor(outputs_dense, row_lengths)
        else:
            outputs = outputs_dense

        return outputs

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def get_config(self):
        return super(WithRagged, self).get_config()


@register_keras_serializable(package="Miss")
class MapFlat(layers.Wrapper):
    """Calls layer on the flat values of ragged tensor.

    Arguments:
      layer: The `Layer` instance to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self._supports_ragged_inputs = True

    def build(self, input_shape=None):
        super(MapFlat, self).build([None])

    def call(self, inputs, **kwargs):
        return tf.ragged.map_flat_values(self.layer, inputs)

    def compute_output_shape(self, input_shape):
        return input_shape + self.layer.compute_output_shape((None,))[1:]
