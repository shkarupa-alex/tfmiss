# Inspired by https://github.com/tensorflow/text/blob/master/tensorflow_text/python/keras/layers/todense.py
import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import register_keras_serializable


@register_keras_serializable(package='Miss')
class ToDense(layers.Layer):
    """ Layer that makes padding and masking a Composite Tensors effortless. The layer takes a RaggedTensor or
    a SparseTensor and converts it to a uniform tensor by right-padding it or filling in missing values.

    Arguments:
      pad_value: A value used to pad and fill in the missing values. Should be a meaningless value for the input data.
      mask: A Boolean value representing whether to mask the padded values. If true, no any downstream Masking layer
        or Embedding layer with mask_zero=True should be added. Default is 'False'.

    Input shape: Any Ragged or Sparse Tensor is accepted, but it requires the type of input to be specified via the
        Input or InputLayer from the Keras API.

    Output shape: The output is a uniform tensor having the same shape, in case of a ragged input or the same dense
        shape, in case of a sparse input.
    """

    def __init__(self, pad_value, mask=False, **kwargs):
        super(ToDense, self).__init__(**kwargs)
        self.input_spec = layers.InputSpec(min_ndim=2)
        self._supports_ragged_inputs = True
        self.pad_value = pad_value
        self.mask = mask

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tf.RaggedTensor):
            outputs = inputs.to_tensor(default_value=self.pad_value)
        elif isinstance(inputs, tf.SparseTensor):
            outputs = tf.sparse.to_dense(inputs, default_value=self.pad_value)
        elif tf.is_tensor(inputs):
            raise TypeError('Tensor is already dense')
        else:
            raise TypeError('Unexpected tensor type {}'.format(type(inputs).__name__))

        return outputs

    def compute_mask(self, inputs, mask=None):
        if not self.mask:
            return None

        if isinstance(inputs, tf.RaggedTensor):
            mask = tf.ones_like(inputs.flat_values, 'bool')
            mask = inputs.with_flat_values(mask)
            mask = mask.to_tensor(False)
        elif isinstance(inputs, tf.SparseTensor):
            mask = tf.ones_like(inputs.values, 'bool')
            mask = inputs.with_values(mask)
            mask = tf.sparse.to_dense(mask, default_value=False)
        elif tf.is_tensor(inputs):
            raise TypeError('Tensor is already dense')
        else:
            raise TypeError('Unexpected tensor type {}'.format(type(inputs).__name__))

        mask = tf.reduce_any(mask, axis=-1)

        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ToDense, self).get_config()
        config.update({'pad_value': self.pad_value, 'mask': self.mask})

        return config
