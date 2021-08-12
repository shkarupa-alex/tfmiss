# Taken from https://github.com/tensorflow/text/blob/master/tensorflow_text/python/keras/layers/todense.py
#
# Copyright 2020 TF.Text Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import layers
from keras.utils.generic_utils import register_keras_serializable


@register_keras_serializable(package='Miss')
class ToDense(layers.Layer):
    """ Layer that makes padding and masking a Composite Tensors effortless. The layer takes a RaggedTensor or
    a SparseTensor and converts it to a uniform tensor by right-padding it or filling in missing values.

    Arguments:
      pad_value: A value used to pad and fill in the missing values. Should be a meaningless value for the input data.
        Default is '0'.
      mask: A Boolean value representing whether to mask the padded values. If true, no any downstream Masking layer
        or Embedding layer with mask_zero=True should be added. Default is 'False'.

    Input shape: Any Ragged or Sparse Tensor is accepted, but it requires the type of input to be specified via the
        Input or InputLayer from the Keras API.

    Output shape: The output is a uniform tensor having the same shape, in case of a ragged input or the same dense
        shape, in case of a sparse input.
    """

    def __init__(self, pad_value=0, mask=False, **kwargs):
        kwargs['trainable'] = False
        super(ToDense, self).__init__(**kwargs)
        self._compute_output_and_mask_jointly = True
        self._supports_ragged_inputs = True
        self.pad_value = pad_value
        self.mask = mask

    def build(self, input_shape):
        if self.mask:
            self.masking_layer = layers.Masking(mask_value=self.pad_value)

    def call(self, inputs, **kwargs):
        if isinstance(inputs, tf.RaggedTensor):
            # Convert the ragged tensor to a padded uniform tensor
            outputs = inputs.to_tensor(default_value=self.pad_value)
        elif isinstance(inputs, tf.SparseTensor):
            # Fill in the missing value in the sparse_tensor
            outputs = tf.sparse.to_dense(inputs, default_value=self.pad_value)
        elif tf.is_tensor(inputs):
            outputs = inputs
        else:
            raise TypeError('Unexpected tensor type {}'.format(type(inputs).__name__))

        if self.mask:
            outputs = self.masking_layer(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ToDense, self).get_config()
        config.update({'pad_value': self.pad_value, 'mask': self.mask})

        return config
