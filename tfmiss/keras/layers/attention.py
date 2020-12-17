from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils


@tf.keras.utils.register_keras_serializable(package='Miss')
class SelfAttention(tf.keras.layers.Layer):
    """L2-constrained and scaled layer.
    Reference: https://arxiv.org/pdf/1703.09507.pdf
    L2-constrained Softmax Loss for Discriminative Face Verification
    Rajeev Ranjan, Carlos D. Castillo and Rama Chellappa (2017)

    Notes
        As mentioned in paper, theoretically good `alpha` can be estimated as `log(p * (C - 2) / (1 - p))` where `p`
        is the average softmax probability p for correctly classifying a feature and C is a number of classes.
        Usually good `alpha` will be in range [10; 30].
    """

    def __init__(self, units=None, additive=True, use_bias=False, use_scale=True, causal=False, dropout=0.0, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)
        self.supports_masking = True

        self.units = units
        self.additive = additive
        self.use_bias = use_bias
        self.use_scale = use_scale
        self.causal = causal
        self.dropout = dropout

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        att_cls = tf.keras.layers.AdditiveAttention if self.additive else tf.keras.layers.Attention
        self.attend = att_cls(use_scale=self.use_scale, causal=self.causal, dropout=self.dropout)

        num_channels = input_shape[-1]
        if num_channels is None:
            raise ValueError('Channel dimension of the inputs should be defined. Found `None`.')

        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={-1: num_channels})

        if self.additive:
            self.wt = tf.keras.layers.Dense(num_channels, use_bias=self.use_bias, name='Wt')
            self.wx = tf.keras.layers.Dense(num_channels, use_bias=self.use_bias, name='Wx')
            self.wa = self.add_weight(shape=(self.units, 1),
                                      name='{}_Add_Wa'.format(self.name),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        else:
            self.wa = self.add_weight(shape=(self.units, 1),
                                      name='{}_Add_Wa'.format(self.name),
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)

        super(SelfAttention, self).build(input_shape)

    def call(self, inputs, mask=None, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        query = inputs
        value = inputs
        key = inputs

        if self.additive:
            query = set.wt(inputs)
            key = set.wx(inputs)
        else:
            pass
        # scores = self._calculate_scores(query, key)
        #     Add: return math_ops.reduce_sum(math_ops.tanh(q_reshaped + k_reshaped), axis=-1)
        #     Mul: return math_ops.matmul(query, key, transpose_b=True)

        # result = self._apply_scores(scores, value)
        #     weights = nn.softmax(scores)
        #     result = math_ops.matmul(weights, value)

        return self.attend([query, value, key], mask=mask, training=training)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            'units': self.units,
            'additive': self.additive,
            'use_bias': self.use_bias,
            'use_scale': self.use_scale,
            'causal': self.causal,
            'dropout': self.dropout,
        })

        return config


class AdditiveAttention(BaseDenseAttention):
  """Additive attention layer, a.k.a. Bahdanau-style attention.

  Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor of
  shape `[batch_size, Tv, dim]` and `key` tensor of shape
  `[batch_size, Tv, dim]`. The calculation follows the steps:

  1. Reshape `query` and `value` into shapes `[batch_size, Tq, 1, dim]`
     and `[batch_size, 1, Tv, dim]` respectively.
  2. Calculate scores with shape `[batch_size, Tq, Tv]` as a non-linear
     sum: `scores = tf.reduce_sum(tf.tanh(query + value), axis=-1)`
  3. Use scores to calculate a distribution with shape
     `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
  4. Use `distribution` to create a linear combination of `value` with
     shape `batch_size, Tq, dim]`:
     `return tf.matmul(distribution, value)`.

  Args:
    use_scale: If `True`, will create a variable to scale the attention scores.
    causal: Boolean. Set to `True` for decoder self-attention. Adds a mask such
      that position `i` cannot attend to positions `j > i`. This prevents the
      flow of information from the future towards the past.
    dropout: Float between 0 and 1. Fraction of the units to drop for the
      attention scores.

  Call Arguments:

    inputs: List of the following tensors:
      * query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
      * value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
      * key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If not
        given, will use `value` for both `key` and `value`, which is the
        most common case.
    mask: List of the following tensors:
      * query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
        If given, the output will be zero at the positions where
        `mask==False`.
      * value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
        If given, will apply the mask such that values at positions where
        `mask==False` do not contribute to the result.
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (no dropout).

  Output shape:

    Attention outputs of shape `[batch_size, Tq, dim]`.

  The meaning of `query`, `value` and `key` depend on the application. In the
  case of text similarity, for example, `query` is the sequence embeddings of
  the first piece of text and `value` is the sequence embeddings of the second
  piece of text. `key` is usually the same tensor as `value`.

  Here is a code example for using `AdditiveAttention` in a CNN+Attention
  network:

  ```python
  # Variable-length int sequences.
  query_input = tf.keras.Input(shape=(None,), dtype='int32')
  value_input = tf.keras.Input(shape=(None,), dtype='int32')

  # Embedding lookup.
  token_embedding = tf.keras.layers.Embedding(max_tokens, dimension)
  # Query embeddings of shape [batch_size, Tq, dimension].
  query_embeddings = token_embedding(query_input)
  # Value embeddings of shape [batch_size, Tv, dimension].
  value_embeddings = token_embedding(value_input)

  # CNN layer.
  cnn_layer = tf.keras.layers.Conv1D(
      filters=100,
      kernel_size=4,
      # Use 'same' padding so outputs have the same shape as inputs.
      padding='same')
  # Query encoding of shape [batch_size, Tq, filters].
  query_seq_encoding = cnn_layer(query_embeddings)
  # Value encoding of shape [batch_size, Tv, filters].
  value_seq_encoding = cnn_layer(value_embeddings)

  # Query-value attention of shape [batch_size, Tq, filters].
  query_value_attention_seq = tf.keras.layers.AdditiveAttention()(
      [query_seq_encoding, value_seq_encoding])

  # Reduce over the sequence axis to produce encodings of shape
  # [batch_size, filters].
  query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
      query_seq_encoding)
  query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
      query_value_attention_seq)

  # Concatenate query and document encodings to produce a DNN input layer.
  input_layer = tf.keras.layers.Concatenate()(
      [query_encoding, query_value_attention])

  # Add DNN layers, and create Model.
  # ...
  ```
  """

  def __init__(self, use_scale=True, **kwargs):
    super(AdditiveAttention, self).__init__(**kwargs)
    self.use_scale = use_scale

  def build(self, input_shape):
    v_shape = tensor_shape.TensorShape(input_shape[1])
    dim = v_shape[-1]
    if isinstance(dim, tensor_shape.Dimension):
      dim = dim.value
    if self.use_scale:
      self.scale = self.add_weight(
          name='scale',
          shape=[dim],
          initializer=init_ops.glorot_uniform_initializer(),
          dtype=self.dtype,
          trainable=True)
    else:
      self.scale = None
    super(AdditiveAttention, self).build(input_shape)

  def _calculate_scores(self, query, key):
    """Calculates attention scores as a nonlinear sum of query and key.

    Args:
      query: Query tensor of shape `[batch_size, Tq, dim]`.
      key: Key tensor of shape `[batch_size, Tv, dim]`.
    Returns:
      Tensor of shape `[batch_size, Tq, Tv]`.
    """
    # Reshape tensors to enable broadcasting.
    # Reshape into [batch_size, Tq, 1, dim].
    q_reshaped = array_ops.expand_dims(query, axis=-2)
    # Reshape into [batch_size, 1, Tv, dim].
    k_reshaped = array_ops.expand_dims(key, axis=-3)
    if self.use_scale:
      scale = self.scale
    else:
      scale = 1.
    return math_ops.reduce_sum(
        scale * math_ops.tanh(q_reshaped + k_reshaped), axis=-1)

  def get_config(self):
    config = {'use_scale': self.use_scale}
    base_config = super(AdditiveAttention, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
