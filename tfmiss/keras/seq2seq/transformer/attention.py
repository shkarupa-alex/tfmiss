from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.python.keras import backend as K


class ScaledDotProductAttention(keras.layers.Layer):
    """Scaled Dot-Product attention layer."""

    def __init__(self, dropout_rate, *args, **kwargs):
        """Initialize `ScaledDotProductAttention`."""
        super(ScaledDotProductAttention, self).__init__(*args, **kwargs)
        self.input_spec = [
            keras.layers.InputSpec(min_ndim=3),  # Query
            keras.layers.InputSpec(min_ndim=3),  # Key
            keras.layers.InputSpec(min_ndim=3),  # Value
            keras.layers.InputSpec(min_ndim=3),  # Mask
        ]
        # self.supports_masking = True # TODO

        self.dropout_rate = dropout_rate
        self.drop = None
        self.depth_scale = None

    def build(self, input_shape):
        # Check the number of inputs
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
            raise ValueError('A `ScaledDotProductAttention` layer should be called '
                             'on exactly 4 inputs: `query`, `key`, `value`, `mask`')

        # Check for inputs have same rank
        ranks = [len(shape) for shape in input_shape]
        if len(set(ranks)) > 1:
            raise ValueError('All `ScaledDotProductAttention` inputs should have '
                             'same rank: {}'.format(input_shape))

        # Check for all inputs have equal starting dimensions:
        # batch size, number of heads & etc.
        for dim in range(ranks[0] - 2):
            values = [input_shape[i][dim] for i in range(4)]
            if len(set(values)) > 1:
                raise ValueError('All `ScaledDotProductAttention` inputs should have '
                                 'same non-channel dimensions: {}'.format(input_shape))

        # Check for dimensions required by matmul are equal
        query_shape, key_shape, value_shape, mask_shape = input_shape
        if query_shape[-1] != key_shape[-1]:
            raise ValueError('Last dimension of `query` and `key` should be equal. '
                             'Got {}'.format([query_shape, key_shape]))
        if query_shape[-2] != mask_shape[-2] and None not in {query_shape[-2], mask_shape[-2]}:
            raise ValueError('Last but one dimensions of `query` and `mask` should be '
                             'equal. Got {}'.format([query_shape, mask_shape]))
        if key_shape[-2] != mask_shape[-1] and key_shape[-2] is not None:
            raise ValueError('Last but one dimension of `key` and last dimension of `mask` '
                             'should be equal. Got {}'.format([key_shape, mask_shape]))
        if key_shape[-2] != value_shape[-2] and key_shape[-2] is not None:
            raise ValueError('Last but one dimensions of `key` and `value` should be '
                             'equal. Got {}'.format([key_shape, value_shape]))

        # Remember actual inputs shape
        for i, name in enumerate(['query', 'key', 'value', 'mask']):
            if None in set(input_shape[i][-1:]):
                raise ValueError('Channel dimension of `{}` should be statically '
                                 'known. Got {}'.format(name, input_shape[i]))
            self.input_spec[i] = keras.layers.InputSpec(
                ndim=len(input_shape[i]),
                axes={-1: input_shape[i][-1]}
            )

        # Estimate scale constant
        depth = key_shape[-1]
        self.depth_scale = K.cast(depth ** -0.5, K.floatx())

        self.drop = keras.layers.Dropout(rate=self.dropout_rate)

        super(ScaledDotProductAttention, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        query, key, value, mask = inputs

        # Scale query to prevent the dot product between
        # query and key from growing too large.
        query *= self.depth_scale

        # Calculate dot product attention
        score = K.math_ops.matmul(query, key, transpose_b=True)
        score += mask
        weight = K.nn.softmax(score)
        weight = self.drop(weight, training=training)
        attention = K.math_ops.matmul(weight, value)

        return attention

    def compute_output_shape(self, input_shape):
        query_shape, _, value_shape, _ = input_shape

        return query_shape[:-1].concatenate(value_shape[-1])

    def get_config(self):
        config = {'dropout_rate': self.dropout_rate}
        base_config = super(ScaledDotProductAttention, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class MultiHeadAttention(keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, dropout_rate, *args, **kwargs):
        """Initialize Attention.
        Args:
          hidden_size: int, output dim of hidden layer.
          num_heads: int, number of heads to repeat the same attention structure.
          dropout_rate: float, dropout rate inside attention for training.
        """
        if hidden_size % num_heads:
            raise ValueError('Hidden size ({}) must be divisible by the number '
                             'of heads ({}).'.format(hidden_size, num_heads))

        super(MultiHeadAttention, self).__init__(*args, **kwargs)
        self.input_spec = [
            keras.layers.InputSpec(ndim=3),  # Query
            keras.layers.InputSpec(ndim=3),  # Key
            keras.layers.InputSpec(ndim=3),  # Value
            keras.layers.InputSpec(ndim=4),  # Mask
        ]
        # self.supports_masking = True # TODO

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        self.query_dense = None
        self.key_dense = None
        self.value_dense = None
        self.out_dense = None
        self.sdp_attention = None

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 4:
            raise ValueError('A `MultiHeadAttention` layer should be called '
                             'on exactly 4 inputs: `query`, `key`, `value`, `mask`')

        self.query_dense = keras.layers.Dense(self.hidden_size, use_bias=False)
        self.key_dense = keras.layers.Dense(self.hidden_size, use_bias=False)
        self.value_dense = keras.layers.Dense(self.hidden_size, use_bias=False)
        self.out_dense = keras.layers.Dense(self.hidden_size, use_bias=False)
        self.sdp_attention = ScaledDotProductAttention(self.dropout_rate)

        super(MultiHeadAttention, self).build(input_shape)

    def call(self, inputs, training=None, cache=None):
        """Apply attention mechanism to x and y.
        Args:
          inputs: a list of tensors
              query: a tensor with shape [batch_size, length, hidden_size]
              key: a tensor with shape [batch_size, length, hidden_size]
              value: a tensor with shape [batch_size, length, hidden_size]
              mask: attention bias that will be added to the result of the dot product.
          training: boolean, whether in training mode or not.
          cache: (Used during prediction) dictionary with tensors containing results
            of previous attentions. The dictionary must have the items:
                {'k': tensor with shape [batch_size, i, key_channels],
                 'v': tensor with shape [batch_size, i, value_channels]}
            where i is the current decoded length.
        Returns:
          Attention layer output with shape [batch_size, length_x, hidden_size]
        """
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 4:
            raise ValueError('An `Attention` layer should be called on exactly '
                             '4 inputs: `query`, `key`, `value`, `mask`')

        query, key, value, mask = inputs

        # Linearly project the query, key and value using different learned projections.
        # This is in preparation of splitting them into multiple heads.
        # Multi-head attention uses multiple queries, keys, and values rather than
        # regular attention (which uses a single query, key and value).
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # TODO
        # if cache is not None:
        #     # Combine cached keys and values with new keys and values.
        #     key = K.array_ops.concat([cache['key'], key], axis=1)
        #     value = K.array_ops.concat([cache['value'], value], axis=1)
        #
        #     # Update cache
        #     cache['key'] = key
        #     cache['value'] = value

        # Split query, key, value into heads.
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        # Calculate dot product attention
        attention = self.sdp_attention([query, key, value, mask], training=training)

        # Recombine heads --> [batch_size, length, hidden_size]
        attention = self.combine_heads(attention)

        # Run the combined outputs through another linear projection layer.
        attention = self.out_dense(attention)

        return attention

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1].concatenate(self.hidden_size)

    def get_config(self):
        config = {
            'hidden_size': self.hidden_size,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate,
        }
        base_config = super(MultiHeadAttention, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def split_heads(self, x):
        """Split x into different heads, and transpose the resulting value.
        The tensor is transposed to insure the inner dimensions hold the correct
        values during the matrix multiplication.
        Args:
          x: A tensor with shape [batch_size, length, hidden_size]
        Returns:
          A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
        """
        with K.name_scope('split_heads'):
            batch_size = K.shape(x)[0]
            length = K.shape(x)[1]

            # Calculate depth of last dimension after it has been split.
            depth = self.hidden_size // self.num_heads

            # Split the last dimension
            x = K.reshape(x, [batch_size, length, self.num_heads, depth])

            # Transpose the result
            return K.permute_dimensions(x, [0, 2, 1, 3])

    def combine_heads(self, x):
        """Combine tensor that has been split.
        Args:
          x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]
        Returns:
          A tensor with shape [batch_size, length, hidden_size]
        """
        with K.name_scope('combine_heads'):
            batch_size = K.shape(x)[0]
            length = K.shape(x)[2]
            x = K.permute_dimensions(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]

            return K.reshape(x, [batch_size, length, self.hidden_size])


class SelfAttention(MultiHeadAttention):
    """Multi-headed self-attention layer."""

    def __init__(self, *args, **kwargs):
        """Initialize SelfAttention."""
        super(SelfAttention, self).__init__(*args, **kwargs)
        # self.input_spec = [self.input_spec[0], self.input_spec[-1]]  # skip key and value

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError('A `SelfAttention` layer should be called '
                             'on exactly 2 inputs: `query`, `mask`')
        query_shape, mask_shape = input_shape

        super(SelfAttention, self).build([query_shape, query_shape, query_shape, mask_shape])
        # self.input_spec = [self.input_spec[0], self.input_spec[-1]]  # skip key and value

    def call(self, inputs, training=None, cache=None):
        query, mask = inputs

        return super(SelfAttention, self).call([query, query, query, mask], training, cache)
