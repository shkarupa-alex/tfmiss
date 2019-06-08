from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.ragged import ragged_tensor
from tfmiss.ops import load_so


def down_sample(source, keys, freqs, replacement='', threshold=1e-3, min_freq=0, seed=None, name=None):
    """Randomly down-sample high frequency tokens in `source` with `replacement` value.

    Args:
        source: string `Tensor` or `RaggedTensor` or `SparseTensor` of any shape, items to be sampled.
        keys: list of `string`s, keys in frequency vocabulary.
        freqs: list of `int`s, values in frequency vocabulary. Must have same size as keys.
        replacement: `string`, value to set instead of downsampled ones
        threshold: `float`, items occurrence threshold.
        min_freq: `int`, items below that frequency will be treated as unique.
        seed: `int`, used to create a random seed (optional).
            See @{tf.random.set_seed} for behavior.
        name: `string`, a name for the operation (optional).

    Returns:
      A boolean `Tensor` of same shape as source: "keep" flags.
    """
    with tf.name_scope(name or 'down_sample'):
        if isinstance(source, sparse_tensor.SparseTensorValue) or isinstance(source, sparse_tensor.SparseTensor):
            source = sparse_tensor.convert_to_tensor_or_sparse_tensor(source, dtype=tf.string, name=name)
        else:
            source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, dtype=tf.string, name=name)

        if not tf.string.is_compatible_with(source.dtype):
            raise RuntimeError('"Source" must have dtype compatible with "string". '
                               'Actual: {}'.format(source.dtype))

        if isinstance(source, tf.SparseTensor):
            return tf.SparseTensor(
                values=down_sample(source.values, keys, freqs, replacement, threshold, min_freq, seed),
                indices=source.indices,
                dense_shape=source.dense_shape
            )
        elif isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                down_sample(source.flat_values, keys, freqs, replacement, threshold, min_freq, seed)
            )

        keep = sample_mask(
            source=source,
            keys=keys,
            freqs=freqs,
            threshold=threshold,
            min_freq=min_freq,
            seed=seed,
        )
        defaults = tf.fill(tf.shape(source), replacement)

        return tf.where(keep, source, defaults)


def sample_mask(source, keys, freqs, threshold=1e-3, min_freq=0, seed=None, name=None):
    """Generates random mask for downsampling high frequency items.

    Args:
        source: string `Tensor` of any shape, items to be sampled.
        keys: list of `string`s, keys in frequency vocabulary.
        freqs: list of `int`s, values in frequency vocabulary. Must have same size as keys.
        threshold: `float`, items occurrence threshold.
        min_freq: `int`, items below that frequency will be treated as unique.
        seed: `int`, used to create a random seed (optional).
            See @{tf.random.set_seed} for behavior.
        name: `string`, a name for the operation (optional).

    Returns:
      A boolean `Tensor` of same shape as source: "keep" flags.
    """

    with tf.name_scope(name or 'sample_mask'):
        source = tf.convert_to_tensor(source, dtype=tf.string, name='source')
        seed1, seed2 = random_seed.get_seed(seed)

        return load_so().sample_mask(
            source=source,
            keys=keys,
            freqs=freqs,
            threshold=threshold,
            min_freq=min_freq,
            seed=seed1,
            seed2=seed2
        )
