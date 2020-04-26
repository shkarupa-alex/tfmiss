from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.ragged import ragged_tensor
from tfmiss.ops import load_so


def down_sample(source, freq_vocab, replacement='', threshold=1e-3, min_freq=0, seed=None, name=None):
    """Randomly down-sample high frequency tokens in `source` with `replacement` value.

    Args:
        source: string `Tensor` or `RaggedTensor` or `SparseTensor` of any shape, items to be sampled.
        freq_vocab: `Counter` with frequencies vocabulary.
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
                values=down_sample(source.values, freq_vocab, replacement, threshold, min_freq, seed),
                indices=source.indices,
                dense_shape=source.dense_shape
            )
        elif isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                down_sample(source.flat_values, freq_vocab, replacement, threshold, min_freq, seed)
            )

        keep = sample_mask(
            source=source,
            freq_vocab=freq_vocab,
            threshold=threshold,
            min_freq=min_freq,
            seed=seed,
        )
        defaults = tf.fill(tf.shape(source), replacement)

        return tf.where(keep, source, defaults)


def sample_mask(source, freq_vocab, threshold=1e-3, min_freq=0, seed=None, name=None):
    """Generates random mask for downsampling high frequency items.

    Args:
        source: string `Tensor` of any shape, items to be sampled.
        freq_vocab: `Counter` with frequencies vocabulary.
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

        if not isinstance(freq_vocab, Counter):
            raise ValueError('Frequency vocabulary should be a Counter instance')

        keys, freqs = zip(*freq_vocab.most_common())

        return load_so().miss_sample_mask(
            source=source,
            keys=keys,
            freqs=freqs,
            threshold=threshold,
            min_freq=min_freq,
            seed=seed1,
            seed2=seed2
        )


def sample_probs(freq_vocab, threshold=1e-3):
    """Estimates downsampling probabilities for word2vec.
    "Keep probabilities" below estimated with word2vec equation from source code (not from paper)
    See https://github.com/maxoodf/word2vec#subsampling-down-sampling

    Args:
        freq_vocab: `Counter` with frequencies vocabulary.
        threshold: `float`, items occurrence threshold.

    Returns:
        A `dict` of downsampling keys with corresponding keep probabilities and a discard fraction size.
    """
    if not isinstance(freq_vocab, Counter):
        raise ValueError('Frequency vocabulary should be a Counter instance')

    freq_keys, freq_vals = zip(*freq_vocab.most_common())
    freq_vals = np.array(freq_vals)

    total_freq = np.sum(freq_vals)
    total_thold = total_freq * threshold
    keep_probs = (np.sqrt(freq_vals / total_thold) + 1.) * total_thold / freq_vals

    samp_size = np.argmax(np.append(keep_probs, [1.0]) >= 1.0)
    freq_keys, keep_probs = freq_keys[:samp_size], keep_probs[:samp_size]

    return dict(zip(freq_keys, keep_probs))
