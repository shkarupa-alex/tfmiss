from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.ragged import ragged_tensor
from tfmiss.ops import load_so


def char_ngrams(source, minn, maxn, itself, name=None):
    """Split unicode strings into character ngrams.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to split
        minn: Minimum length of character ngram
        maxn: Maximum length of character ngram
        itself: Strategy for source word preserving.
            One of `"ASIS"`, `"NEVER"`, `"ALWAYS"`, `"ALONE"`.
        name: A name for the operation (optional).
    Returns:
        `Tensor` if rank(source) is 0, `RaggedTensor` with an additional dimension otherwise.
    """
    with tf.name_scope(name or 'char_ngrams'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)
        if source.shape.rank is None:
            raise ValueError('Rank of `source` must be statically known.')

        if isinstance(source, tf.Tensor) and source.shape.rank > 1:
            source = ragged_tensor.RaggedTensor.from_tensor(source, ragged_rank=source.shape.rank - 1)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                char_ngrams(source.flat_values, minn, maxn, itself)
            )

        result_values, result_splits = load_so().char_ngrams(
            source=source,
            minn=minn,
            maxn=maxn,
            itself=itself
        )

        if source.shape.rank == 0:
            return result_values

        return tf.RaggedTensor.from_row_splits(result_values, result_splits)


def split_chars(source, name=None):
    """Split unicode strings into characters.
    Result tokens could be simply joined with empty separator to obtain original strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to split
        name: A name for the operation (optional).
    Returns:
        `Tensor` if rank(source) is 0, `RaggedTensor` with an additional dimension otherwise.
    """
    with tf.name_scope(name or 'split_chars'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)
        if source.shape.rank is None:
            raise ValueError('Rank of `source` must be statically known.')

        if isinstance(source, tf.Tensor) and source.shape.rank > 1:
            source = ragged_tensor.RaggedTensor.from_tensor(source, ragged_rank=source.shape.rank - 1)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                split_chars(source.flat_values)
            )

        result_values, result_splits = load_so().split_chars(
            source=source
        )

        if source.shape.rank == 0:
            return result_values

        return tf.RaggedTensor.from_row_splits(result_values, result_splits)


def split_words(source, stop=False, name=None):
    """Split unicode strings into words.
    Result tokens could be simply joined with empty separator to obtain original strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to split
        stop: Boolean flag, partly ignore rules WB6 and WB7 and break on "stop" marks
        name: A name for the operation (optional).
    Returns:
        `Tensor` if rank(source) is 0, `RaggedTensor` with an additional dimension otherwise.
    """
    with tf.name_scope(name or 'split_words'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)
        if source.shape.rank is None:
            raise ValueError('Rank of `source` must be statically known.')

        if isinstance(source, tf.Tensor) and source.shape.rank > 1:
            source = ragged_tensor.RaggedTensor.from_tensor(source, ragged_rank=source.shape.rank - 1)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                split_words(source.flat_values, stop)
            )

        result_values, result_splits = load_so().split_words(
            source=source,
            stop=stop
        )

        if source.shape.rank == 0:
            return result_values

        return tf.RaggedTensor.from_row_splits(result_values, result_splits)
