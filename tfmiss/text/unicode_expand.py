import tensorflow as tf
from tensorflow.python.ops.ragged import ragged_tensor

from tfmiss.ops import tfmiss_ops


def char_ngrams(source, minn, maxn, itself, skip=None, name=None):
    """Split unicode strings into character ngrams.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to split
        minn: Minimum length of character ngram
        maxn: Maximum length of character ngram
        itself: Strategy for source word preserving.
            One of `"asis"`, `"never"`, `"always"`, `"alone"`.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` if rank(source) is 0, `RaggedTensor` with an additional
          dimension otherwise.
    """
    with tf.name_scope(name or "char_ngrams"):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(
            source, name="source", dtype=tf.string
        )
        if source.shape.rank is None:
            raise ValueError("Rank of `source` must be statically known.")

        if not isinstance(source, tf.RaggedTensor) and source.shape.rank > 1:
            source = ragged_tensor.RaggedTensor.from_tensor(
                source, ragged_rank=source.shape.rank - 1
            )

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                char_ngrams(source.flat_values, minn, maxn, itself, skip)
            )

        result_values, result_splits = tfmiss_ops.miss_char_ngrams(
            source=source,
            minn=minn,
            maxn=maxn,
            itself=itself.upper(),
            skip=skip or [],
        )

        if source.shape.rank == 0:
            return result_values

        return tf.RaggedTensor.from_row_splits(result_values, result_splits)


def split_chars(source, skip=None, name=None):
    """Split unicode strings into characters.
    Result tokens could be simply joined with empty separator to obtain
    original strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to split
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` if rank(source) is 0, `RaggedTensor` with an additional
          dimension otherwise.
    """
    with tf.name_scope(name or "split_chars"):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(
            source, name="source", dtype=tf.string
        )
        if source.shape.rank is None:
            raise ValueError("Rank of `source` must be statically known.")

        if not isinstance(source, tf.RaggedTensor) and source.shape.rank > 1:
            source = ragged_tensor.RaggedTensor.from_tensor(
                source, ragged_rank=source.shape.rank - 1
            )

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                split_chars(source.flat_values, skip)
            )

        result_values, result_splits = tfmiss_ops.miss_split_chars(
            source=source,
            skip=skip or [],
        )

        if source.shape.rank == 0:
            return result_values

        return tf.RaggedTensor.from_row_splits(result_values, result_splits)


def split_words(source, extended=False, skip=None, name=None):
    """Split unicode strings into words.
    Result tokens could be simply joined with empty separator to obtain
    original strings.
    See http://www.unicode.org/reports/tr29/#Word_Boundaries

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to split
        extended: Ignore rules WB6, WB7, WB9, WB10, WB11 and WB12 to break on
          "stop", "colon" & etc.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` if rank(source) is 0, `RaggedTensor` with an additional
          dimension otherwise.
    """
    with tf.name_scope(name or "split_words"):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(
            source, name="source", dtype=tf.string
        )
        if source.shape.rank is None:
            raise ValueError("Rank of `source` must be statically known.")

        if not isinstance(source, tf.RaggedTensor) and source.shape.rank > 1:
            source = ragged_tensor.RaggedTensor.from_tensor(
                source, ragged_rank=source.shape.rank - 1
            )

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                split_words(source.flat_values, extended, skip)
            )

        result_values, result_splits = tfmiss_ops.miss_split_words(
            source=source,
            extended=extended,
            skip=skip or [],
        )

        if source.shape.rank == 0:
            return result_values

        return tf.RaggedTensor.from_row_splits(result_values, result_splits)
