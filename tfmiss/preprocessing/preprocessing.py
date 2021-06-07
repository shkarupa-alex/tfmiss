from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import random_seed
from tensorflow.python.ops.ragged import ragged_tensor
from tfmiss.ops import tfmiss_ops


def cbow_context(source, window, empty, name=None):
    """Generates `Continuous bag-of-words` contexts for inference from batched list of tokens.

    Args:
        source: `2-D` string `Tensor` or `RaggedTensor`, batched lists of tokens [sentences, tokens].
        window: `int`, size of context before and after target token, must be > 0.
        name: `string`, a name for the operation (optional).

    Returns:
        `2-D` string `RaggedTensor`: context tokens.
        `2-D` int32 `RaggedTensor`: context positions.
    """
    with tf.name_scope(name or 'cbow_context'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source')

        if source.shape.rank != 2:
            raise ValueError('Rank of `source` must equals 2')

        if not ragged_tensor.is_ragged(source):
            source = ragged_tensor.RaggedTensor.from_tensor(source, ragged_rank=1)

        if source.ragged_rank != 1:
            raise ValueError('Ragged rank of `source` must equals 1')

        context_values, context_splits, context_positions = tfmiss_ops.miss_cbow_context(
            source_values=source.values,
            source_splits=source.row_splits,
            window=window,
            empty=empty
        )

        context = tf.RaggedTensor.from_row_splits(context_values, context_splits)
        position = tf.RaggedTensor.from_row_splits(context_positions, context_splits)

        return context, position


def cont_bow(source, window, seed=None, name=None):
    """Generates `Continuous bag-of-words` target and context pairs from batched list of tokens.

    Args:
        source: `2-D` string `Tensor` or `RaggedTensor`, batched lists of tokens [sentences, tokens].
        window: `int`, size of context before and after target token, must be > 0.
        seed: `int`, used to create a random seed (optional).
            See @{tf.random.set_seed} for behavior.
        name: `string`, a name for the operation (optional).

    Returns:
        `1-D` string `Tensor`: target tokens.
        `2-D` string `RaggedTensor`: context tokens.
        `2-D` int32 `RaggedTensor`: context positions.
    """
    with tf.name_scope(name or 'cont_bow'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source')

        if source.shape.rank != 2:
            raise ValueError('Rank of `source` must equals 2')

        if not ragged_tensor.is_ragged(source):
            source = ragged_tensor.RaggedTensor.from_tensor(source, ragged_rank=1)

        if source.ragged_rank != 1:
            raise ValueError('Ragged rank of `source` must equals 1')

        seed1, seed2 = random_seed.get_seed(seed)

        target, context_values, context_splits, context_positions = tfmiss_ops.miss_cont_bow(
            source_values=source.values,
            source_splits=source.row_splits,
            window=window,
            seed=seed1,
            seed2=seed2
        )

        context = tf.RaggedTensor.from_row_splits(context_values, context_splits)
        position = tf.RaggedTensor.from_row_splits(context_positions, context_splits)

        return target, context, position


def skip_gram(source, window, seed=None, name=None):
    """Generates `Skip-Gram` target and context pairs from batched list of tokens.

    Args:
        source: `2-D` string `Tensor` or `RaggedTensor`, batched lists of tokens [sentences, tokens].
        window: `int`, size of context before and after target token, must be > 0.
        seed: `int`, used to create a random seed (optional).
            See @{tf.random.set_seed} for behavior.
        name: `string`, a name for the operation (optional).

    Returns:
      Two `1-D` string `Tensor`s: target and context tokens.
    """
    with tf.name_scope(name or 'skip_gram'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source')

        if source.shape.rank != 2:
            raise ValueError('Rank of `source` must equals 2')

        if not ragged_tensor.is_ragged(source):
            source = ragged_tensor.RaggedTensor.from_tensor(source, ragged_rank=1)

        if source.ragged_rank != 1:
            raise ValueError('Ragged rank of `source` must equals 1')

        seed1, seed2 = random_seed.get_seed(seed)

        target, context = tfmiss_ops.miss_skip_gram(
            source_values=source.values,
            source_splits=source.row_splits,
            window=window,
            seed=seed1,
            seed2=seed2
        )
        return target, context


def spaces_after(source, name=None):
    """Separates spaces from tokens.

    Args:
        source: `2-D` string `Tensor` or `RaggedTensor`, batched lists of "tokens with spaces" [sentences, tokens].
        name: `string`, a name for the operation (optional).

    Returns:
        `2-D` string `RaggedTensor`: tokens.
        `2-D` string `RaggedTensor`: spaces.
    """
    with tf.name_scope(name or 'spaces_after'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source')

        if source.shape.rank != 2:
            raise ValueError('Rank of `source` must equals 2')

        if not ragged_tensor.is_ragged(source):
            source = ragged_tensor.RaggedTensor.from_tensor(source, ragged_rank=1)

        if source.ragged_rank != 1:
            raise ValueError('Ragged rank of `source` must equals 1')

        token_values, space_values, common_splits = tfmiss_ops.miss_spaces_after(
            source_values=source.values,
            source_splits=source.row_splits
        )

        tokens = tf.RaggedTensor.from_row_splits(token_values, common_splits)
        spaces = tf.RaggedTensor.from_row_splits(space_values, common_splits)

        return tokens, spaces
