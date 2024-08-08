import tensorflow as tf
from tensorflow.python.ops.lookup_ops import LookupInterface
from tensorflow.python.ops.ragged import ragged_tensor

from tfmiss.ops import tfmiss_ops


def word_piece(
    source,
    lookup_table,
    joiner_prefix="##",
    max_bytes=100,
    max_chars=0,
    unknown_token="[UNK]",
    split_unknown=False,
    skip=None,
    name=None,
):
    """Generates `Continuous bag-of-words` contexts for inference from batched
    list of tokens.

    Args:
        source: `2-D` string `Tensor` or `RaggedTensor`, batched lists of
          tokens [sentences, tokens].
        lookup_table: A lookup table implementing the LookupInterface
          containing the vocabulary of subwords.
        joiner_prefix: `string`, the characters prepended to a wordpiece to
          indicate that it is a suffix to another subword. Default is '##'.
        max_bytes: `int`, maximum size of input token.
        max_chars: `int`, maximum size of subwords, excluding suffix indicator.
          If known, providing this improves the efficiency of decoding long
          words.
        unknown_token: `string` or None, the string value to substitute for an
          unknown token. If set to `None`, no substitution occurs.
        split_unknown: `bool`, whether to split out single unknown characters
          as subtokens. If False (default), words containing unknown characters
          will be treated as single unknown tokens.
        skip: list of strings to pass without changes or None.
        name: `string`, a name for the operation (optional).

    Returns:
        A subwords `RaggedTensor`.
    """
    if not isinstance(lookup_table, LookupInterface):
        raise TypeError(
            "Lookup table should supports LookupInterface, got {}".format(
                lookup_table
            )
        )

    with tf.name_scope(name or "word_piece"):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(
            source, name="source", dtype=tf.string
        )
        if source.shape.rank is None:
            raise ValueError("Rank of `source` must be statically known.")
        if source.shape.rank == 0:
            raise ValueError("Rank of `source` must be greater then 0")

        if not isinstance(source, tf.RaggedTensor) and source.shape.rank > 1:
            source = ragged_tensor.RaggedTensor.from_tensor(
                source, ragged_rank=source.shape.rank - 1
            )

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                word_piece(
                    source.flat_values,
                    lookup_table,
                    joiner_prefix,
                    max_bytes,
                    max_chars,
                    unknown_token,
                    split_unknown,
                    skip,
                )
            )

        values, row_splits, _, _ = tfmiss_ops.miss_wordpiece_tokenize(
            input_values=source,
            vocab_lookup_table=lookup_table.resource_handle,
            suffix_indicator=joiner_prefix,
            use_unknown_token=bool(unknown_token),
            max_bytes_per_word=max_bytes,
            max_chars_per_token=max_chars,
            unknown_token=unknown_token or "[UNK]",
            split_unknown_characters=split_unknown,
            output_row_partition_type="row_splits",
            skip=skip or [],
        )

        if source.shape.rank == 0:
            return values

        return tf.RaggedTensor.from_row_splits(values, row_splits)
