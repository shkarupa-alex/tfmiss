import tensorflow as tf
from tensorflow.python.ops.ragged import ragged_tensor
from tfmiss.ops import tfmiss_ops


def char_category(source, first=True, skip=None, name=None):
    """Get first/last character category in unicode strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to make lower.
        first: boolean flag indicating which character should be tested: first or last.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape as input.
    """
    with tf.name_scope(name or 'char_category'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                char_category(source.flat_values, first, skip)
            )

        return tfmiss_ops.miss_char_category(
            source=source,
            first=first,
            skip=skip or [],
        )


def lower_case(source, skip=None, name=None):
    """Lowercases unicode strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to make lower.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape as input.
    """
    with tf.name_scope(name or 'lower_case'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                lower_case(source.flat_values, skip)
            )

        return tfmiss_ops.miss_lower_case(
            source=source,
            skip=skip or [],
        )


def normalize_unicode(source, form, skip=None, name=None):
    """Normalizes unicode strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to normalize.
        form: Scalar value, name of normalization algorithm.
            One of `"NFD"`, `"NFC"`, `"NFKD"`, `"NFKC"`.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'normalize_unicode'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                normalize_unicode(source.flat_values, form, skip)
            )

        return tfmiss_ops.miss_normalize_unicode(
            source=source,
            form=form,
            skip=skip or [],
        )


def replace_regex(source, pattern, rewrite, skip=None, name=None):
    """Replaces all regex matchs from `needle` to corresponding unicode strings in `haystack`.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, source strings for replacing.
        pattern: List of RE2 patterns to search in source
        rewrite: List of strings to replace with. Should have same length as `needle`.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'replace_regex'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                replace_regex(source.flat_values, pattern, rewrite, skip)
            )

        return tfmiss_ops.miss_replace_regex(
            source=source,
            pattern=pattern,
            rewrite=rewrite,
            skip=skip or [],
        )


def replace_string(source, needle, haystack, skip=None, name=None):
    """Replaces all unicode substrings from `needle` to corresponding unicode strings in `haystack`.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, source strings for replacing.
        needle: List of strings to search in source
        haystack: List of strings to replace with. Should have same length as `needle`.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'replace_string'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                replace_string(source.flat_values, needle, haystack, skip)
            )

        return tfmiss_ops.miss_replace_string(
            source=source,
            needle=needle,
            haystack=haystack,
            skip=skip or [],
        )


def sub_string(source, start, limit=None, skip=None, name=None):
    """Cuts substrings starting at position `start` and spans `limit` characters.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, source strings for cut substring.
        start: Substring start position. If negative, will be interpreted as "from the end of string"
        limit: Substring length. `None` or any negative value will be interpreted as "to the end of string".
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'sub_string'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                sub_string(source.flat_values, start, limit, skip)
            )

        return tfmiss_ops.miss_sub_string(
            source=source,
            start=start,
            limit=-1 if limit is None else limit,
            skip=skip or [],
        )


def title_case(source, skip=None, name=None):
    """Titlecases unicode strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to make title.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'title_case'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                title_case(source.flat_values, skip)
            )

        return tfmiss_ops.miss_title_case(
            source=source,
            skip=skip or [],
        )


def upper_case(source, skip=None, name=None):
    """Uppercases unicode strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to make upper.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'upper_case'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                upper_case(source.flat_values, skip)
            )

        return tfmiss_ops.miss_upper_case(
            source=source,
            skip=skip or [],
        )


def wrap_with(source, left, right, skip=None, name=None):
    """Wraps unicode strings with "left" and "right"

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to replace digits.
        left: Scalar string to add in the beginning
        right: Scalar string to add in the ending
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'wrap_with'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                wrap_with(source.flat_values, left, right, skip)
            )

        return tfmiss_ops.miss_wrap_with(
            source=source,
            left=left,
            right=right,
            skip=skip or [],
        )


def zero_digits(source, skip=None, name=None):
    """Replaces each digit in unicode strings with 0.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to replace digits.
        skip: list of strings to pass without changes or None.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'zero_digits'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                zero_digits(source.flat_values, skip)
            )

        return tfmiss_ops.miss_zero_digits(
            source=source,
            skip=skip or [],
        )
