from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops.ragged import ragged_tensor
from tfmiss.ops import load_so


def lower_case(source, name=None):
    """Lowercases unicode strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to make lower.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape as input.
    """
    with tf.name_scope(name or 'lower_case'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                lower_case(source.flat_values)
            )

        return load_so().lower_case(
            source=source
        )


def normalize_unicode(source, form, name=None):
    """Normalizes unicode strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to normalize.
        form: Scalar value, name of normalization algorithm.
            One of `"NFD"`, `"NFC"`, `"NFKD"`, `"NFKC"`.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'normalize_unicode'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                normalize_unicode(source.flat_values, form)
            )

        return load_so().normalize_unicode(
            source=source,
            form=form
        )


def replace_regex(source, pattern, rewrite, name=None):
    """Replaces all regex matchs from `needle` to corresponding unicode strings in `haystack`.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, source strings for replacing.
        pattern: List of RE2 patterns to search in source
        rewrite: List of strings to replace with. Should have same length as `needle`.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'replace_regex'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                replace_regex(source.flat_values, pattern, rewrite)
            )

        return load_so().replace_regex(
            source=source,
            pattern=pattern,
            rewrite=rewrite
        )


def replace_string(source, needle, haystack, name=None):
    """Replaces all unicode substrings from `needle` to corresponding unicode strings in `haystack`.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, source strings for replacing.
        needle: List of strings to search in source
        haystack: List of strings to replace with. Should have same length as `needle`.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'replace_string'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                replace_string(source.flat_values, needle, haystack)
            )

        return load_so().replace_string(
            source=source,
            needle=needle,
            haystack=haystack
        )


def title_case(source, name=None):
    """Titlecases unicode strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to make title.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'title_case'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                title_case(source.flat_values)
            )

        return load_so().title_case(
            source=source
        )


def upper_case(source, name=None):
    """Uppercases unicode strings.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to make upper.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'upper_case'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                upper_case(source.flat_values)
            )

        return load_so().upper_case(
            source=source
        )


def wrap_with(source, left, right, name=None):
    """Wraps unicode strings with "left" and "right"

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to replace digits.
        left: Scalar string to add in the beginning
        right: Scalar string to add in the ending
        name: A name for the operation (optional).
    Returns:
        `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'wrap_with'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                wrap_with(source.flat_values, left, right)
            )

        return load_so().wrap_with(
            source=source,
            left=left,
            right=right
        )


def zero_digits(source, name=None):
    """Replaces each digit in unicode strings with 0.

    Args:
        source: `Tensor` or `RaggedTensor` of any shape, strings to replace digits.
        name: A name for the operation (optional).
    Returns:
        `Tensor` or `RaggedTensor` of same shape and size as input.
    """

    with tf.name_scope(name or 'zero_digits'):
        source = ragged_tensor.convert_to_tensor_or_ragged_tensor(source, name='source', dtype=tf.string)

        if isinstance(source, tf.RaggedTensor):
            return source.with_flat_values(
                zero_digits(source.flat_values)
            )

        return load_so().zero_digits(
            source=source
        )
