from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tfmiss.text import wrap_with, char_ngrams, lower_case, title_case, upper_case


@tf.keras.utils.register_keras_serializable(package='Miss')
class CharNgams(tf.keras.layers.Layer):
    def __init__(self, minn, maxn, itself, reserved=None, left='<', right='>', *args, **kwargs):
        super(CharNgams, self).__init__(*args, **kwargs)
        self.input_spec = tf.keras.layers.InputSpec(dtype='string')
        self._supports_ragged_inputs = True

        self.minn = minn
        self.maxn = maxn
        self.itself = itself
        self.reserved = [] if reserved is None else reserved
        self.left = left
        self.right = right

    def call(self, inputs, **kwargs):
        outputs = wrap_with(inputs, self.left, self.right, skip=self.reserved)
        outputs = char_ngrams(outputs, self.minn, self.maxn, self.itself, skip=self.reserved)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + (None,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'minn': self.minn,
            'maxn': self.maxn,
            'itself': self.itself,
            'reserved': self.reserved,
            'left': self.left,
            'right': self.right,
        })

        return config


@tf.keras.utils.register_keras_serializable(package='Miss')
class WordShape(tf.keras.layers.Layer):
    SHAPE_HAS_CASE = 1
    SHAPE_LOWER_CASE = 2
    SHAPE_UPPER_CASE = 4
    SHAPE_TITLE_CASE = 8
    SHAPE_MIXED_CASE = 16
    SHAPE_ALL_CASES = SHAPE_HAS_CASE | SHAPE_LOWER_CASE | SHAPE_UPPER_CASE | SHAPE_TITLE_CASE | SHAPE_MIXED_CASE

    # Mean and std length from Universal Dependencies and large russian POS corporas
    # Tokens (split_words): 3.057 and 3.118
    # Words: 4.756 and 3.453
    SHAPE_LENGTH_NORM = 32

    SHAPE_ALL = SHAPE_ALL_CASES | SHAPE_LENGTH_NORM

    def __init__(self, options, mean_len=3.906, std_len=3.285, *args, **kwargs):
        super(WordShape, self).__init__(*args, **kwargs)
        self.input_spec = tf.keras.layers.InputSpec(dtype='string')
        self._supports_ragged_inputs = True

        self.options = options
        self.mean_len = mean_len
        self.std_len = std_len

    # def build(self, input_shape):
    #     pass

    def call(self, inputs, **kwargs):
        outputs = []

        inputs_lower = lower_case(inputs)
        inputs_upper = upper_case(inputs)
        has_case = tf.not_equal(inputs_lower, inputs_upper)
        if self.options & self.SHAPE_HAS_CASE:
            outputs.append(has_case)

        if self.options & self.SHAPE_LOWER_CASE or self.options & self.SHAPE_MIXED_CASE:
            is_lower = tf.logical_and(
                has_case,
                tf.equal(inputs, inputs_lower)
            )
        if self.options & self.SHAPE_LOWER_CASE:
            outputs.append(is_lower)

        if self.options & self.SHAPE_UPPER_CASE or self.options & self.SHAPE_MIXED_CASE:
            is_upper = tf.logical_and(
                has_case,
                tf.equal(inputs, inputs_upper)
            )
        if self.options & self.SHAPE_UPPER_CASE:
            outputs.append(is_upper)

        if self.options & self.SHAPE_TITLE_CASE or self.options & self.SHAPE_MIXED_CASE:
            inputs_title = title_case(inputs)
            is_title = tf.logical_and(
                has_case,
                tf.equal(inputs, inputs_title)
            )
        if self.options & self.SHAPE_TITLE_CASE:
            outputs.append(is_title)

        if self.options & self.SHAPE_MIXED_CASE:
            no_case = tf.logical_not(has_case)
            is_mixed = tf.logical_not(tf.logical_or(
                tf.logical_or(no_case, is_lower),
                tf.logical_or(is_upper, is_title)
            ))
            outputs.append(is_mixed)

        if self.options & self.SHAPE_LENGTH_NORM:
            length_norm = tf.strings.length(inputs, unit='UTF8_CHAR')
            length_norm = (tf.cast(length_norm, self._compute_dtype) - self.mean_len) / self.std_len
            outputs.append(length_norm)

        outputs = [tf.cast(o, self._compute_dtype) for o in outputs]
        outputs = tf.stack(outputs, axis=-1)

        return outputs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        units = 0
        options = [
            self.SHAPE_HAS_CASE, self.SHAPE_LOWER_CASE, self.SHAPE_UPPER_CASE, self.SHAPE_TITLE_CASE,
            self.SHAPE_MIXED_CASE, self.SHAPE_LENGTH_NORM]
        for opt in options:
            if self.options & opt:
                units += 1

        return input_shape + (units,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'options': self.options,
            'mean_len': self.mean_len,
            'std_len': self.std_len
        })

        return config
