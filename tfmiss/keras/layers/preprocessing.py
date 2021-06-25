from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow.python.keras.utils import tf_utils
from tfmiss.text import char_category, lower_case, title_case, upper_case


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

    SHAPE_LEFT_SAME = 64
    SHAPE_RIGHT_SAME = 128
    SHAPE_LEFT2_SAME = 256
    SHAPE_RIGHT2_SAME = 512
    SHAPE_ALL_SAME = SHAPE_LEFT_SAME | SHAPE_RIGHT_SAME | SHAPE_LEFT2_SAME | SHAPE_RIGHT2_SAME

    SHAPE_CHAR_CAT_FIRST = 1024
    SHAPE_CHAR_CAT_LAST = 2048
    SHAPE_CHAR_CAT_BOTH = SHAPE_CHAR_CAT_FIRST | SHAPE_CHAR_CAT_LAST

    SHAPE_ALL = SHAPE_ALL_CASES | SHAPE_LENGTH_NORM | SHAPE_ALL_SAME | SHAPE_CHAR_CAT_BOTH

    def __init__(self, options, mean_len=3.906, std_len=3.285, char_embed=5, *args, **kwargs):
        super(WordShape, self).__init__(*args, **kwargs)
        self.input_spec = tf.keras.layers.InputSpec(dtype='string')
        self._supports_ragged_inputs = True

        if 0 == options:
            raise ValueError('At least one shape option should be selected')

        self.options = options
        self.mean_len = mean_len
        self.std_len = std_len

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if self.options & WordShape.SHAPE_CHAR_CAT_FIRST or self.options & WordShape.SHAPE_CHAR_CAT_LAST:
            category_vocab = [
                'Lu', 'Ll', 'Lt', 'Lm', 'Lo', 'Mn', 'Me', 'Mc', 'Nd', 'Nl', 'No', 'Zs', 'Zl', 'Zp', 'Cc', 'Cf',
                'Co', 'Cs', 'Pd', 'Ps', 'Pe', 'Pc', 'Po', 'Sm', 'Sc', 'Sk', 'So', 'Pi', 'Pf']
            self.cat_lookup = StringLookup(num_oov_indices=0, oov_token='Cn', vocabulary=category_vocab)
            if self.cat_lookup.vocab_size() != 30:
                raise ValueError('Wrong vocabulary size')

        super(WordShape, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs_one, outputs_many = [], []

        # Case
        any_case = self.SHAPE_HAS_CASE | self.SHAPE_LOWER_CASE | self.SHAPE_UPPER_CASE | self.SHAPE_TITLE_CASE | \
                   self.SHAPE_MIXED_CASE
        if self.options & any_case:
            inputs_lower = lower_case(inputs)
            inputs_upper = upper_case(inputs)
            has_case = tf.not_equal(inputs_lower, inputs_upper)

        if self.options & self.SHAPE_HAS_CASE:
            outputs_one.append(has_case)

        if self.options & self.SHAPE_LOWER_CASE or self.options & self.SHAPE_MIXED_CASE:
            is_lower = tf.logical_and(
                has_case,
                tf.equal(inputs, inputs_lower)
            )
        if self.options & self.SHAPE_LOWER_CASE:
            outputs_one.append(is_lower)

        if self.options & self.SHAPE_UPPER_CASE or self.options & self.SHAPE_MIXED_CASE:
            is_upper = tf.logical_and(
                has_case,
                tf.equal(inputs, inputs_upper)
            )
        if self.options & self.SHAPE_UPPER_CASE:
            outputs_one.append(is_upper)

        if self.options & self.SHAPE_TITLE_CASE or self.options & self.SHAPE_MIXED_CASE:
            inputs_title = title_case(inputs)
            is_title = tf.logical_and(
                has_case,
                tf.equal(inputs, inputs_title)
            )
        if self.options & self.SHAPE_TITLE_CASE:
            outputs_one.append(is_title)

        if self.options & self.SHAPE_MIXED_CASE:
            no_case = tf.logical_not(has_case)
            is_mixed = tf.logical_not(tf.logical_or(
                tf.logical_or(no_case, is_lower),
                tf.logical_or(is_upper, is_title)
            ))
            outputs_one.append(is_mixed)

        # Length
        if self.options & self.SHAPE_LENGTH_NORM:
            length_norm = tf.strings.length(inputs, unit='UTF8_CHAR')
            length_norm = (tf.cast(length_norm, self.compute_dtype) - self.mean_len) / self.std_len
            outputs_one.append(length_norm)

        # Same
        any_same = self.SHAPE_LEFT_SAME | self.SHAPE_RIGHT_SAME | self.SHAPE_LEFT2_SAME | self.SHAPE_RIGHT2_SAME
        if self.options & any_same:
            empty_pad = tf.zeros_like(inputs[..., :1])
            inputs_padded = tf.concat([empty_pad, empty_pad, inputs, empty_pad, empty_pad], axis=-1)

        if self.options & (self.SHAPE_LEFT_SAME | self.SHAPE_RIGHT_SAME):
            same_one = tf.equal(inputs_padded[..., 1:], inputs_padded[..., :-1])

        if self.options & self.SHAPE_LEFT_SAME:
            same_left = same_one[..., 1:-2]
            outputs_one.append(same_left)

        if self.options & self.SHAPE_RIGHT_SAME:
            same_right = same_one[..., 2:-1]
            outputs_one.append(same_right)

        if self.options & (self.SHAPE_LEFT2_SAME | self.SHAPE_RIGHT2_SAME):
            same_two = tf.equal(inputs_padded[..., 2:], inputs_padded[..., :-2])

        if self.options & self.SHAPE_LEFT2_SAME:
            same_left2 = same_two[..., :-2]
            outputs_one.append(same_left2)

        if self.options & self.SHAPE_RIGHT2_SAME:
            same_right2 = same_two[..., 2:]
            outputs_one.append(same_right2)

        # Char category
        if self.options & WordShape.SHAPE_CHAR_CAT_FIRST:
            first_cats = char_category(inputs)
            first_ids = self.cat_lookup(first_cats)
            first_feats = tf.one_hot(first_ids, depth=30)
            outputs_many.append(first_feats)

        if self.options & WordShape.SHAPE_CHAR_CAT_LAST:
            last_cats = char_category(inputs, first=False)
            last_ids = self.cat_lookup(last_cats)
            last_feats = tf.one_hot(last_ids, depth=30)
            outputs_many.append(last_feats)

        outputs_one = [tf.cast(o, self.compute_dtype) for o in outputs_one]
        outputs_many = [tf.cast(o, self.compute_dtype) for o in outputs_many]

        if not outputs_one:
            return tf.concat(outputs_many, axis=-1)

        outputs_one = tf.stack(outputs_one, axis=-1)
        if not outputs_many:
            return outputs_one

        return tf.concat([outputs_one, *outputs_many], axis=-1)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        units = 0
        options = [
            self.SHAPE_HAS_CASE, self.SHAPE_LOWER_CASE, self.SHAPE_UPPER_CASE, self.SHAPE_TITLE_CASE,
            self.SHAPE_MIXED_CASE, self.SHAPE_LENGTH_NORM, self.SHAPE_LEFT_SAME, self.SHAPE_RIGHT_SAME,
            self.SHAPE_LEFT2_SAME, self.SHAPE_RIGHT2_SAME]
        for opt in options:
            if self.options & opt:
                units += 1

        if self.options & WordShape.SHAPE_CHAR_CAT_FIRST:
            units += 30
        if self.options & WordShape.SHAPE_CHAR_CAT_LAST:
            units += 30

        return input_shape + (units,)

    def get_config(self):
        config = super().get_config()
        config.update({
            'options': self.options,
            'mean_len': self.mean_len,
            'std_len': self.std_len
        })

        return config
