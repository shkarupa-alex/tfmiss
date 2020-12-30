from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tfmiss.keras.layers.attention import SelfAttentionWithContext, MultiplicativeSelfAttention, AdditiveSelfAttention


@keras_parameterized.run_all_keras_modes
class SelfAttentionWithContextTest(keras_parameterized.TestCase):
    def setUp(self):
        super(SelfAttentionWithContextTest, self).setUp()
        self.default_policy = tf.keras.mixed_precision.global_policy()
        self.mf16_policy = tf.keras.mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(SelfAttentionWithContextTest, self).tearDown()
        tf.keras.mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            SelfAttentionWithContext,
            kwargs={},
            input_shape=(2, 10, 5),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5)
        )

        tf.keras.mixed_precision.set_global_policy(self.mf16_policy)
        testing_utils.layer_test(
            SelfAttentionWithContext,
            kwargs={},
            input_shape=(2, 10, 5),
            input_dtype='float16',
            expected_output_dtype='float16',
            expected_output_shape=(None, 5)
        )
        tf.keras.mixed_precision.set_global_policy(self.default_policy)


@keras_parameterized.run_all_keras_modes
class MultiplicativeSelfAttentionTest(keras_parameterized.TestCase):
    def setUp(self):
        super(MultiplicativeSelfAttentionTest, self).setUp()
        self.default_policy = tf.keras.mixed_precision.global_policy()
        self.mf16_policy = tf.keras.mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(MultiplicativeSelfAttentionTest, self).tearDown()
        tf.keras.mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            MultiplicativeSelfAttention,
            kwargs={},
            input_shape=(2, 10, 5),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 10, 5)
        )

        tf.keras.mixed_precision.set_global_policy(self.mf16_policy)
        testing_utils.layer_test(
            MultiplicativeSelfAttention,
            kwargs={'use_scale': True, 'causal': True, 'dropout': 0.1},
            input_shape=(2, 10, 5),
            input_dtype='float16',
            expected_output_dtype='float16',
            expected_output_shape=(None, 10, 5)
        )
        tf.keras.mixed_precision.set_global_policy(self.default_policy)


@keras_parameterized.run_all_keras_modes
class AdditiveSelfAttentionTest(keras_parameterized.TestCase):
    def setUp(self):
        super(AdditiveSelfAttentionTest, self).setUp()
        self.default_policy = tf.keras.mixed_precision.global_policy()
        self.mf16_policy = tf.keras.mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(AdditiveSelfAttentionTest, self).tearDown()
        tf.keras.mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            AdditiveSelfAttention,
            kwargs={},
            input_shape=(2, 10, 5),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 10, 5)
        )

        tf.keras.mixed_precision.set_global_policy(self.mf16_policy)
        testing_utils.layer_test(
            AdditiveSelfAttention,
            kwargs={'use_scale': True, 'causal': True, 'dropout': 0.1},
            input_shape=(2, 10, 5),
            input_dtype='float16',
            expected_output_dtype='float16',
            expected_output_shape=(None, 10, 5)
        )
        tf.keras.mixed_precision.set_global_policy(self.default_policy)


if __name__ == "__main__":
    tf.test.main()
