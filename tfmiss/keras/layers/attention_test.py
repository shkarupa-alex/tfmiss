from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from tfmiss.keras.layers.attention import SelfAttentionWithContext, MultiplicativeSelfAttention, AdditiveSelfAttention


@test_combinations.run_all_keras_modes
class SelfAttentionWithContextTest(test_combinations.TestCase):
    def setUp(self):
        super(SelfAttentionWithContextTest, self).setUp()
        self.default_policy = mixed_precision.global_policy()
        self.mf16_policy = mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(SelfAttentionWithContextTest, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            SelfAttentionWithContext,
            kwargs={},
            input_shape=(2, 10, 5),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5)
        )

        mixed_precision.set_global_policy(self.mf16_policy)
        test_utils.layer_test(
            SelfAttentionWithContext,
            kwargs={},
            input_shape=(2, 10, 5),
            input_dtype='float16',
            expected_output_dtype='float16',
            expected_output_shape=(None, 5)
        )
        mixed_precision.set_global_policy(self.default_policy)


@test_combinations.run_all_keras_modes
class MultiplicativeSelfAttentionTest(test_combinations.TestCase):
    def setUp(self):
        super(MultiplicativeSelfAttentionTest, self).setUp()
        self.default_policy = mixed_precision.global_policy()
        self.mf16_policy = mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(MultiplicativeSelfAttentionTest, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            MultiplicativeSelfAttention,
            kwargs={},
            input_shape=(2, 10, 5),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 10, 5)
        )

        mixed_precision.set_global_policy(self.mf16_policy)
        test_utils.layer_test(
            MultiplicativeSelfAttention,
            kwargs={'use_scale': True, 'dropout': 0.1},  # TODO: causal=True leads to test failure
            input_shape=(2, 10, 5),
            input_dtype='float16',
            expected_output_dtype='float16',
            expected_output_shape=(None, 10, 5)
        )
        mixed_precision.set_global_policy(self.default_policy)


@test_combinations.run_all_keras_modes
class AdditiveSelfAttentionTest(test_combinations.TestCase):
    def setUp(self):
        super(AdditiveSelfAttentionTest, self).setUp()
        self.default_policy = mixed_precision.global_policy()
        self.mf16_policy = mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(AdditiveSelfAttentionTest, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            AdditiveSelfAttention,
            kwargs={'units': 1},
            input_shape=(2, 10, 5),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 10, 1)
        )

        mixed_precision.set_global_policy(self.mf16_policy)
        test_utils.layer_test(
            AdditiveSelfAttention,
            kwargs={'units': 7, 'use_scale': True, 'dropout': 0.1},  # TODO: causal=True leads to test failure
            input_shape=(2, 10, 5),
            input_dtype='float16',
            expected_output_dtype='float16',
            expected_output_shape=(None, 10, 7)
        )
        mixed_precision.set_global_policy(self.default_policy)


if __name__ == "__main__":
    tf.test.main()
