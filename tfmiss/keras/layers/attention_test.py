from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tfmiss.keras.layers.attention import AttentionWithContext


@keras_parameterized.run_all_keras_modes
class AttentionWithContextTest(keras_parameterized.TestCase):
    def setUp(self):
        super(AttentionWithContextTest, self).setUp()
        self.default_policy = tf.keras.mixed_precision.global_policy()
        self.mf16_policy = tf.keras.mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(AttentionWithContextTest, self).tearDown()
        tf.keras.mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        testing_utils.layer_test(
            AttentionWithContext,
            kwargs={},
            input_shape=(2, 10, 5),
            input_dtype='float32',
            expected_output_dtype='float32'
        )

        tf.keras.mixed_precision.set_global_policy(self.mf16_policy)
        testing_utils.layer_test(
            AttentionWithContext,
            kwargs={},
            input_shape=(2, 10, 5),
            input_dtype='float16',
            expected_output_dtype='float16'
        )
        tf.keras.mixed_precision.set_global_policy(self.default_policy)


if __name__ == "__main__":
    tf.test.main()
