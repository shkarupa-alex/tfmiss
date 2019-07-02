from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tfmiss.keras.seq2seq.transformer.position import PositionalEncoding


@keras_parameterized.run_all_keras_modes
class PositionalEncodingTest(keras_parameterized.TestCase):
    def testLayer(self):
        with tf.keras.utils.custom_object_scope({'PositionalEncoding': PositionalEncoding}):
            testing_utils.layer_test(
                PositionalEncoding,
                kwargs={
                    'max_length': 1000,
                },
                input_shape=(32, 64, 4)  # Odd
            )

            testing_utils.layer_test(
                PositionalEncoding,
                kwargs={
                    'max_length': 1000,
                },
                input_shape=(32, 64, 5)  # Even
            )

    def testMaxLength(self):
        with tf.keras.utils.custom_object_scope({'PositionalEncoding': PositionalEncoding}):
            with self.assertRaisesRegex(
                    tf.errors.InvalidArgumentError,
                    'Inputs length should be less then reserved'):
                testing_utils.layer_test(
                    PositionalEncoding,
                    kwargs={
                        'max_length': 10,
                    },
                    input_shape=(32, 64, 5)
                )


if __name__ == "__main__":
    tf.test.main()
