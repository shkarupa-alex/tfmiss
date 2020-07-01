from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tfmiss.keras.layers.scale import L2Scale


@keras_parameterized.run_all_keras_modes
class L2ScaleTest(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            L2Scale,
            kwargs={'alpha': 10},
            input_shape=(10, 5),
            input_dtype='float32',
            expected_output_dtype='float32'
        )
        testing_utils.layer_test(
            L2Scale,
            kwargs={'alpha': 30, 'dtype': 'float16'},
            input_shape=(2, 10, 5),
            input_dtype='float16',
            expected_output_dtype='float16'
        )

    def test_ragged_input(self):
        layer = L2Scale(alpha=10)
        logits = tf.ragged.constant([
            [[1., 2.], [2., 3.], [2., 5.]],
            [[0., 9.]],
            [[1., 1.], [2., 9.]]
        ], ragged_rank=1)
        inputs = tf.keras.layers.Input(shape=(None, 2), dtype=tf.float32, ragged=True)
        outputs = layer(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.run_eagerly = testing_utils.should_run_eagerly()
        predictions = model.predict(logits)
        self.assertAllClose(
            predictions,
            tf.ragged.constant([
                [
                    [4.4721360206604, 8.9442720413208],
                    [5.547001838684082, 8.320503234863281],
                    [3.713906764984131, 9.284767150878906]
                ],
                [[0.0, 10.0]],
                [
                    [7.071067810058594, 7.071067810058594],
                    [2.169304609298706, 9.761870384216309]
                ]
            ], ragged_rank=1)
        )


if __name__ == "__main__":
    tf.test.main()
