from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_keras import layers, models
from tf_keras.src.testing_infra import test_combinations, test_utils
from tfmiss.keras.layers.scale import L2Scale
from tfmiss.keras.layers.wrappers import WithRagged


@test_combinations.run_all_keras_modes
class L2ScaleTest(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            L2Scale,
            kwargs={'alpha': 10},
            input_shape=(10, 5),
            input_dtype='float32',
            expected_output_dtype='float32'
        )
        test_utils.layer_test(
            L2Scale,
            kwargs={'alpha': 30, 'dtype': 'float16'},
            input_shape=(2, 10, 5),
            input_dtype='float16',
            expected_output_dtype='float16'
        )

    def test_with_ragged(self):
        layer = WithRagged(L2Scale(alpha=10))
        logits = tf.ragged.constant([
            [[1., 2.], [2., 3.], [2., 5.]],
            [[0., 9.]],
            [[1., 1.], [2., 9.]]
        ], ragged_rank=1)
        inputs = layers.Input(shape=(None, 2), dtype=tf.float32)
        outputs = layer(inputs)

        model = models.Model(inputs=inputs, outputs=outputs)
        model.run_eagerly = test_utils.should_run_eagerly()
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
