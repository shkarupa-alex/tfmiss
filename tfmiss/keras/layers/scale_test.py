import tensorflow as tf
from keras.src import layers
from keras.src import models
from keras.src import testing

from tfmiss.keras.layers.scale import L2Scale
from tfmiss.keras.layers.wrappers import WithRagged


class L2ScaleTest(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            L2Scale,
            init_kwargs={"alpha": 10},
            input_shape=(10, 5),
            input_dtype="float32",
            expected_output_dtype="float32",
        )
        self.run_layer_test(
            L2Scale,
            init_kwargs={"alpha": 30, "dtype": "float16"},
            input_shape=(2, 10, 5),
            input_dtype="float16",
            expected_output_dtype="float16",
        )

    def test_with_ragged(self):
        layer = WithRagged(L2Scale(alpha=10))
        logits = tf.ragged.constant(
            [
                [[1.0, 2.0], [2.0, 3.0], [2.0, 5.0]],
                [[0.0, 9.0]],
                [[1.0, 1.0], [2.0, 9.0]],
            ],
            ragged_rank=1,
        )
        inputs = layers.Input(shape=(None, 2), dtype=tf.float32)
        outputs = layer(inputs)

        model = models.Model(inputs=inputs, outputs=outputs)
        predictions = model.predict(logits)
        self.assertAllClose(
            predictions,
            tf.ragged.constant(
                [
                    [
                        [4.4721360206604, 8.9442720413208],
                        [5.547001838684082, 8.320503234863281],
                        [3.713906764984131, 9.284767150878906],
                    ],
                    [[0.0, 10.0]],
                    [
                        [7.071067810058594, 7.071067810058594],
                        [2.169304609298706, 9.761870384216309],
                    ],
                ],
                ragged_rank=1,
            ),
        )


if __name__ == "__main__":
    tf.test.main()
