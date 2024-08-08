import tensorflow as tf
from keras.src import layers
from keras.src import testing

from tfmiss.keras.layers.wrappers import MapFlat
from tfmiss.keras.layers.wrappers import WithRagged


class MapFlatTest(testing.TestCase):
    def test_layer(self):
        class Stack2(layers.Layer):
            def call(self, inputs, *args, **kwargs):
                return tf.stack([inputs, inputs], axis=-1)

            def compute_output_shape(self, input_shape):
                return input_shape + (2,)

        self.run_layer_test(
            MapFlat,
            init_kwargs={"layer": Stack2()},
            input_shape=(3, 10),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(3, 10, 2),
            custom_objects={"Stack2": Stack2},
            run_mixed_precision_check=False,
        )


class WithRaggedTest(testing.TestCase):
    def test_layer(self):
        inputs = tf.ragged.constant(
            [
                [[1.0, 2.0], [2.0, 3.0], [2.0, 5.0]],
                [[0.0, 9.0]],
                [[1.0, 1.0], [2.0, 9.0]],
            ],
            ragged_rank=1,
        )
        outputs = WithRagged(layers.Dense(4))(inputs)
        self.assertIsInstance(outputs, tf.RaggedTensor)

        self.assertLen(outputs.shape, 3)
        self.assertEqual(outputs.shape[-1], 4)

    # TODO https://github.com/keras-team/keras/issues/19646
    # def test_model(self):
    #     logits = tf.ragged.constant([
    #         [[1., 2.], [2., 3.], [2., 5.]],
    #         [[0., 9.]],
    #         [[1., 1.], [2., 9.]]
    #     ], ragged_rank=1)
    #
    #     inputs = layers.Input(shape=(None, 2), dtype=tf.float32, ragged=True)
    #     outputs = WithRagged(layers.Dense(3, activation='sigmoid'))(inputs)
    #     model = models.Model(inputs=inputs, outputs=outputs)
    #     model.compile()
    #     model.predict(logits)


if __name__ == "__main__":
    tf.test.main()
