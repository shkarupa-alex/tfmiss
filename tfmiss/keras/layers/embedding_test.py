import numpy as np
import tensorflow as tf
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.dtype_policies import dtype_policy

from tfmiss.keras.layers.embedding import AdaptiveEmbedding


class AdaptiveEmbeddingTest(testing.TestCase):
    def setUp(self):
        super(AdaptiveEmbeddingTest, self).setUp()
        self.default_policy = dtype_policy.dtype_policy()

    def tearDown(self):
        super(AdaptiveEmbeddingTest, self).tearDown()
        dtype_policy.set_dtype_policy(self.default_policy)

    def test_layer(self):
        self.run_layer_test(
            AdaptiveEmbedding,
            init_kwargs={
                "cutoff": [50, 100],
                "input_dim": 200,
                "output_dim": 128,
            },
            input_shape=(2, 3),
            input_dtype="int32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 128),
        )
        self.run_layer_test(
            AdaptiveEmbedding,
            init_kwargs={
                "cutoff": [50, 100],
                "input_dim": 200,
                "output_dim": 128,
                "proj0": True,
            },
            input_shape=(2, 3),
            input_dtype="int32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 128),
        )
        self.run_layer_test(
            AdaptiveEmbedding,
            init_kwargs={
                "cutoff": [50, 100],
                "input_dim": 200,
                "output_dim": 128,
                "mask_zero": True,
            },
            input_shape=(2, 3),
            input_dtype="int32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 128),
        )
        self.run_layer_test(
            AdaptiveEmbedding,
            init_kwargs={
                "cutoff": [50, 100],
                "input_dim": 200,
                "output_dim": 129,
            },
            input_shape=(2, 3, 7),
            input_dtype="int32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 7, 129),
        )

    def test_layer_fp16(self):
        dtype_policy.set_dtype_policy("mixed_float16")
        self.run_layer_test(
            AdaptiveEmbedding,
            init_kwargs={
                "cutoff": [50, 100],
                "input_dim": 200,
                "output_dim": 128,
            },
            input_shape=(2, 3),
            input_dtype="int32",
            expected_output_dtype="float16",
            expected_output_shape=(2, 3, 128),
        )

    def test_embedding_correctness(self):
        layer = AdaptiveEmbedding(
            cutoff=[1], output_dim=16, input_dim=2, factor=2
        )
        layer.build(tuple())
        layer.set_weights(
            [
                np.array([[1] * 16]),
                np.array([[2] * 8]),
                # proj0 == False
                # np.array(...),
                np.array([[3] * 16] * 8),
            ]
        )
        model = models.Sequential([layer])
        outputs = model.predict(np.array([[0, 1, 0]], dtype="int32"))
        self.assertAllClose([[[1] * 16, [48] * 16, [1] * 16]], outputs)

    def test_eager_gpu_cpu(self):
        layer = AdaptiveEmbedding(
            cutoff=[100], output_dim=32, input_dim=200, proj0=True
        )
        layer.build((None, 2))
        inputs = tf.constant([[0, 1, 0]], dtype="int32")
        with tf.GradientTape() as tape:
            output = layer(inputs)
        gs = tape.gradient(output, layer.weights)
        opt = optimizers.Adagrad()
        opt.apply_gradients(zip(gs, layer.weights))
        self.assertEqual(len(gs), 4)

    # TODO: https://github.com/keras-team/keras/issues/18414
    # def test_embedding_with_ragged_input(self):
    #     layer = AdaptiveEmbedding(
    #     cutoff=[1], output_dim=16, input_dim=4, factor=2)
    #     data = tf.ragged.constant([
    #         [1., 2., 2.],
    #         [0.],
    #         [1., 2.]
    #     ], ragged_rank=1)
    #     layer(data)
    #     layer.set_weights([
    #         np.array([[1] * 16]),
    #         np.array([[2] * 8] * 3),
    #         # proj0 == False
    #         # np.array(...),
    #         np.array([[3] * 16] * 8),
    #     ])
    #
    #     inputs = layers.Input(shape=(None,), dtype=tf.float32, ragged=True)
    #     outputs = layers.Lambda(lambda args: tf.identity(args))(inputs)
    #     outputs = layer(outputs)
    #
    #     model = models.Model(inputs, outputs)
    #     outputs = model.predict(data)
    #     self.assertAllClose(
    #         outputs,
    #         tf.ragged.constant([
    #             [[48.] * 16, [48.] * 16, [48.] * 16],
    #             [[1.] * 16],
    #             [[48.] * 16, [48.] * 16]
    #         ], ragged_rank=1)
    #     )


if __name__ == "__main__":
    tf.test.main()
