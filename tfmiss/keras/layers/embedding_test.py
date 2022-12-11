from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend, layers, models, optimizers
from keras.testing_infra import test_combinations, test_utils
from keras.mixed_precision import policy as mixed_precision
from tfmiss.keras.layers.embedding import AdaptiveEmbedding


@test_combinations.run_all_keras_modes
class AdaptiveEmbeddingTest(test_combinations.TestCase):
    def setUp(self):
        super(AdaptiveEmbeddingTest, self).setUp()
        self.default_policy = mixed_precision.global_policy()
        self.mf16_policy = mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(AdaptiveEmbeddingTest, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            AdaptiveEmbedding,
            kwargs={
                'cutoff': [50, 100],
                'input_dim': 200,
                'output_dim': 128,
            },
            input_shape=(2, 3),
            input_dtype='int32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 3, 128)
        )
        test_utils.layer_test(
            AdaptiveEmbedding,
            kwargs={
                'cutoff': [50, 100],
                'input_dim': 200,
                'output_dim': 128,
                'proj0': True,
            },
            input_shape=(2, 3),
            input_dtype='int32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 3, 128)
        )
        test_utils.layer_test(
            AdaptiveEmbedding,
            kwargs={
                'cutoff': [50, 100],
                'input_dim': 200,
                'output_dim': 128,
                'input_length': 3,
            },
            input_shape=(2, 3),
            input_dtype='int32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 3, 128)
        )
        test_utils.layer_test(
            AdaptiveEmbedding,
            kwargs={
                'cutoff': [50, 100],
                'input_dim': 200,
                'output_dim': 128,
                'mask_zero': True,
            },
            input_shape=(2, 3),
            input_dtype='int32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 3, 128)
        )
        test_utils.layer_test(
            AdaptiveEmbedding,
            kwargs={
                'cutoff': [50, 100],
                'input_dim': 200,
                'output_dim': 129,
            },
            input_shape=(2, 3, 7),
            input_dtype='int32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 3, 7, 129)
        )
        test_utils.layer_test(
            AdaptiveEmbedding,
            kwargs={
                'cutoff': [50, 100],
                'input_dim': 200,
                'output_dim': 128,
                'input_length': (None, 7)
            },
            input_shape=(2, 3, 7),
            input_dtype='int32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 3, 7, 128)
        )

        mixed_precision.set_global_policy(self.mf16_policy)
        test_utils.layer_test(
            AdaptiveEmbedding,
            kwargs={
                'cutoff': [50, 100],
                'input_dim': 200,
                'output_dim': 128,
            },
            input_shape=(2, 3),
            input_dtype='int32',
            expected_output_dtype='float16',
            expected_output_shape=(None, 3, 128)
        )
        mixed_precision.set_global_policy(self.default_policy)

    def test_embedding_correctness(self):
        layer = AdaptiveEmbedding(cutoff=[1], output_dim=16, input_dim=2, factor=2)
        model = models.Sequential([layer])
        layer.set_weights([
            np.array([[1] * 16]),
            np.array([[2] * 8]),
            # proj0 == False
            # np.array(...),
            np.array([[3] * 16] * 8),
        ])
        model.run_eagerly = test_utils.should_run_eagerly()
        outputs = model.predict(np.array([[0, 1, 0]], dtype='int32'))
        self.assertAllClose([[[1] * 16, [48] * 16, [1] * 16]], outputs)

    def test_eager_gpu_cpu(self):
        layer = AdaptiveEmbedding(cutoff=[100], output_dim=32, input_dim=200, proj0=True)
        layer.build((None, 2))
        inputs = backend.constant([[0, 1, 0]], dtype='int32')
        with tf.GradientTape() as tape:
            output = layer(inputs)
        gs = tape.gradient(output, layer.weights)
        opt = optimizers.adagrad_experimental.Adagrad()
        opt.apply_gradients(zip(gs, layer.weights))
        self.assertAllEqual(len(gs), 4)

    def test_embedding_with_ragged_input(self):
        layer = AdaptiveEmbedding(cutoff=[1], output_dim=16, input_dim=4, factor=2)
        data = tf.ragged.constant([
            [1., 2., 2.],
            [0.],
            [1., 2.]
        ], ragged_rank=1)
        layer(data)
        layer.set_weights([
            np.array([[1] * 16]),
            np.array([[2] * 8] * 3),
            # proj0 == False
            # np.array(...),
            np.array([[3] * 16] * 8),
        ])

        inputs = layers.Input(shape=(None,), dtype=tf.float32, ragged=True)
        outputs = layers.Lambda(lambda args: tf.identity(args))(inputs)
        outputs = layer(outputs)

        model = models.Model(inputs, outputs)
        model.run_eagerly = test_utils.should_run_eagerly()
        outputs = model.predict(data)
        self.assertAllClose(
            outputs,
            tf.ragged.constant([
                [[48.] * 16, [48.] * 16, [48.] * 16],
                [[1.] * 16],
                [[48.] * 16, [48.] * 16]
            ], ragged_rank=1)
        )


if __name__ == "__main__":
    tf.test.main()
