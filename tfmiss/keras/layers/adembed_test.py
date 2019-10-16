from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tfmiss.keras.layers.adembed import AdaptiveEmbedding


@keras_parameterized.run_all_keras_modes
class AdaptiveEmbeddingTest(keras_parameterized.TestCase):
    def testLayer(self):
        with tf.keras.utils.custom_object_scope({'AdaptiveEmbedding': AdaptiveEmbedding}):
            testing_utils.layer_test(
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
            testing_utils.layer_test(
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
            testing_utils.layer_test(
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
            testing_utils.layer_test(
                AdaptiveEmbedding,
                kwargs={
                    'cutoff': [50, 100],
                    'input_dim': 200,
                    'output_dim': 128,
                },
                input_shape=(2, 3, 7),
                input_dtype='int32',
                expected_output_dtype='float32',
                expected_output_shape=(None, 3, 7, 128)
            )
            testing_utils.layer_test(
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

    def testEmbeddingCorrectness(self):
        layer = AdaptiveEmbedding(cutoff=[1], output_dim=2, input_dim=2)
        model = tf.keras.models.Sequential([layer])
        layer.set_weights([
            np.array([[1, 1]]),
            np.array([[2]]),
            np.array([[1, 1], [1, 1]]),
            np.array([[3, 3]]),
        ])
        model.run_eagerly = testing_utils.should_run_eagerly()
        model._experimental_run_tf_function = testing_utils.should_run_tf_function()
        outputs = model.predict(np.array([[0, 1, 0]], dtype='int32'))
        self.assertAllClose(outputs, [[[2, 2], [6, 6], [2, 2]]])

    def testEagerGpuCpu(self):
        l = AdaptiveEmbedding(cutoff=[100], output_dim=2, input_dim=200)
        l.build((None, 2))
        inputs = tf.keras.backend.constant([[0, 1, 0]], dtype='int32')
        with tf.GradientTape() as tape:
            output = l(inputs)
        gs = tape.gradient(output, l.weights)
        opt = tf.keras.optimizers.Adagrad(0.1)
        opt.apply_gradients(zip(gs, l.weights))
        self.assertAllEqual(len(gs), 4)


if __name__ == "__main__":
    tf.test.main()
