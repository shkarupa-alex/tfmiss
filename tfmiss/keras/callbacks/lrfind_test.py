from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tfmiss.keras.callbacks.lrfind import LRFinder


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class LRFInderTest(keras_parameterized.TestCase):
    def testNoExceptions(self):
        lrf_cb = LRFinder(200, min_lr=1e-2, max_lr=10.)
        model = _get_model(['accuracy'], out_dim=1)
        x = np.random.rand(1000, 10)
        y = np.mean(x, axis=-1) + np.random.rand(1000) / 10.
        y = np.logical_and(np.less(y, 0.1), np.greater(y, 0.9)).astype(np.int32)
        model.fit(x, y, batch_size=5, callbacks=[lrf_cb])
        _ = lrf_cb.plot(5)
        lr = lrf_cb.find()
        self.assertAlmostEqual(lr, 9.351, places=3)


def _get_model(compile_metrics, out_dim):
    layers = [
        tf.keras.layers.Dense(3, activation='relu', kernel_initializer='ones'),
        tf.keras.layers.Dense(out_dim, activation='sigmoid', kernel_initializer='ones')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(10,))
    model.compile(
        loss='mae',
        metrics=compile_metrics,
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function()
    )

    return model


if __name__ == "__main__":
    tf.test.main()
