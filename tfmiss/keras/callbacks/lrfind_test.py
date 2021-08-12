from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from keras import layers, keras_parameterized, testing_utils
from tfmiss.keras.callbacks.lrfind import LRFinder


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class LRFInderTest(keras_parameterized.TestCase):
    def test_no_exceptions(self):
        model = testing_utils.get_model_from_layers([
            layers.Dense(3, activation='relu', kernel_initializer='ones'),
            layers.Dense(1, activation='sigmoid', kernel_initializer='ones')
        ], input_shape=(10,))
        model.compile(loss='mae', optimizer='rmsprop', run_eagerly=testing_utils.should_run_eagerly())

        x = np.random.rand(1000, 10)
        y = np.mean(x, axis=-1) + np.random.rand(1000) / 10.
        y = np.logical_and(np.less(y, 0.1), np.greater(y, 0.9)).astype(np.int32)
        lrf_cb = LRFinder(200, min_lr=1e-2, max_lr=10.)
        model.fit(x, y, batch_size=5, callbacks=[lrf_cb])
        best_lr, loss_graph = lrf_cb.plot()

        self.assertAlmostEqual(best_lr, 0.033496544, places=7)
        self.assertTrue(os.path.exists(loss_graph))


if __name__ == "__main__":
    tf.test.main()
