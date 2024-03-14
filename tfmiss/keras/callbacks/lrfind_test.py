import os
import numpy as np
import tensorflow as tf
from keras import layers, models, utils
from keras.src import testing
from tfmiss.keras.callbacks.lrfind import LRFinder


class LRFInderTest(testing.TestCase):
    def test_no_exceptions(self):
        utils.set_random_seed(87654321)

        model = models.Sequential([
            layers.Dense(3, activation='relu', kernel_initializer='ones'),
            layers.Dense(1, activation='sigmoid', kernel_initializer='ones')
        ])
        model.compile(loss='mae', optimizer='rmsprop', run_eagerly=True)

        x = np.random.rand(1000, 10)
        y = np.mean(x, axis=-1) + np.random.rand(1000) / 10.
        y = np.logical_and(np.less(y, 0.1), np.greater(y, 0.9)).astype(np.int32)
        lrf_cb = LRFinder(200, min_lr=1e-2, max_lr=10.)
        model.fit(x, y, batch_size=5, callbacks=[lrf_cb])
        best_lr, loss_graph = lrf_cb.plot()

        self.assertAlmostEqual(best_lr, 0.3273407, decimal=7)
        self.assertTrue(os.path.exists(loss_graph))


if __name__ == "__main__":
    tf.test.main()
