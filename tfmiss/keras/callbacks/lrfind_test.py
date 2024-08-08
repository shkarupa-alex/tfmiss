import os

import numpy as np
import tensorflow as tf
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src import utils

from tfmiss.keras.callbacks.lrfind import LRFinder


class LRFInderTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        utils.set_random_seed(87654321)

    def test_no_exceptions(self):
        model = models.Sequential(
            [
                layers.Dense(3, activation="relu"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(loss="mae", optimizer="rmsprop")

        x = np.random.rand(1000, 10)
        y = np.mean(x, axis=-1) + np.random.rand(1000) / 10.0
        y = np.logical_and(np.less(y, 0.1), np.greater(y, 0.9)).astype(np.int32)
        lrf_cb = LRFinder(200, min_lr=1e-2, max_lr=10.0)
        model.fit(x, y, batch_size=5, callbacks=[lrf_cb])
        best_lr, loss_graph = lrf_cb.plot()

        self.assertAlmostEqual(best_lr, 2.72e-02, decimal=4)
        self.assertTrue(os.path.exists(loss_graph))


if __name__ == "__main__":
    tf.test.main()
