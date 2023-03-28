from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras.testing_infra import test_combinations, test_utils
from tfmiss.keras.optimizers.schedules import WarmHoldCoolAnnihilateScheduler, WarmHoldCosineCoolAnnihilateScheduler


@test_combinations.run_all_keras_modes
class WarmHoldCoolAnnihilateSchedulerTest(test_combinations.TestCase):
    def test_values(self):
        schedule = WarmHoldCoolAnnihilateScheduler(
            min_lr=1.0, max_lr=100., warm_steps=3, hold_steps=3, cool_steps=3, annih_steps=3)
        lrs = [self.evaluate(schedule(i)).item() for i in range(15)]
        self.assertAllClose([1.0, 34.0, 67.0,
                             100.0, 100.0, 100.0, 100.0,
                             67.0, 34.0, 1.0,
                             0.6699999, 0.3399999, 0.0099999,
                             0.0099999, 0.0099999], lrs)

    def test_model(self):
        schedule = WarmHoldCoolAnnihilateScheduler(
            min_lr=1.0, max_lr=100., warm_steps=3, hold_steps=3, cool_steps=3, annih_steps=3)
        optimizer = optimizers.adam.Adam(schedule)
        model = models.Sequential()
        model.add(layers.Dense(3))
        model.compile(optimizer=optimizer, loss='mse', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 3)), epochs=2, batch_size=10)
        model.get_config()

    def test_config(self):
        schedule = WarmHoldCoolAnnihilateScheduler(
            min_lr=1.0, max_lr=100., warm_steps=3, hold_steps=3, cool_steps=3, annih_steps=3)
        WarmHoldCoolAnnihilateScheduler.from_config(schedule.get_config())


@test_combinations.run_all_keras_modes
class WarmHoldCosineCoolAnnihilateSchedulerTest(test_combinations.TestCase):
    def test_values(self):
        schedule = WarmHoldCosineCoolAnnihilateScheduler(
            min_lr=1.0, max_lr=100., warm_steps=3, hold_steps=3, cool_steps=9, annih_steps=3, cosine_cycles=2)
        lrs = [self.evaluate(schedule(i)).item() for i in range(20)]
        self.assertAllClose([1.0, 34.0, 67.0,
                             100.0, 100.0, 100.0, 100.0,
                             75.2504501, 25.7508965, 1.0,
                             84.1318970, 67.8260040, 45.5513992,
                             23.2764149, 6.9694981, 1.0,
                             0.66999995, 0.33999997, 0.0099999, 0.0099999], lrs)

    def test_model(self):
        schedule = WarmHoldCosineCoolAnnihilateScheduler(
            min_lr=1.0, max_lr=100., warm_steps=3, hold_steps=3, cool_steps=9, annih_steps=3, cosine_cycles=2)
        optimizer = optimizers.adam.Adam(schedule)
        model = models.Sequential()
        model.add(layers.Dense(3))
        model.compile(optimizer=optimizer, loss='mse', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 3)), epochs=20, batch_size=10)
        model.get_config()

    def test_config(self):
        schedule = WarmHoldCosineCoolAnnihilateScheduler(
            min_lr=1.0, max_lr=100., warm_steps=3, hold_steps=3, cool_steps=9, annih_steps=3, cosine_cycles=2)
        WarmHoldCosineCoolAnnihilateScheduler.from_config(schedule.get_config())


if __name__ == "__main__":
    tf.test.main()
