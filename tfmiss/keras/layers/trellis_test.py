from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.training.tracking import object_identity
from tensorflow.python.training.tracking import util as trackable_util
from tfmiss.keras.layers.trellis import VariationalDropout, WeightShareConv1D


@keras_parameterized.run_all_keras_modes
class VariationalDropoutTest(keras_parameterized.TestCase):
    def testLayer(self):
        with tf.keras.utils.custom_object_scope({'VariationalDropout': VariationalDropout}):
            testing_utils.layer_test(
                VariationalDropout,
                kwargs={
                    'rate': 0.5,
                },
                input_shape=(2, 3, 7)
            )

    def testMaskReset(self):
        np.random.seed(1)
        source = np.random.random((2, 3, 4))

        drop = VariationalDropout(rate=0.5, seed=1)

        result1 = self.evaluate(drop(source, training=True))
        result2 = self.evaluate(drop(source, training=True, reset_mask=False))
        result3 = self.evaluate(drop(source, training=True))

        self.assertAllEqual(result1, result2)
        self.assertNotAllClose(result2, result3)

    def testModel(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(3, input_shape=(3, 4)))
        model.add(VariationalDropout(rate=0.5))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 3)), epochs=1, batch_size=10)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


# @keras_parameterized.run_all_keras_modes
# class WeightShareConv1DTest(keras_parameterized.TestCase):
#     def testLayer(self):
#         with tf.keras.utils.custom_object_scope({'WeightShareConv1D': WeightShareConv1D}):
#             testing_utils.layer_test(
#                 WeightShareConv1D,
#                 kwargs={
#                     'rate': 0.5,
#                 },
#                 input_shape=(2, 3, 7)
#             )

if __name__ == "__main__":
    tf.test.main()
