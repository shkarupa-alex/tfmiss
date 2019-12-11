from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.util import object_identity
from tensorflow.python.training.tracking import util as trackable_util
from tfmiss.keras.layers.wrappers import WeightNorm


@keras_parameterized.run_all_keras_modes
class WeightNormTest(keras_parameterized.TestCase):
    def test_layer(self):
        with tf.keras.utils.custom_object_scope({'WeightNorm': WeightNorm}):
            testing_utils.layer_test(
                WeightNorm,
                kwargs={'layer': tf.keras.layers.Dense(1)},
                input_shape=(3, 7)
            )

    def test_double_wrap(self):
        layer = tf.keras.layers.Dense(7)
        WeightNorm(layer)
        with self.assertRaisesRegexp(ValueError, 'Weight normalization already applied'):
            WeightNorm(layer)

    def test_no_kernel(self):
        with self.assertRaisesRegexp(ValueError, 'Weights with name .* not found in layer'):
            WeightNorm(tf.keras.layers.MaxPooling2D(2, 2)).build((2, 2))

    def test_dense(self):
        inputs = tf.random.normal([3, 5])
        layer = tf.keras.layers.Dense(7)
        wrapper = WeightNorm(layer)
        original = self.evaluate(layer(inputs))
        weighted = self.evaluate(wrapper(inputs))

        self.assertAllClose(original, weighted)
        self.assertNotEqual(original.tolist(), weighted.tolist())

    def test_conv(self):
        inputs = tf.random.normal([3, 5, 7, 9])
        layer = tf.keras.layers.Conv2D(2, 4)
        wrapper = WeightNorm(layer)
        original = self.evaluate(layer(inputs))
        weighted = self.evaluate(wrapper(inputs))

        self.assertAllClose(original, weighted)
        self.assertNotEqual(original.tolist(), weighted.tolist())

    def test_vars_and_shapes(self):
        inputs = tf.random.normal([3, 5])
        layer = tf.keras.layers.Dense(7)
        wrapper = WeightNorm(layer)
        result = wrapper(inputs)
        self.evaluate(result)

        self.assertListEqual(wrapper.kernel_v.shape.as_list(), layer.kernel.shape.as_list())
        self.assertListEqual(wrapper.kernel_g.shape.as_list(), [7])

    def test_weight_norm_dense(self):
        model = tf.keras.models.Sequential()
        model.add(WeightNorm(tf.keras.layers.Dense(2), input_shape=(3, 4)))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly(),
                      experimental_run_tf_function=testing_utils.should_run_tf_function())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 2)), epochs=1, batch_size=10)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_weight_norm_stacked(self):
        model = tf.keras.models.Sequential()
        model.add(WeightNorm(tf.keras.layers.Dense(2), input_shape=(3, 4)))
        model.add(WeightNorm(tf.keras.layers.Dense(3)))
        model.add(tf.keras.layers.Activation('relu'))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly(),
                      experimental_run_tf_function=testing_utils.should_run_tf_function())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 3)), epochs=1, batch_size=10)

    def test_regularizers(self):
        model = tf.keras.models.Sequential()
        model.add(WeightNorm(tf.keras.layers.Dense(2, kernel_regularizer='l1'), input_shape=(3, 4)))
        model.add(tf.keras.layers.Activation('relu'))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly(),
                      experimental_run_tf_function=testing_utils.should_run_tf_function())
        self.assertEqual(len(model.losses), 1)

    def test_estimator(self):
        def _input_fn():
            x = np.random.rand(100, 3)
            y = np.random.rand(100) > 0.5
            dataset = tf.data.Dataset.from_tensor_slices((x, y))

            return dataset.batch(4)

        model = tf.keras.models.Sequential()
        model.add(WeightNorm(tf.keras.layers.Dense(2, kernel_regularizer='l1'), input_shape=(3,)))
        model.add(tf.keras.layers.Dense(1))
        model.add(tf.keras.layers.Activation('relu'))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly(),
                      experimental_run_tf_function=testing_utils.should_run_tf_function())
        model.fit(_input_fn(), steps_per_epoch=5)
        estimator = tf.keras.estimator.model_to_estimator(
            keras_model=model
        )
        estimator.train(input_fn=_input_fn, steps=5)


if __name__ == "__main__":
    tf.test.main()
