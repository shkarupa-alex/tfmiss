from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_keras import layers, models, utils
from tf_keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.util import object_identity
from tensorflow.python.checkpoint import checkpoint
from tfmiss.keras.layers.wrappers import MapFlat, WeightNorm, WithRagged


@test_combinations.run_all_keras_modes
class MapFlatTest(test_combinations.TestCase):
    def test_layer(self):
        class Stack2(layers.Layer):
            def call(self, inputs, *args, **kwargs):
                return tf.stack([inputs, inputs], axis=-1)

        with utils.custom_object_scope({'Stack2': Stack2}):
            test_utils.layer_test(
                MapFlat,
                kwargs={'layer': Stack2()},
                input_shape=(3, 10),
                input_dtype='float32',
                expected_output_dtype='float32',
                expected_output_shape=(None, 10, 2)
            )


@test_combinations.run_all_keras_modes
class WithRaggedTest(test_combinations.TestCase):
    def test_layer(self):
        inputs = tf.ragged.constant([
            [[1., 2.], [2., 3.], [2., 5.]],
            [[0., 9.]],
            [[1., 1.], [2., 9.]]
        ], ragged_rank=1)
        outputs = WithRagged(layers.Dense(4))(inputs)
        self.assertIsInstance(outputs, tf.RaggedTensor)

        outputs = self.evaluate(outputs)
        self.assertLen(outputs.shape, 3)
        self.assertEqual(outputs.shape[-1], 4)

    def test_model(self):
        logits = tf.ragged.constant([
            [[1., 2.], [2., 3.], [2., 5.]],
            [[0., 9.]],
            [[1., 1.], [2., 9.]]
        ], ragged_rank=1)

        inputs = layers.Input(shape=(None, 2), dtype=tf.float32, ragged=True)
        outputs = WithRagged(layers.Dense(3, activation='sigmoid'))(inputs)
        model = models.Model(inputs=inputs, outputs=outputs)
        model.compile(run_eagerly=test_utils.should_run_eagerly())
        model.predict(logits)


@test_combinations.run_all_keras_modes
class WeightNormTest(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            WeightNorm,
            kwargs={'layer': layers.Dense(1)},
            input_shape=(3, 7)
        )
        test_utils.layer_test(
            WeightNorm,
            kwargs={'layer': layers.Conv2D(5, (2, 2))},
            input_shape=(2, 4, 4, 3)
        )

    def test_double_wrap(self):
        layer = layers.Dense(7)
        WeightNorm(layer)
        with self.assertRaisesRegexp(ValueError, 'Weight normalization already applied'):
            WeightNorm(layer)

    def test_no_kernel(self):
        with self.assertRaisesRegexp(ValueError, 'Weights with name .* not found in layer'):
            WeightNorm(layers.MaxPooling2D(2, 2)).build((2, 2))

    def test_dense(self):
        inputs = tf.random.normal([3, 5])
        layer = layers.Dense(7)
        wrapper = WeightNorm(layer)
        original = self.evaluate(layer(inputs))
        weighted = self.evaluate(wrapper(inputs))

        self.assertAllClose(original, weighted)
        self.assertNotEqual(original.tolist(), weighted.tolist())

    def test_conv(self):
        inputs = tf.random.normal([3, 5, 7, 9])
        layer = layers.Conv2D(2, 4)
        wrapper = WeightNorm(layer)
        original = self.evaluate(layer(inputs))
        weighted = self.evaluate(wrapper(inputs))

        self.assertAllClose(original, weighted)
        self.assertNotEqual(original.tolist(), weighted.tolist())

    def test_vars_and_shapes(self):
        inputs = tf.random.normal([3, 5])
        layer = layers.Dense(7)
        wrapper = WeightNorm(layer)
        result = wrapper(inputs)
        self.evaluate(result)

        self.assertListEqual(wrapper.kernel_v.shape.as_list(), layer.kernel.shape.as_list())
        self.assertListEqual(wrapper.kernel_g.shape.as_list(), [7])

    def test_weight_norm_dense(self):
        model = models.Sequential()
        model.add(WeightNorm(layers.Dense(2), input_shape=(3, 4)))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 2)), epochs=1, batch_size=10)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(checkpoint.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_weight_norm_stacked(self):
        model = models.Sequential()
        model.add(WeightNorm(layers.Dense(2), input_shape=(3, 4)))
        model.add(WeightNorm(layers.Dense(3)))
        model.add(layers.Activation('relu'))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 3)), epochs=1, batch_size=10)

    def test_regularizers(self):
        model = models.Sequential()
        model.add(WeightNorm(layers.Dense(2, kernel_regularizer='l1'), input_shape=(3, 4)))
        model.add(layers.Activation('relu'))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())
        self.assertEqual(len(model.losses), 1)


if __name__ == "__main__":
    tf.test.main()
