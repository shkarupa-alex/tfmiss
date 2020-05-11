from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import testing_utils
from tensorflow.python.util import object_identity
from tensorflow.python.training.tracking import util as trackable_util
from tfmiss.keras.layers.tcn import TemporalBlock, TemporalConvNet


@keras_parameterized.run_all_keras_modes
class TemporalBlockTest(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            TemporalBlock,
            kwargs={
                'filters': 3,
                'kernel_size': 2,
                'dilation': 1,
                'dropout': 0.2,
            },
            input_shape=(2, 3, 7)
        )
        testing_utils.layer_test(
            TemporalBlock,
            kwargs={
                'filters': 3,
                'kernel_size': 2,
                'dilation': 1,
                'dropout': 0.2,
                'padding': 'same',
            },
            input_shape=(2, 3, 7)
        )

    def test_shape(self):
        layer = TemporalBlock(
            filters=3,
            kernel_size=2,
            dilation=1,
            dropout=0.2,
        )

        static_shape = layer.compute_output_shape(tf.TensorShape([None, None, 7])).as_list()
        self.assertAllEqual((None, None, 3), static_shape)

        out = layer(tf.keras.backend.variable(np.ones((1, 5, 7))))
        dynamic_shape = self.evaluate(out).shape
        self.assertAllEqual((1, 5, 3), dynamic_shape)

    def test_model(self):
        model = tf.keras.models.Sequential()
        model.add(TemporalBlock(
            filters=3,
            kernel_size=2,
            dilation=1,
            dropout=0.2,
            input_shape=(3, 4)
        ))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly(),
                      experimental_run_tf_function=testing_utils.should_run_tf_function())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 3)), epochs=1, batch_size=10)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


@keras_parameterized.run_all_keras_modes
class TemporalConvNetTest(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            TemporalConvNet,
            kwargs={
                'filters': [5, 4, 3],
                'kernel_size': 3,
                'dropout': 0.2,
            },
            input_shape=(2, 3, 7)
        )

        testing_utils.layer_test(
            TemporalConvNet,
            kwargs={
                'filters': [5, 4, 3],
                'kernel_size': 3,
                'dropout': 0.2,
                'padding': 'same',
            },
            input_shape=(2, 3, 7)
        )

    def test_shape(self):
        layer = TemporalConvNet(
            filters=[5, 4, 3],
            kernel_size=3,
            dropout=0.2
        )

        static_shape = layer.compute_output_shape(tf.TensorShape([None, None, 7])).as_list()
        self.assertAllEqual((None, None, 3), static_shape)

        out = layer(tf.keras.backend.variable(np.ones((1, 5, 7))))
        dynamic_shape = self.evaluate(out).shape
        self.assertAllEqual((1, 5, 3), dynamic_shape)

    def test_model(self):
        model = tf.keras.models.Sequential()
        model.add(TemporalConvNet(
            filters=[5, 4, 3],
            kernel_size=3,
            dropout=0.2
        ))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly(),
                      experimental_run_tf_function=testing_utils.should_run_tf_function())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 3)), epochs=1, batch_size=10)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)

    def test_model_with_ragged_input(self):
        inputs = tf.keras.layers.Input(shape=(None, 2), dtype=tf.float32, ragged=True)
        outputs = TemporalConvNet(
            filters=[5, 4, 3],
            kernel_size=3,
            dropout=0.2
        )(inputs)

        model = tf.keras.Model(inputs, outputs)
        model._experimental_run_tf_function = testing_utils.should_run_tf_function()
        model.run_eagerly = testing_utils.should_run_eagerly()

        data = tf.ragged.constant([
            [[.1, 1.], [.2, 2.], [.3, 3.]],
            [[.4, 4.]],
            [[.5, 5.], [.6, 6.]]
        ], ragged_rank=1)
        outputs = model.predict(data)
        self.assertAllClose(
            outputs,
            tf.ragged.constant([
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
            ], ragged_rank=1)
        )


if __name__ == "__main__":
    tf.test.main()
