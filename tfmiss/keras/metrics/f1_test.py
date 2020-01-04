from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.ops import variables
from tfmiss.keras.metrics.f1 import F1Binary


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class F1BinaryTest(keras_parameterized.TestCase):
    def test_config(self):
        r_obj = F1Binary(name='my_f1binary')
        self.assertEqual(r_obj.name, 'my_f1binary')
        self.assertEqual(len(r_obj.variables), 4)
        self.assertEqual([v.name for v in r_obj.variables],
                         ['true_positives:0', 'false_positives:0', 'true_positives:0', 'false_negatives:0'])

        # Check save and restore config
        r_obj2 = F1Binary.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, 'my_f1binary')
        self.assertEqual(len(r_obj2.variables), 4)

    def test_value_is_idempotent(self):
        r_obj = F1Binary()
        y_pred = tf.random.uniform(shape=(10, 1), dtype=tf.float32)
        y_true = tf.random.uniform(shape=(10,), dtype=tf.int32, maxval=2)
        update_op = r_obj.update_state(y_true, y_pred)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        # Run several updates.
        for _ in range(10):
            self.evaluate(update_op)

        # Then verify idempotency.
        initial_f1binary = self.evaluate(r_obj.result())
        for _ in range(10):
            self.assertAlmostEqual(initial_f1binary, self.evaluate(r_obj.result()))

    def test_unweighted(self):
        r_obj = F1Binary()
        y_pred = tf.constant([[0.9], [0.1], [0.9], [0.1]], shape=(1, 4))
        y_true = tf.constant([0, 1, 1, 0], shape=(1, 4))
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.5, self.evaluate(result))

    def test_unweighted_all_incorrect(self):
        r_obj = F1Binary()
        y_pred = tf.constant([[[0.9], [0.1], [0.9], [0.1]], [[0.1], [0.9], [0.1], [0.9]]])
        y_true = tf.constant([[0, 1, 0, 1], [1, 0, 1, 0]])
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0, self.evaluate(result))

    def test_weighted(self):
        r_obj = F1Binary()
        y_pred = tf.constant([[[0.9], [0.1], [0.9], [0.1]], [[0.1], [0.9], [0.1], [0.9]]])
        y_true = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred, sample_weight=tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))
        self.assertAlmostEqual(0.44444448, self.evaluate(result))

    def test_div_by_zero(self):
        r_obj = F1Binary()
        y_pred = tf.constant([[0.1], [0.1], [0.1], [0.1]])
        y_true = tf.constant([0, 0, 0, 0])
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertEqual(0, self.evaluate(result))

    def test_multiple_updates(self):
        r_obj = F1Binary()
        y_true = tf.constant([[0, 1], [1, 0]], shape=(2, 2))
        y_pred = tf.constant([[[0.9], [0.1]], [[0.9], [0.1]]], shape=(2, 2, 1), dtype=tf.float32)
        weights = tf.constant([[1, 4], [3, 2]], shape=(2, 2), dtype=tf.float32)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        update_op = r_obj.update_state(y_true, y_pred, sample_weight=weights)
        for _ in range(2):
            self.evaluate(update_op)

        self.assertAlmostEqual(0.5454545, self.evaluate(r_obj.result()))

    def test_all_true(self):
        r_obj = F1Binary()
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant([[0.1], [0.9], [0.1], [0.9], [0.1], [0.9], [0.1], [0.9], [0.1], [0.9]], shape=(1, 10, 1))
        y_true = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], shape=(1, 10, 1))
        result = r_obj(y_true, y_pred)
        self.assertEqual(1.0, self.evaluate(result))

    def test_known_result(self):
        r_obj = F1Binary()
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant([[0.9], [0.1], [0.1], [0.9], [0.1], [0.9], [0.9], [0.9]])
        y_true = tf.constant([1, 1, 0, 0, 0, 1, 0, 1])
        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.6666667, self.evaluate(result))

    def test_reset_states(self):
        f1_obj = F1Binary()
        layers = [
            tf.keras.layers.Dense(3, activation='relu', kernel_initializer='ones'),
            tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='ones')
        ]
        model = testing_utils.get_model_from_layers(layers, input_shape=(4,))
        model.compile(
            loss='mae',
            metrics=[f1_obj],
            optimizer='rmsprop',
        )
        model.run_eagerly = testing_utils.should_run_eagerly()
        model._experimental_run_tf_function = testing_utils.should_run_tf_function()
        x = np.ones((100, 4))
        y = np.zeros((100, 1), dtype=np.int32)

        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.precision.true_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.precision.false_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.recall.true_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.recall.false_negatives), 0.)

        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.precision.true_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.precision.false_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.recall.true_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.recall.false_negatives), 0.)

    def test_metric_rises(self):
        f1_obj = F1Binary()
        layers = [
            tf.keras.layers.Dense(3, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ]
        model = testing_utils.get_model_from_layers(layers, input_shape=(4,))
        model.compile(
            loss='mae',
            metrics=[f1_obj],
            optimizer=tf.keras.optimizers.Adam(0.01),
        )
        model.run_eagerly = True
        model._experimental_run_tf_function = False
        model.run_eagerly = testing_utils.should_run_eagerly()
        model._experimental_run_tf_function = testing_utils.should_run_tf_function()
        x = np.random.rand(100, 4)
        y = np.sum(x, axis=-1)

        history = model.fit(x, y, epochs=10).history
        self.assertGreater(history['f1_binary'][-1], history['f1_binary'][0])


if __name__ == "__main__":
    tf.test.main()
