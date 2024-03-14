from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_keras import layers, optimizers
from tf_keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.ops import variables
from tfmiss.keras.metrics.f1 import F1Binary, F1Micro, F1Macro


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class F1BinaryTest(test_combinations.TestCase):
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
        y_pred = tf.constant([[[0.9], [0.1], [0.9], [0.1]], [[0.1], [0.9], [0.1], [0.9]]], shape=(2, 4, 1))
        y_true = tf.constant([[0, 1, 0, 1], [1, 0, 1, 0]], shape=(2, 4, 1))
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
        model = test_utils.get_model_from_layers([
            layers.Dense(3, activation='relu', kernel_initializer='ones'),
            layers.Dense(1, activation='sigmoid', kernel_initializer='ones')
        ], input_shape=(4,))
        model.compile(loss='mae', metrics=[f1_obj], optimizer='rmsprop', run_eagerly=test_utils.should_run_eagerly())
        model.run_eagerly = test_utils.should_run_eagerly()
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
        model = test_utils.get_model_from_layers([
            layers.Dense(3),
            layers.Dense(1, activation='sigmoid')
        ], input_shape=(4,))
        model.compile(
            loss='binary_crossentropy', metrics=[f1_obj], optimizer=optimizers.Adam(),
            run_eagerly=test_utils.should_run_eagerly())
        model.run_eagerly = test_utils.should_run_eagerly()
        x = np.random.rand(100, 4)
        y = x.mean(-1).round().astype('int32')

        history = model.fit(x, y, epochs=10).history
        self.assertGreater(history['f1_binary'][-1], history['f1_binary'][0])


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class F1MicroTest(test_combinations.TestCase):
    def test_config(self):
        r_obj = F1Micro(name='my_f1micro')
        self.assertEqual(r_obj.name, 'my_f1micro')
        self.assertEqual(len(r_obj.variables), 4)
        self.assertEqual([v.name for v in r_obj.variables],
                         ['true_positives:0', 'false_positives:0', 'true_positives:0', 'false_negatives:0'])

        # Check save and restore config
        r_obj2 = F1Micro.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, 'my_f1micro')
        self.assertEqual(len(r_obj2.variables), 4)

    def test_value_is_idempotent(self):
        r_obj = F1Micro()
        y_pred = tf.random.uniform(shape=(10, 3), dtype=tf.float32)
        y_true = tf.random.uniform(shape=(10,), dtype=tf.int32, maxval=3)
        update_op = r_obj.update_state(y_true, y_pred)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        # Run several updates.
        for _ in range(10):
            self.evaluate(update_op)

        # Then verify idempotency.
        initial_f1micro = self.evaluate(r_obj.result())
        for _ in range(10):
            self.assertAlmostEqual(initial_f1micro, self.evaluate(r_obj.result()))

    def test_unweighted(self):
        r_obj = F1Micro()
        y_pred = tf.constant([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]], shape=(1, 4, 2))
        y_true = tf.constant([0, 1, 1, 0], shape=(1, 4))
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.5, self.evaluate(result))

    def test_unweighted_all_incorrect(self):
        r_obj = F1Micro()
        y_pred = tf.constant([[[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],
                              [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]], shape=(2, 4, 2))
        y_true = tf.constant([[1, 0, 1, 0], [1, 0, 1, 0]], shape=(2, 4))
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0, self.evaluate(result))

    def test_weighted(self):
        r_obj = F1Micro()
        y_pred = tf.constant([[[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],
                              [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]])
        y_true = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred, sample_weight=tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))
        self.assertAlmostEqual(0.3, self.evaluate(result))

    def test_div_by_zero(self):
        r_obj = F1Micro()
        y_pred = tf.constant([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])
        y_true = tf.constant([0, 0, 0, 0])
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertEqual(0, self.evaluate(result))

    def test_multiple_updates(self):
        r_obj = F1Micro()
        y_true = tf.constant([[0, 1], [1, 0]], shape=(2, 2))
        y_pred = tf.constant([[[0.1, 0.9], [0.9, 0.1]], [[0.1, 0.9], [0.9, 0.1]]], shape=(2, 2, 2), dtype=tf.float32)
        weights = tf.constant([[1, 4], [3, 2]], shape=(2, 2), dtype=tf.float32)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        update_op = r_obj.update_state(y_true, y_pred, sample_weight=weights)
        for _ in range(2):
            self.evaluate(update_op)

        self.assertAlmostEqual(0.5, self.evaluate(r_obj.result()))

    def test_all_true(self):
        r_obj = F1Micro()
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant(
            [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1],
             [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]], shape=(1, 10, 2))
        y_true = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], shape=(1, 10, 1))
        result = r_obj(y_true, y_pred)
        self.assertEqual(1.0, self.evaluate(result))

    def test_known_result(self):
        r_obj = F1Micro()
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant(
            [[0.5, 0.4, 0.1], [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.4, 0.5, 0.1], [0.4, 0.5, 0.1], [0.4, 0.5, 0.1],
             [0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.5, 0.4, 0.1]])
        y_true = tf.constant([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.75, self.evaluate(result))

    def test_reset_states(self):
        f1_obj = F1Micro()
        model = test_utils.get_model_from_layers([
            layers.Dense(3, activation='relu', kernel_initializer='ones'),
            layers.Dense(2, activation='softmax', kernel_initializer='ones')
        ], input_shape=(4,))
        model.compile(loss='mae', metrics=[f1_obj], optimizer='rmsprop', run_eagerly=test_utils.should_run_eagerly())
        model.run_eagerly = test_utils.should_run_eagerly()
        x = np.ones((100, 4))
        y = np.zeros((100, 1), dtype=np.int32)

        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.precision.true_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.precision.false_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.recall.true_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.recall.false_negatives), 0.)

        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.precision.true_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.precision.false_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.recall.true_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.recall.false_negatives), 0.)

    def test_metric_rises(self):
        f1_obj = F1Micro()
        model = test_utils.get_model_from_layers([
            layers.Dense(3),
            layers.Dense(2, activation='softmax')
        ], input_shape=(4,))
        model.compile(
            loss='sparse_categorical_crossentropy', metrics=[f1_obj], optimizer=optimizers.Adam(),
            run_eagerly=test_utils.should_run_eagerly())
        model.run_eagerly = test_utils.should_run_eagerly()
        x = np.random.rand(100, 4)
        y = x.argmax(-1).astype('int32') // 2

        history = model.fit(x, y, epochs=10).history
        self.assertGreater(history['f1_micro'][-1], history['f1_micro'][0])


@test_combinations.run_with_all_model_types
@test_combinations.run_all_keras_modes
class F1MacroTest(test_combinations.TestCase):
    def test_config(self):
        r_obj = F1Macro(name='my_f1Macro')
        self.assertEqual(r_obj.name, 'my_f1Macro')
        self.assertEqual(len(r_obj.variables), 4)
        self.assertEqual([v.name for v in r_obj.variables],
                         ['true_positives:0', 'false_positives:0', 'true_positives:0', 'false_negatives:0'])

        # Check save and restore config
        r_obj2 = F1Macro.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, 'my_f1Macro')
        self.assertEqual(len(r_obj2.variables), 4)

    def test_value_is_idempotent(self):
        r_obj = F1Macro()
        y_pred = tf.random.uniform(shape=(10, 3), dtype=tf.float32)
        y_true = tf.random.uniform(shape=(10,), dtype=tf.int32, maxval=3)
        update_op = r_obj.update_state(y_true, y_pred)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        # Run several updates.
        for _ in range(10):
            self.evaluate(update_op)

        # Then verify idempotency.
        initial_f1Macro = self.evaluate(r_obj.result())
        for _ in range(10):
            self.assertAlmostEqual(initial_f1Macro, self.evaluate(r_obj.result()))

    def test_unweighted(self):
        r_obj = F1Macro()
        y_pred = tf.constant([[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]], shape=(1, 4, 2))
        y_true = tf.constant([0, 1, 1, 0], shape=(1, 4))
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.5, self.evaluate(result))

    def test_unweighted_all_incorrect(self):
        r_obj = F1Macro()
        y_pred = tf.constant([[[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],
                              [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]], shape=(2, 4, 2))
        y_true = tf.constant([[1, 0, 1, 0], [1, 0, 1, 0]], shape=(2, 4))
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0, self.evaluate(result))

    def test_weighted(self):
        r_obj = F1Macro()
        y_pred = tf.constant([[[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]],
                              [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]]])
        y_true = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred, sample_weight=tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))
        self.assertAlmostEqual(0.3, self.evaluate(result))

    def test_div_by_zero(self):
        r_obj = F1Macro()
        y_pred = tf.constant([[0.1, 0.9], [0.1, 0.9], [0.1, 0.9], [0.1, 0.9]])
        y_true = tf.constant([0, 0, 0, 0])
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertEqual(0, self.evaluate(result))

    def test_multiple_updates(self):
        r_obj = F1Macro()
        y_true = tf.constant([[0, 1], [1, 0]], shape=(2, 2))
        y_pred = tf.constant([[[0.1, 0.9], [0.9, 0.1]], [[0.1, 0.9], [0.9, 0.1]]], shape=(2, 2, 2), dtype=tf.float32)
        weights = tf.constant([[1, 4], [3, 2]], shape=(2, 2), dtype=tf.float32)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        update_op = r_obj.update_state(y_true, y_pred, sample_weight=weights)
        for _ in range(2):
            self.evaluate(update_op)

        self.assertAlmostEqual(0.4949495, self.evaluate(r_obj.result()))

    def test_all_true(self):
        r_obj = F1Macro()
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant(
            [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1],
             [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9]], shape=(1, 10, 2))
        y_true = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], shape=(1, 10, 1))
        result = r_obj(y_true, y_pred)
        self.assertEqual(1.0, self.evaluate(result))

    def test_known_result(self):
        r_obj = F1Macro()
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant(
            [[0.5, 0.4, 0.1], [0.5, 0.4, 0.1], [0.4, 0.5, 0.1], [0.4, 0.5, 0.1], [0.4, 0.5, 0.1], [0.4, 0.5, 0.1],
             [0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.4, 0.1, 0.5], [0.5, 0.4, 0.1]])
        y_true = tf.constant([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2])

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.7388889, self.evaluate(result))

    def test_reset_states(self):
        f1_obj = F1Macro()
        model = test_utils.get_model_from_layers([
            layers.Dense(3, activation='relu', kernel_initializer='ones'),
            layers.Dense(2, activation='softmax', kernel_initializer='ones')
        ], input_shape=(4,))
        model.compile(loss='mae', metrics=[f1_obj], optimizer='rmsprop', run_eagerly=test_utils.should_run_eagerly())
        model.run_eagerly = test_utils.should_run_eagerly()
        x = np.ones((100, 4))
        y = np.zeros((100, 1), dtype=np.int32)

        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.class2f1[0].precision.true_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[0].precision.false_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[0].recall.true_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[0].recall.false_negatives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[1].precision.true_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[1].precision.false_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[1].recall.true_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[1].recall.false_negatives), 0.)

        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.class2f1[0].precision.true_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[0].precision.false_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[0].recall.true_positives), 100.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[0].recall.false_negatives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[1].precision.true_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[1].precision.false_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[1].recall.true_positives), 0.)
        self.assertEqual(self.evaluate(f1_obj.class2f1[1].recall.false_negatives), 0.)

    def test_metric_rises(self):
        f1_obj = F1Macro()
        model = test_utils.get_model_from_layers([
            layers.Dense(3),
            layers.Dense(2, activation='softmax')
        ], input_shape=(4,))
        model.compile(
            loss='sparse_categorical_crossentropy', metrics=[f1_obj], optimizer=optimizers.Adam(),
            run_eagerly=test_utils.should_run_eagerly())
        model.run_eagerly = test_utils.should_run_eagerly()
        x = np.random.rand(100, 4)
        y = x.argmax(-1).astype('int32') // 2

        history = model.fit(x, y, epochs=10).history
        self.assertGreater(history['f1_macro'][-1], history['f1_macro'][0])


if __name__ == "__main__":
    tf.test.main()
