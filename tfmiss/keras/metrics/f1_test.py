from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.ops import variables
from tfmiss.keras.metrics.f1 import F1Binary, F1Macro, F1Micro


@test_util.run_all_in_graph_and_eager_modes
class F1BinaryTest(tf.test.TestCase):
    def testConfig(self):
        r_obj = F1Binary(name='my_f1binary')
        self.assertEqual(r_obj.name, 'my_f1binary')
        self.assertEqual(len(r_obj.variables), 3)
        self.assertEqual([v.name for v in r_obj.variables],
                         ['true_positives_0:0', 'false_positives_0:0', 'false_negatives_0:0'])

        # Check save and restore config
        r_obj2 = F1Binary.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, 'my_f1binary')
        self.assertEqual(len(r_obj2.variables), 3)

    def testValueIsIdempotent(self):
        r_obj = F1Binary()
        y_pred = tf.random.uniform(shape=(10, 1), dtype=tf.int32, maxval=2)
        y_true = tf.random.uniform(shape=(10, 1), dtype=tf.int32, maxval=2)
        update_op = r_obj.update_state(y_true, y_pred)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        # Run several updates.
        for _ in range(10):
            self.evaluate(update_op)

        # Then verify idempotency.
        initial_f1binary = self.evaluate(r_obj.result())
        for _ in range(10):
            self.assertAlmostEqual(initial_f1binary, self.evaluate(r_obj.result()))

    def testUnweighted(self):
        r_obj = F1Binary()
        y_pred = tf.constant([1, 0, 1, 0], shape=(1, 4))
        y_true = tf.constant([0, 1, 1, 0], shape=(1, 4))
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.5, self.evaluate(result))

    def testUnweightedAllIncorrect(self):
        r_obj = F1Binary()
        inputs = np.random.randint(0, 2, size=(100, 1))
        y_pred = tf.constant(inputs)
        y_true = tf.constant(1 - inputs)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0, self.evaluate(result))

    def testWeighted(self):
        r_obj = F1Binary()
        y_pred = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
        y_true = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred, sample_weight=tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))
        self.assertAlmostEqual(0.44444448, self.evaluate(result))

    def testDivByZero(self):
        r_obj = F1Binary()
        y_pred = tf.constant([0, 0, 0, 0])
        y_true = tf.constant([0, 0, 0, 0])
        self.evaluate(variables.variables_initializer(r_obj.variables))

        result = r_obj(y_true, y_pred)
        self.assertEqual(0, self.evaluate(result))

    def testMultipleUpdates(self):
        r_obj = F1Binary()
        y_true = tf.constant([[0, 1], [1, 0]], shape=(2, 2))
        y_pred = tf.constant([[1, 0], [0.6, 0]], shape=(2, 2), dtype=tf.float32)
        weights = tf.constant([[1, 4], [3, 2]], shape=(2, 2), dtype=tf.float32)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        update_op = r_obj.update_state(y_true, y_pred, sample_weight=weights)
        for _ in range(2):
            self.evaluate(update_op)

        self.assertAlmostEqual(0.5454545, self.evaluate(r_obj.result()))

    def testAllTrue(self):
        r_obj = F1Binary()
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], shape=(1, 10))
        y_true = tf.constant([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], shape=(1, 10))
        result = r_obj(y_true, y_pred)
        self.assertEqual(1.0, self.evaluate(result))

    def testKnownResult(self):
        r_obj = F1Binary()
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant([1, 0, 0, 1, 0, 1, 1, 1], shape=(1, 8))
        y_true = tf.constant([1, 1, 0, 0, 0, 1, 0, 1], shape=(1, 8))
        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.6666667, self.evaluate(result))


@test_util.run_all_in_graph_and_eager_modes
class F1MacroTest(tf.test.TestCase):
    def testConfig(self):
        r_obj = F1Macro(num_classes=3, name='my_f1macro')
        self.assertEqual(r_obj.name, 'my_f1macro')
        self.assertEqual(len(r_obj.variables), 9)
        self.assertEqual([v.name for v in r_obj.variables],
                         ['true_positives_0:0', 'false_positives_0:0', 'false_negatives_0:0',
                          'true_positives_1:0', 'false_positives_1:0', 'false_negatives_1:0',
                          'true_positives_2:0', 'false_positives_2:0', 'false_negatives_2:0'])

        # Check save and restore config
        r_obj2 = F1Macro.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, 'my_f1macro')
        self.assertEqual(len(r_obj2.variables), 9)

    def testValueIsIdempotent(self):
        r_obj = F1Macro(num_classes=3)
        y_pred = tf.random.uniform(shape=(10, 3), dtype=tf.int32, maxval=3)
        y_true = tf.random.uniform(shape=(10, 1), dtype=tf.int32, maxval=3)
        update_op = r_obj.update_state(y_true, y_pred)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        # Run several updates.
        for _ in range(10):
            self.evaluate(update_op)

        # Then verify idempotency.
        initial_f1macro = self.evaluate(r_obj.result())
        for _ in range(10):
            self.assertAlmostEqual(initial_f1macro, self.evaluate(r_obj.result()), places=6)

    def testKnownResult(self):
        r_obj = F1Macro(num_classes=3)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ], shape=(6, 1, 3))
        y_true = tf.constant([0, 1, 2, 0, 1, 2], shape=(6, 1))
        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.26666666666666666, self.evaluate(result))


@test_util.run_all_in_graph_and_eager_modes
class F1MicroTest(tf.test.TestCase):
    def testConfig(self):
        r_obj = F1Micro(num_classes=3, name='my_f1micro')
        self.assertEqual(r_obj.name, 'my_f1micro')
        self.assertEqual(len(r_obj.variables), 9)
        self.assertEqual([v.name for v in r_obj.variables],
                         ['true_positives_0:0', 'false_positives_0:0', 'false_negatives_0:0',
                          'true_positives_1:0', 'false_positives_1:0', 'false_negatives_1:0',
                          'true_positives_2:0', 'false_positives_2:0', 'false_negatives_2:0'])

        # Check save and restore config
        r_obj2 = F1Micro.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, 'my_f1micro')
        self.assertEqual(len(r_obj2.variables), 9)

    def testValueIsIdempotent(self):
        r_obj = F1Micro(num_classes=3)
        y_pred = tf.random.uniform(shape=(10, 3), dtype=tf.int32, maxval=3)
        y_true = tf.random.uniform(shape=(10, 1), dtype=tf.int32, maxval=3)
        update_op = r_obj.update_state(y_true, y_pred)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        # Run several updates.
        for _ in range(10):
            self.evaluate(update_op)

        # Then verify idempotency.
        initial_f1micro = self.evaluate(r_obj.result())
        for _ in range(10):
            self.assertAlmostEqual(initial_f1micro, self.evaluate(r_obj.result()))

    def testKnownResult(self):
        r_obj = F1Micro(num_classes=3)
        self.evaluate(variables.variables_initializer(r_obj.variables))

        y_pred = tf.constant([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ], shape=(6, 1, 3))
        y_true = tf.constant([0, 2, 1, 0, 0, 1], shape=(6, 1))
        result = r_obj(y_true, y_pred)
        self.assertAlmostEqual(0.3333333333333333, self.evaluate(result))


@keras_parameterized.run_with_all_model_types
@keras_parameterized.run_all_keras_modes
class ResetStatesTest(keras_parameterized.TestCase):
    def testF1Binary(self):
        f1_obj = F1Binary()
        model = _get_model([f1_obj], out_dim=1)
        x = np.ones((100, 4))
        y = np.zeros((100, 1), dtype=np.int32)
        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.true_positives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_0), 100.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_0), 0.)

        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.true_positives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_0), 100.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_0), 0.)

    def testF1Macro(self):
        f1_obj = F1Macro(num_classes=3)
        model = _get_model([f1_obj], out_dim=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 1), dtype=np.int32)
        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.true_positives_0), 100.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.true_positives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.true_positives_2), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_2), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_2), 0.)

        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.true_positives_0), 100.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.true_positives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.true_positives_2), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_2), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_2), 0.)

    def testF1Micro(self):
        f1_obj = F1Micro(num_classes=3)
        model = _get_model([f1_obj], out_dim=3)
        x = np.ones((100, 4))
        y = np.zeros((100, 1))
        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.true_positives_0), 100.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.true_positives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.true_positives_2), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_2), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_2), 0.)

        model.evaluate(x, y)
        self.assertEqual(self.evaluate(f1_obj.true_positives_0), 100.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_0), 0.)
        self.assertEqual(self.evaluate(f1_obj.true_positives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_1), 0.)
        self.assertEqual(self.evaluate(f1_obj.true_positives_2), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_positives_2), 0.)
        self.assertEqual(self.evaluate(f1_obj.false_negatives_2), 0.)


def _get_model(compile_metrics, out_dim):
    layers = [
        tf.keras.layers.Dense(3, activation='relu', kernel_initializer='ones'),
        tf.keras.layers.Dense(out_dim, activation='sigmoid', kernel_initializer='ones')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(4,))
    model.compile(
        loss='mae',
        metrics=compile_metrics,
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly(),
        experimental_run_tf_function=testing_utils.should_run_tf_function()
    )

    return model


if __name__ == "__main__":
    tf.test.main()
