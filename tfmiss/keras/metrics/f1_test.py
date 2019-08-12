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
        r_obj = F1Binary(name='my_f1binary', thresholds=[0.4, 0.9], top_k=15)
        self.assertEqual(r_obj.name, 'my_f1binary')
        self.assertEqual(len(r_obj.variables), 3)
        self.assertEqual([v.name for v in r_obj.variables],
                         ['true_positives_0:0', 'false_positives_0:0', 'false_negatives_0:0'])
        self.assertEqual(r_obj.thresholds, [0.4, 0.9])
        self.assertEqual(r_obj.top_k, 15)

        # Check save and restore config
        r_obj2 = F1Binary.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, 'my_f1binary')
        self.assertEqual(len(r_obj2.variables), 3)
        self.assertEqual(r_obj2.thresholds, [0.4, 0.9])
        self.assertEqual(r_obj2.top_k, 15)

    # def testValueIsIdempotent(self):
    #     r_obj = F1Binary(thresholds=[0.3, 0.72])
    #     y_pred = tf.random.uniform(shape=(10, 3))
    #     y_true = tf.random.uniform(shape=(10, 3))
    #     update_op = r_obj.update_state(y_true, y_pred)
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #
    #     # Run several updates.
    #     for _ in range(10):
    #         self.evaluate(update_op)
    #
    #     # Then verify idempotency.
    #     initial_f1binary = self.evaluate(r_obj.result())
    #     for _ in range(10):
    #         self.assertArrayNear(initial_f1binary, self.evaluate(r_obj.result()), 1e-3)

    # def testUnweighted(self):
    #     r_obj = F1Binary()
    #     y_pred = tf.constant([1, 0, 1, 0], shape=(1, 4))
    #     y_true = tf.constant([0, 1, 1, 0], shape=(1, 4))
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #     result = r_obj(y_true, y_pred)
    #     self.assertAlmostEqual(0.5, self.evaluate(result))
    #
    # def testUnweightedAllIncorrect(self):
    #     r_obj = F1Binary(thresholds=[0.5])
    #     inputs = np.random.randint(0, 2, size=(100, 1))
    #     y_pred = tf.constant(inputs)
    #     y_true = tf.constant(1 - inputs)
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #     result = r_obj(y_true, y_pred)
    #     self.assertAlmostEqual(0, self.evaluate(result))
    #
    # def testWeighted(self):
    #     r_obj = F1Binary()
    #     y_pred = tf.constant([[1, 0, 1, 0], [0, 1, 0, 1]])
    #     y_true = tf.constant([[0, 1, 1, 0], [1, 0, 0, 1]])
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #     result = r_obj(
    #         y_true,
    #         y_pred,
    #         sample_weight=tf.constant([[1, 2, 3, 4], [4, 3, 2, 1]]))
    #     weighted_tp = 3.0 + 1.0
    #     weighted_t = (2.0 + 3.0) + (4.0 + 1.0)
    #     expected_f1binary = weighted_tp / weighted_t
    #     self.assertAlmostEqual(expected_f1binary, self.evaluate(result))
    #
    # def testDivByZero(self):
    #     r_obj = F1Binary()
    #     y_pred = tf.constant([0, 0, 0, 0])
    #     y_true = tf.constant([0, 0, 0, 0])
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #     result = r_obj(y_true, y_pred)
    #     self.assertEqual(0, self.evaluate(result))
    #
    # def testUnweightedWithThreshold(self):
    #     r_obj = F1Binary(thresholds=[0.5, 0.7])
    #     y_pred = tf.constant([1, 0, 0.6, 0], shape=(1, 4))
    #     y_true = tf.constant([0, 1, 1, 0], shape=(1, 4))
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #     result = r_obj(y_true, y_pred)
    #     self.assertArrayNear([0.5, 0.], self.evaluate(result), 0)
    #
    # def testWeightedWithThreshold(self):
    #     r_obj = F1Binary(thresholds=[0.5, 1.])
    #     y_true = tf.constant([[0, 1], [1, 0]], shape=(2, 2))
    #     y_pred = tf.constant([[1, 0], [0.6, 0]],
    #                          shape=(2, 2),
    #                          dtype=tf.float32)
    #     weights = tf.constant([[1, 4], [3, 2]],
    #                           shape=(2, 2),
    #                           dtype=tf.float32)
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #     result = r_obj(y_true, y_pred, sample_weight=weights)
    #     weighted_tp = 0 + 3.
    #     weighted_positives = (0 + 3.) + (4. + 0.)
    #     expected_f1binary = weighted_tp / weighted_positives
    #     self.assertArrayNear([expected_f1binary, 0], self.evaluate(result), 1e-3)
    #
    # def testMultipleUpdates(self):
    #     r_obj = F1Binary(thresholds=[0.5, 1.])
    #     y_true = tf.constant([[0, 1], [1, 0]], shape=(2, 2))
    #     y_pred = tf.constant([[1, 0], [0.6, 0]],
    #                          shape=(2, 2),
    #                          dtype=tf.float32)
    #     weights = tf.constant([[1, 4], [3, 2]],
    #                           shape=(2, 2),
    #                           dtype=tf.float32)
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #     update_op = r_obj.update_state(y_true, y_pred, sample_weight=weights)
    #     for _ in range(2):
    #         self.evaluate(update_op)
    #
    #     weighted_tp = (0 + 3.) + (0 + 3.)
    #     weighted_positives = ((0 + 3.) + (4. + 0.)) + ((0 + 3.) + (4. + 0.))
    #     expected_f1binary = weighted_tp / weighted_positives
    #     self.assertArrayNear([expected_f1binary, 0], self.evaluate(r_obj.result()),
    #                          1e-3)
    #
    # def testUnweightedTopK(self):
    #     r_obj = F1Binary(top_k=3)
    #     y_pred = tf.constant([0.2, 0.1, 0.5, 0, 0.2], shape=(1, 5))
    #     y_true = tf.constant([0, 1, 1, 0, 0], shape=(1, 5))
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #     result = r_obj(y_true, y_pred)
    #     self.assertAlmostEqual(0.5, self.evaluate(result))
    #
    # def testWeightedTopK(self):
    #     r_obj = F1Binary(top_k=3)
    #     y_pred1 = tf.constant([0.2, 0.1, 0.4, 0, 0.2], shape=(1, 5))
    #     y_true1 = tf.constant([0, 1, 1, 0, 1], shape=(1, 5))
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #     self.evaluate(
    #         r_obj(
    #             y_true1,
    #             y_pred1,
    #             sample_weight=tf.constant([[1, 4, 2, 3, 5]])))
    #
    #     y_pred2 = tf.constant([0.2, 0.6, 0.4, 0.2, 0.2], shape=(1, 5))
    #     y_true2 = tf.constant([1, 0, 1, 1, 1], shape=(1, 5))
    #     result = r_obj(y_true2, y_pred2, sample_weight=tf.constant(3))
    #
    #     tp = (2 + 5) + (3 + 3)
    #     positives = (4 + 2 + 5) + (3 + 3 + 3 + 3)
    #     expected_f1binary = tp / positives
    #     self.assertAlmostEqual(expected_f1binary, self.evaluate(result))
    #
    # def testUnweightedClassId(self):
    #     r_obj = F1Binary(class_id=2)
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #
    #     y_pred = tf.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
    #     y_true = tf.constant([0, 1, 1, 0, 0], shape=(1, 5))
    #     result = r_obj(y_true, y_pred)
    #     self.assertAlmostEqual(1, self.evaluate(result))
    #     self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    #     self.assertAlmostEqual(0, self.evaluate(r_obj.false_negatives))
    #
    #     y_pred = tf.constant([0.2, 0.1, 0, 0, 0.2], shape=(1, 5))
    #     y_true = tf.constant([0, 1, 1, 0, 0], shape=(1, 5))
    #     result = r_obj(y_true, y_pred)
    #     self.assertAlmostEqual(0.5, self.evaluate(result))
    #     self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    #     self.assertAlmostEqual(1, self.evaluate(r_obj.false_negatives))
    #
    #     y_pred = tf.constant([0.2, 0.1, 0.6, 0, 0.2], shape=(1, 5))
    #     y_true = tf.constant([0, 1, 0, 0, 0], shape=(1, 5))
    #     result = r_obj(y_true, y_pred)
    #     self.assertAlmostEqual(0.5, self.evaluate(result))
    #     self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    #     self.assertAlmostEqual(1, self.evaluate(r_obj.false_negatives))
    #
    # def testUnweightedTopKAndClassId(self):
    #     r_obj = F1Binary(class_id=2, top_k=2)
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #
    #     y_pred = tf.constant([0.2, 0.6, 0.3, 0, 0.2], shape=(1, 5))
    #     y_true = tf.constant([0, 1, 1, 0, 0], shape=(1, 5))
    #     result = r_obj(y_true, y_pred)
    #     self.assertAlmostEqual(1, self.evaluate(result))
    #     self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    #     self.assertAlmostEqual(0, self.evaluate(r_obj.false_negatives))
    #
    #     y_pred = tf.constant([1, 1, 0.9, 1, 1], shape=(1, 5))
    #     y_true = tf.constant([0, 1, 1, 0, 0], shape=(1, 5))
    #     result = r_obj(y_true, y_pred)
    #     self.assertAlmostEqual(0.5, self.evaluate(result))
    #     self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    #     self.assertAlmostEqual(1, self.evaluate(r_obj.false_negatives))
    #
    # def testUnweightedTopKAndThreshold(self):
    #     r_obj = F1Binary(thresholds=.7, top_k=2)
    #     self.evaluate(variables.variables_initializer(r_obj.variables))
    #
    #     y_pred = tf.constant([0.2, 0.8, 0.6, 0, 0.2], shape=(1, 5))
    #     y_true = tf.constant([1, 1, 1, 0, 1], shape=(1, 5))
    #     result = r_obj(y_true, y_pred)
    #     self.assertAlmostEqual(0.25, self.evaluate(result))
    #     self.assertAlmostEqual(1, self.evaluate(r_obj.true_positives))
    #     self.assertAlmostEqual(3, self.evaluate(r_obj.false_negatives))


@test_util.run_all_in_graph_and_eager_modes
class F1MacroTest(tf.test.TestCase):
    def testConfig(self):
        r_obj = F1Macro(num_classes=3, name='my_f1macro', thresholds=[0.4, 0.9], top_k=15)
        self.assertEqual(r_obj.name, 'my_f1macro')
        self.assertEqual(len(r_obj.variables), 9)
        self.assertEqual([v.name for v in r_obj.variables],
                         ['true_positives_0:0', 'false_positives_0:0', 'false_negatives_0:0',
                          'true_positives_1:0', 'false_positives_1:0', 'false_negatives_1:0',
                          'true_positives_2:0', 'false_positives_2:0', 'false_negatives_2:0'])
        self.assertEqual(r_obj.thresholds, [0.4, 0.9])
        self.assertEqual(r_obj.top_k, 15)

        # Check save and restore config
        r_obj2 = F1Macro.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, 'my_f1macro')
        self.assertEqual(len(r_obj2.variables), 9)
        self.assertEqual(r_obj2.thresholds, [0.4, 0.9])
        self.assertEqual(r_obj2.top_k, 15)


@test_util.run_all_in_graph_and_eager_modes
class F1MicroTest(tf.test.TestCase):
    def testConfig(self):
        r_obj = F1Micro(num_classes=3, name='my_f1micro', thresholds=[0.4, 0.9], top_k=15)
        self.assertEqual(r_obj.name, 'my_f1micro')
        self.assertEqual(len(r_obj.variables), 9)
        self.assertEqual([v.name for v in r_obj.variables],
                         ['true_positives_0:0', 'false_positives_0:0', 'false_negatives_0:0',
                          'true_positives_1:0', 'false_positives_1:0', 'false_negatives_1:0',
                          'true_positives_2:0', 'false_positives_2:0', 'false_negatives_2:0'])
        self.assertEqual(r_obj.thresholds, [0.4, 0.9])
        self.assertEqual(r_obj.top_k, 15)

        # Check save and restore config
        r_obj2 = F1Micro.from_config(r_obj.get_config())
        self.assertEqual(r_obj2.name, 'my_f1micro')
        self.assertEqual(len(r_obj2.variables), 9)
        self.assertEqual(r_obj2.thresholds, [0.4, 0.9])
        self.assertEqual(r_obj2.top_k, 15)


# @keras_parameterized.run_with_all_model_types
# @keras_parameterized.run_all_keras_modes
# class ResetStatesTest(keras_parameterized.TestCase):
#     def testF1Binary(self):
#         f1_obj = F1Binary()
#         model = _get_model([f1_obj])
#         x = np.ones((100, 4))
#         y = np.zeros((100, 1))
#         model.evaluate(x, y)
#         self.assertEqual(self.evaluate(f1_obj.accumulator), 100.)
#         model.evaluate(x, y)
#         self.assertEqual(self.evaluate(f1_obj.accumulator), 100.)


def _get_model(compile_metrics):
    layers = [
        tf.keras.layers.Dense(3, activation='relu', kernel_initializer='ones'),
        tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer='ones')
    ]
    model = testing_utils.get_model_from_layers(layers, input_shape=(4,))
    model.compile(
        loss='mae',
        metrics=compile_metrics,
        optimizer='rmsprop',
        run_eagerly=testing_utils.should_run_eagerly()
    )

    return model


if __name__ == "__main__":
    tf.test.main()
