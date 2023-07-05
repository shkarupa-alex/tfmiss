from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.src.utils.losses_utils import ReductionV2 as Reduction
from tensorflow.python.framework import test_util
from tfmiss.keras.losses import bitemp


@test_util.run_all_in_graph_and_eager_modes
class BiTempLossTest(tf.test.TestCase):
    def test_normalization(self):
        """Test the normalization constant."""
        activations = tf.random.normal(shape=[100, 50000])

        for t in [0.99, 1.01]:
            normalization_constants = bitemp.compute_normalization(activations, t, num_iters=20)
            self.assertEqual(normalization_constants.shape, [100, 1])

            probabilities = tf.reduce_sum(bitemp.exp_t(activations - normalization_constants, t), -1)
            self.assertAllClose(self.evaluate(probabilities), [1.0] * 100, atol=1e-5)

        for t in [0.1, 2.0]:
            normalization_constants = bitemp.compute_normalization(activations, t, num_iters=20)
            probabilities = tf.reduce_sum(bitemp.exp_t(activations - normalization_constants, t), -1)
            self.assertAllClose(self.evaluate(probabilities), [1.0] * 100, atol=1e-5)

    def test_limit_case_logistic_loss(self):
        """Test for checking if t1 = t2 = 1.0 yields the logistic bitemp."""
        labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        activations = tf.random.normal(shape=[3, 3])
        actual_loss = bitemp.bi_tempered_logistic_loss(labels, activations, 1.0, 1.0)
        logistic_loss = tf.nn.softmax_cross_entropy_with_logits(logits=activations, labels=labels)
        actual_loss_out, logistic_loss_out = self.evaluate([actual_loss, logistic_loss])
        self.assertAllClose(actual_loss_out, logistic_loss_out)

    def test_loss_value(self):
        """Test the loss based on precomputed values."""
        labels = tf.constant([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1], [0.2, 0.8, 0.0]])
        activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]]

        actual_loss = bitemp.bi_tempered_logistic_loss(labels, activations, 0.5, 1.5)
        self.assertAllClose(self.evaluate(actual_loss), [0.02301914, 0.18972909, 0.93874922])

        actual_loss = bitemp.bi_tempered_logistic_loss(labels, activations, 0.5, 0.8, num_iters=20)
        self.assertAllClose(self.evaluate(actual_loss), [0.21646356, 0.41836615, 1.33997854])

    def test_constant_shift(self):
        """Test if adding a constant to all activations is vacuous."""
        labels = tf.constant([[0.2, 0.3, 0.5], [0.4, 0.4, 0.2], [0.7, 0.2, 0.1]])
        activations = tf.random.normal(shape=[3, 3])
        bias = tf.random.normal(shape=[3, 1])

        for t2 in [0.8, 1.2]:
            actual_loss = bitemp.bi_tempered_logistic_loss(labels, activations, 0.5, t2)
            shifted_loss = bitemp.bi_tempered_logistic_loss(labels, activations + bias, 0.5, t2)
            self.assertEqual(actual_loss.shape, [3])

            actual_loss_out, shifted_loss_out = self.evaluate([actual_loss, shifted_loss])
            self.assertAllClose(actual_loss_out, shifted_loss_out)

    def test_gradient_error(self):
        """Compare custom gradient with tf.gradient."""
        labels = tf.constant([[0.4, 0.3, 0.3], [0.8, 0.1, 0.1], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        activations = tf.Variable(tf.random.normal(shape=[4, 3]))

        with tf.GradientTape() as tape1:
            internal_loss = bitemp._internal_bi_tempered_logistic_loss(activations, labels, 0.5, 1.5)
            numerical_gradient = tape1.gradient(internal_loss, activations)

        with tf.GradientTape() as tape2:
            actual_loss = bitemp.bi_tempered_logistic_loss(labels, activations, 0.5, 1.5)
            actual_gradient = tape2.gradient(actual_loss, activations)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        internal_loss_out, actual_loss_out = self.evaluate([internal_loss, actual_loss])
        numerical_gradient_out, actual_gradient_out = self.evaluate([numerical_gradient[0], actual_gradient[0]])
        self.assertEqual(actual_gradient.shape, (4, 3))
        self.assertAllClose(actual_loss_out, internal_loss_out)
        self.assertAllClose(actual_gradient_out, numerical_gradient_out, atol=1e-5)

    def test_gradient_error_issue_2(self):
        """Compare custom gradient with tf.gradient for case https://github.com/google/bi-tempered-loss/issues/2"""
        labels = tf.constant([[0.4, 0.3, 0.3], [0.8, 0.1, 0.1], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        activations = tf.Variable(tf.random.normal(shape=[4, 3]))

        with tf.GradientTape() as tape1:
            internal_loss = bitemp._internal_bi_tempered_logistic_loss(activations, labels, 0.5, 1.0)
            numerical_gradient = tape1.gradient(internal_loss, activations)

        with tf.GradientTape() as tape2:
            actual_loss = bitemp.bi_tempered_logistic_loss(labels, activations, 0.5, 1.0)
            actual_gradient = tape2.gradient(actual_loss, activations)

        self.evaluate(tf.compat.v1.global_variables_initializer())
        internal_loss_out, actual_loss_out = self.evaluate([internal_loss, actual_loss])
        numerical_gradient_out, actual_gradient_out = self.evaluate([numerical_gradient[0], actual_gradient[0]])
        self.assertEqual(actual_gradient.shape, (4, 3))
        self.assertAllClose(actual_loss_out, internal_loss_out)
        self.assertAllClose(actual_gradient_out, numerical_gradient_out, atol=1e-5)

    def test_label_smoothing(self):
        """Test label smoothing."""
        labels = tf.constant([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]]
        actual_loss = bitemp.bi_tempered_logistic_loss(labels, activations, 0.5, 1.5, label_smoothing=0.1)

        actual_loss_out = self.evaluate(actual_loss)
        self.assertAllClose(actual_loss_out, [0.76652711, 0.08627685, 1.35443510], atol=1e-5)

    def test_binary_logistic_loss(self):
        """Test binary logistic bitemp."""
        labels = tf.constant([1.0, 0.0])
        activations = [0.0, 0.0]
        actual_loss = bitemp.bi_tempered_binary_logistic_loss(labels, activations, 1.0, 1.0)

        actual_loss_out = self.evaluate(actual_loss)
        self.assertAllClose(actual_loss_out, [0.69314718, 0.69314718], atol=1e-5)

    def test_dynamic_temperatures(self):
        """Test changing temperatures dynamically."""
        labels = tf.constant([[0.2, 0.5, 0.3]])
        activations = [[-0.5, 0.1, 2.0]]
        t1_values = [1.0, 0.9, 0.8, 0.7]
        t2_values = [1.0, 1.1, 1.2, 1.3]
        loss_values = [[0.62870466], [0.45677936], [0.34298314], [0.26295574]]
        loss_out = []

        for t1_value, t2_value in zip(t1_values, t2_values):
            actual_loss = bitemp.bi_tempered_logistic_loss(labels, activations, t1_value, t2_value, num_iters=5)
            loss_out.append(self.evaluate(actual_loss))
        self.assertAllClose(loss_values, loss_out, atol=1e-5)

    def test_sparse_loss(self):
        """Test int labels."""
        labels = tf.constant([0, 2, 1, 0])
        activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0], [-1.5, 0.7, 5.2]]
        actual_loss = bitemp.bi_tempered_logistic_loss(tf.one_hot(labels, 3), activations, 0.5, 1.5)
        sparse_loss = bitemp.sparse_bi_tempered_logistic_loss(labels, activations, 0.5, 1.5)

        actual_loss_out = self.evaluate(actual_loss)
        sparse_loss_out = self.evaluate(sparse_loss)
        self.assertAllClose(actual_loss_out, sparse_loss_out)

        labels = tf.constant([[0, 2], [1, 0]])
        activations = [[[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0]], [[4.0, -3.0, -6.0], [-1.5, 0.7, 5.2]]]
        actual_loss = bitemp.bi_tempered_logistic_loss(tf.one_hot(labels, 3), activations, 0.5, 1.5)
        sparse_loss = bitemp.sparse_bi_tempered_logistic_loss(labels, activations, 0.5, 1.5)
        actual_loss_out = self.evaluate(actual_loss)
        sparse_loss_out = self.evaluate(sparse_loss)
        self.assertAllClose(actual_loss_out, sparse_loss_out)

    def test_tempered_softmax(self):
        # Test softmax function with different temperatures.
        activations = [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]]

        # Test with temperature = 1.0, which should recover regular softmax probabilities.
        softmax_probabilities_t_1 = self.evaluate(bitemp.tempered_softmax(activations, t=1.0))
        vanilla_softmax_probabilties = self.evaluate(tf.nn.softmax(activations))
        self.assertAllClose(vanilla_softmax_probabilties, softmax_probabilities_t_1)

        softmax_probabilities_t_4 = self.evaluate(bitemp.tempered_softmax(activations, t=4.0))
        expected_softmax_probabilities_t_4 = ([
            [0.3205458, 0.32714278, 0.3523402],
            [0.3430056, 0.36491093, 0.29220778],
            [0.41369352, 0.30534995, 0.28299212]
        ])
        self.assertAllClose(expected_softmax_probabilities_t_4, softmax_probabilities_t_4)

    def test_tempered_sigmoid(self):
        # Test sigmoid function with different temperatures.
        activations = [0.0, 3.0, 6.0]

        # Test with temperature = 1.0, which should recover regular sigmoid probabilities.
        sigmoid_probabilities_t_1 = self.evaluate(bitemp.tempered_sigmoid(activations, t=1.0))
        vanilla_softmax_probabilties = self.evaluate(tf.nn.sigmoid(activations))
        self.assertAllClose(vanilla_softmax_probabilties, sigmoid_probabilities_t_1)

        sigmoid_probabilities_t_4 = self.evaluate(bitemp.tempered_sigmoid(activations, t=4.0))
        expected_sigmoid_probabilities_t_4 = [0.5, 0.58516014, 0.6421035]
        self.assertAllClose(expected_sigmoid_probabilities_t_4, sigmoid_probabilities_t_4)


@test_util.run_all_in_graph_and_eager_modes
class BiTemperedBinaryLogistic(tf.test.TestCase):
    def test_config(self):
        cl_obj = bitemp.BiTemperedBinaryLogistic(
            reduction=Reduction.SUM, from_logits=True, t1=0.5, t2=1.5)
        self.assertEqual(cl_obj.name, 'bi_tempered_binary_logistic')
        self.assertEqual(cl_obj.reduction, Reduction.SUM)

    def test_normal(self):
        y_true = tf.constant([1, 0], dtype=tf.int64)
        y_pred = tf.constant([0.0, 0.0], dtype=tf.float32)
        btl_obj = bitemp.BiTemperedBinaryLogistic(
            reduction=Reduction.SUM, from_logits=True, t1=1.0, t2=1.0)
        loss = btl_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 2 * 0.69314718, 3)

    def test_from_logits(self):
        y_true = tf.constant([1, 0], dtype=tf.int64)
        y_logits = tf.constant([-12.0, 0.5], dtype=tf.float32)
        y_pred = tf.nn.sigmoid(y_logits)

        logits_btl = bitemp.BiTemperedBinaryLogistic(
            reduction=Reduction.SUM, from_logits=True, t1=0.5, t2=2.0)
        sigmoid_btl = bitemp.BiTemperedBinaryLogistic(
            reduction=Reduction.SUM, t1=0.5, t2=2.0)

        logits_loss = logits_btl(y_true, y_logits)
        sigmoid_loss = sigmoid_btl(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(logits_loss), self.evaluate(sigmoid_loss), 3)


@test_util.run_all_in_graph_and_eager_modes
class BiTemperedLogistic(tf.test.TestCase):
    def test_config(self):
        cl_obj = bitemp.BiTemperedLogistic(
            reduction=Reduction.SUM, from_logits=True, t1=0.5, t2=1.5)
        self.assertEqual(cl_obj.name, 'bi_tempered_logistic')
        self.assertEqual(cl_obj.reduction, Reduction.SUM)

    def test_normal(self):
        y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.int64)
        y_pred = tf.constant([[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]], dtype=tf.float32)
        btl_obj = bitemp.BiTemperedLogistic(
            reduction=Reduction.SUM, from_logits=True, t1=0.5, t2=1.5, label_smoothing=0.1)
        loss = btl_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 0.76652711 + 0.08627685 + 1.35443510, 3)

    def test_from_logits(self):
        if tf.executing_eagerly():
            self.skipTest('Unable to obtain logits in eager mode')

        y_true = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=tf.int64)
        y_logits = tf.constant([[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0]], dtype=tf.float32)
        y_pred = tf.nn.softmax(y_logits)

        logits_btl = bitemp.BiTemperedLogistic(
            reduction=Reduction.SUM, from_logits=True, t1=0.5, t2=2.0)
        softmax_btl = bitemp.BiTemperedLogistic(
            reduction=Reduction.SUM, t1=0.5, t2=2.0)

        logits_loss = logits_btl(y_true, y_logits)
        softmax_loss = softmax_btl(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(logits_loss), self.evaluate(softmax_loss), 3)


@test_util.run_all_in_graph_and_eager_modes
class SparseBiTemperedLogistic(tf.test.TestCase):
    def test_config(self):
        cl_obj = bitemp.SparseBiTemperedLogistic(
            reduction=Reduction.SUM, from_logits=True, t1=0.5, t2=1.5)
        self.assertEqual(cl_obj.name, 'sparse_bi_tempered_logistic')
        self.assertEqual(cl_obj.reduction, Reduction.SUM)

    def test_normal(self):
        y_true = tf.constant([0, 2, 1, 0], dtype=tf.int64)
        y_pred = tf.constant(
            [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0], [-1.5, 0.7, 5.2]], dtype=tf.float32)
        btl_obj = bitemp.SparseBiTemperedLogistic(
            reduction=Reduction.SUM, from_logits=True, t1=0.5, t2=1.5)
        loss = btl_obj(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(loss), 5.269439, 3)

    def test_from_logits(self):
        if tf.executing_eagerly():
            self.skipTest('Unable to obtain logits in eager mode')

        y_true = tf.constant([0, 2, 1, 0], dtype=tf.int64)
        y_logits = tf.constant(
            [[-0.5, 0.1, 2.0], [0.1, 1.5, -5.0], [4.0, -3.0, -6.0], [-1.5, 0.7, 5.2]], dtype=tf.float32)
        y_pred = tf.nn.softmax(y_logits)

        logits_btl = bitemp.SparseBiTemperedLogistic(
            reduction=Reduction.SUM, from_logits=True, t1=0.5, t2=2.0)
        softmax_btl = bitemp.SparseBiTemperedLogistic(
            reduction=Reduction.SUM, t1=0.5, t2=2.0)

        logits_loss = logits_btl(y_true, y_logits)
        softmax_loss = softmax_btl(y_true, y_pred)
        self.assertAlmostEqual(self.evaluate(logits_loss), self.evaluate(softmax_loss), 3)


if __name__ == "__main__":
    tf.test.main()
