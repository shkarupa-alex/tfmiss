from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.framework import test_util
from tensorflow.python.ops.gradient_checker_v2 import max_error
from tfmiss.nn.qrnn import fo_pool


def np_fo_pooling(x, forget, initial_state):
    batch, timesteps, hidden = x.shape
    dst = np.zeros((batch, timesteps + 1, hidden), dtype=x.dtype)
    dst[:, 0] = initial_state
    for ts in range(1, timesteps + 1):
        dst[:, ts] = forget[:, ts - 1] * x[:, ts - 1] + (1.0 - forget[:, ts - 1]) * dst[:, ts - 1]

    return dst[:, 1:]


@test_util.run_all_in_graph_and_eager_modes
class FoPoolTest(tf.test.TestCase, parameterized.TestCase):
    def test_shape_inference_valid(self):
        shape = (8, 128, 32)
        inputs = np.random.random(size=shape)
        forget = np.random.uniform(0, 1, size=shape)
        initial_state = np.random.random(size=shape[:1] + shape[2:])

        result = fo_pool(inputs, forget, initial_state)
        self.assertListEqual([8, 128, 32], result.shape.as_list())
        self.assertListEqual([8, 128, 32], list(self.evaluate(result).shape))

    @parameterized.parameters(
        ('/cpu:0', 'float16', 1e-6), ('/cpu:0', 'bfloat16', 2e-2), ('/cpu:0', 'float32', 1e-6),
        ('/cpu:0', 'float64', 1e-13),
        ('/gpu:0', 'float16', 1e-6), ('/gpu:0', 'bfloat16', 2e-2), ('/gpu:0', 'float32', 1e-6),
        ('/gpu:0', 'float64', 1e-13)
    )
    def test_value(self, dev, dt, tol):
        if 'gpu' in dev and not len(tf.config.list_physical_devices('GPU')):
            return self.skipTest('No GPU available')

        with tf.device(dev):
            shape = (8, 128, 32)
            inputs = np.random.random(size=shape).astype(dt)
            forget = np.random.uniform(0, 1, size=shape).astype(dt)
            initial_state = np.random.random(size=shape[:1] + shape[2:]).astype(dt)
            expected = np_fo_pooling(inputs, forget, initial_state)

            result = fo_pool(inputs, forget, initial_state)
            result = self.evaluate(result)

            self.assertEqual(result.dtype, dt)
            self.assertEqual(result.shape, expected.shape)
            self.assertAllClose(result, expected, atol=tol)

            grad = np.random.uniform(size=shape)

            with tf.GradientTape() as g:
                variables = [tf.constant(v, dt) for v in [inputs, forget, initial_state]]
                g.watch(variables)
                result = fo_pool(variables[0], variables[1], variables[2])

                result_grad = g.gradient(result, variables, output_gradients=tf.constant(grad, dt))
                result_grad = self.evaluate(result_grad)

                self.assertEqual(result_grad[0].dtype, dt)
                self.assertEqual(result_grad[1].dtype, dt)
                self.assertEqual(result_grad[2].dtype, dt)

                self.assertEqual(result_grad[0].shape, inputs.shape)
                self.assertEqual(result_grad[1].shape, forget.shape)
                self.assertEqual(result_grad[2].shape, initial_state.shape)

                self.assertTrue(np.all(np.isfinite(result_grad[0])))
                self.assertTrue(np.all(np.isfinite(result_grad[1])))
                self.assertTrue(np.all(np.isfinite(result_grad[2])))

    @parameterized.parameters(
        ('/cpu:0', 'float16', 9e-4), ('/cpu:0', 'bfloat16', 2e-2), ('/cpu:0', 'float32', 1e-7),
        ('/cpu:0', 'float64', 2e-13),
        ('/gpu:0', 'float16', 4e-3), ('/gpu:0', 'bfloat16', 2e-2), ('/gpu:0', 'float32', 6e-5),
        ('/gpu:0', 'float64', 2e-13)
    )
    def test_grad(self, dev, dt, tol):
        if 'gpu' in dev and not len(tf.config.list_physical_devices('GPU')):
            return self.skipTest('No GPU available')

        def _op(inp, frg, ini):
            with tf.device(dev):
                return fo_pool(inp, frg, ini)

        shape = (2, 16, 3)
        inputs = np.random.random(size=shape)
        forget = np.random.uniform(0, 1, size=shape)
        initial_state = np.random.random(size=shape[:1] + shape[2:])

        arguments = [inputs, forget, initial_state]
        theoretical, _ = tf.test.compute_gradient(_op, [a.astype(dt) for a in arguments])
        _, numerical64 = tf.test.compute_gradient(_op, arguments)
        grad_err = max_error(theoretical, numerical64)

        self.assertLess(grad_err, tol)


if __name__ == "__main__":
    tf.test.main()
