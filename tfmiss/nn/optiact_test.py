import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.ops.gradient_checker_v2 import max_error

from tfmiss.nn.optiact import gelu
from tfmiss.nn.optiact import silu


class GeluTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        ("/cpu:0", "float16", 3e-2),
        ("/cpu:0", "bfloat16", 3e-2),
        ("/cpu:0", "float32", 2e-2),
        ("/cpu:0", "float64", 2e-2),
        ("/gpu:0", "float16", 1e-7),
        ("/gpu:0", "bfloat16", 1e-7),
        ("/gpu:0", "float32", 1e-7),
        ("/gpu:0", "float64", 1e-7),
    )
    def test_grad(self, dev, dt, tol):
        if "gpu" in dev and not len(tf.config.list_physical_devices("GPU")):
            return self.skipTest("No GPU available")

        def _op(x):
            with tf.device(dev):
                return gelu(x)

        arguments = [np.random.uniform(-8, 8.0, size=(1000,))]
        theoretical, _ = tf.test.compute_gradient(
            _op, [a.astype(dt) for a in arguments]
        )
        _, numerical64 = tf.test.compute_gradient(_op, arguments)
        grad_err = max_error(theoretical, numerical64)

        self.assertLess(grad_err, tol)


class SiluTest(tf.test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        ("/cpu:0", "float16", 9e-3),
        ("/cpu:0", "bfloat16", 4e-2),
        ("/cpu:0", "float32", 3e-3),
        ("/cpu:0", "float64", 3e-3),
        ("/gpu:0", "float16", 1e-7),
        ("/gpu:0", "bfloat16", 1e-7),
        ("/gpu:0", "float32", 1e-7),
        ("/gpu:0", "float64", 1e-7),
    )
    def test_grad(self, dev, dt, tol):
        if "gpu" in dev and not len(tf.config.list_physical_devices("GPU")):
            return self.skipTest("No GPU available")

        def _op(x):
            with tf.device(dev):
                return silu(x)

        arguments = [np.random.uniform(-8, 8.0, size=(1000,))]
        theoretical, _ = tf.test.compute_gradient(
            _op, [a.astype(dt) for a in arguments]
        )
        _, numerical64 = tf.test.compute_gradient(_op, arguments)
        grad_err = max_error(theoretical, numerical64)

        self.assertLess(grad_err, tol)


if __name__ == "__main__":
    tf.test.main()
