from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.nn.qrnn import fo_pool
from tfmiss.ops import tfmiss_ops


def np_fo_pooling(x, forget, initial_state, time_major):
    if not time_major:
        return np.transpose(np_fo_pooling(
            np.transpose(x, (1, 0, 2)), np.transpose(forget, (1, 0, 2)), initial_state, time_major=True), (1, 0, 2))
    timesteps, batch, hidden = x.shape
    dst = np.zeros((timesteps + 1, batch, hidden), dtype=x.dtype)
    dst[0] = initial_state
    for ts in range(1, timesteps + 1):
        dst[ts] = (forget[ts - 1] * x[ts - 1] + (1.0 - forget[ts - 1]) * dst[ts - 1])

    return dst[1:]


@test_util.run_all_in_graph_and_eager_modes
class FoPoolTest(tf.test.TestCase):
    def setUp(self):
        # Obtain a list of GPU device specifications ['/gpu:0', '/gpu:1', ...]
        self.gpu_devs = [d.name.replace('physical_device:', '') for d in tf.config.list_physical_devices() if d.device_type == 'GPU']
        super(FoPoolTest, self).setUp()

    def test_fo_pool(self):
        for FT in [np.float32, np.float64]:
            for time_major in [False, True]:
                self._impl_test_fo_pool(FT, time_major)

    def _impl_test_fo_pool(self, FT, time_major):
        # Create input variables
        timesteps = 20
        batch_size = 32
        channels = 64
        if time_major:
            shape = (timesteps, batch_size, channels)
        else:
            shape = (batch_size, timesteps, channels)
        inputs = np.random.random(size=shape).astype(FT)
        forget = np.random.uniform(0, 1, size=shape).astype(FT)
        initial_state = np.random.random(size=(batch_size, channels)).astype(FT)

        # Argument list
        np_args = [inputs, forget, initial_state]
        # Argument string name list
        arg_names = ['inputs', 'forget', 'initial_state']
        # Constructor tensorflow variables
        tf_args = [tf.constant(v, name=n) for v, n in zip(np_args, arg_names)]

        def _pin_op(device, *tf_args):
            """ Pin operation to device """
            with tf.device(device):
                return fo_pool(*tf_args, time_major=time_major)

        # Pin operation to CPU
        cpu_op = _pin_op("/cpu:0", *tf_args)

        # Run the op on all GPUs
        gpu_ops = [_pin_op(d, *tf_args) for d in self.gpu_devs]

        cpu_result = self.evaluate(cpu_op)
        self.assertEqual(cpu_result.shape, shape)
        gpu_results = self.evaluate(gpu_ops)
        for gpu_result in gpu_results:
            self.assertEqual(gpu_result.shape, shape)
        expected = np_fo_pooling(inputs, forget, initial_state, time_major=time_major)
        self.assertTrue(np.allclose(cpu_result, expected))
        for gpu_result in gpu_results:
            self.assertTrue(np.allclose(gpu_result, expected))

    def test_time_major_fo_pool_grad(self):
        # List of type constraint for testing this operator
        type_permutations = [(np.float32, 1e-2), (np.float64, 1e-4)]

        # Run test with the type combinations above
        for FT, tolerance in type_permutations:
            self._impl_test_time_major_fo_pool_grad(FT, tolerance)

    def _impl_test_time_major_fo_pool_grad(self, FT, tolerance):
        shape = (5, 3, 2)
        np_args = [np.random.random(size=shape).astype(FT),
                   np.random.uniform(0, 1, size=shape).astype(FT),
                   np.random.random(size=shape[1:]).astype(FT)]
        tf_args = [tf.constant(arg, shape=arg.shape, dtype=FT) for arg in np_args]

        def test_func(*args):
            return tf.reduce_sum(tfmiss_ops.miss_time_major_fo_pool(*args))

        for d in ['/cpu:0'] + self.gpu_devs:
            err = 0
            with tf.device(d):
                theoretical, numerical = tf.test.compute_gradient(test_func, tf_args)
                for j_t, j_n in zip(theoretical, numerical):
                    if j_t.size or j_n.size:  # Handle zero size tensors correctly
                        err = np.maximum(err, np.fabs(j_t - j_n).max())
            self.assertLess(err, tolerance)

    def test_batch_major_fo_pool_grad(self):
        # List of type constraint for testing this operator
        type_permutations = [(np.float32, 1e-2), (np.float64, 1e-4)]

        # Run test with the type combinations above
        for FT, tolerance in type_permutations:
            self._impl_test_batch_major_fo_pool_grad(FT, tolerance)

    def _impl_test_batch_major_fo_pool_grad(self, FT, tolerance):
        shape = (3, 5, 2)
        np_args = [np.random.random(size=shape).astype(FT),
                   np.random.uniform(0, 1, size=shape).astype(FT),
                   np.random.random(size=(shape[0], shape[-1])).astype(FT)]
        tf_args = [tf.constant(arg, shape=arg.shape, dtype=FT) for arg in np_args]

        def test_func(*args):
            return tf.reduce_sum(tfmiss_ops.miss_batch_major_fo_pool(*args))

        for d in ['/cpu:0'] + self.gpu_devs:
            err = 0
            with tf.device(d):
                theoretical, numerical = tf.test.compute_gradient(test_func, tf_args)
                for j_t, j_n in zip(theoretical, numerical):
                    if j_t.size or j_n.size:  # Handle zero size tensors correctly
                        err = np.maximum(err, np.fabs(j_t - j_n).max())
            self.assertLess(err, tolerance)


if __name__ == "__main__":
    tf.test.main()
