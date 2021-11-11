from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from keras import layers, models, keras_parameterized, testing_utils
from tfmiss.keras.layers.reduction import Reduction


@keras_parameterized.run_all_keras_modes
class ReductionTest(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            Reduction,
            kwargs={'reduction': 'mean', 'axis': -2},
            input_shape=(2, 10, 5),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5)
        )

    @parameterized.named_parameters(
        {
            "testcase_name": "max",
            "reduction_str": "max",
            "expected_output": [[3.0, 3.0], [3.0, 2.0]]
        }, {
            "testcase_name": "mean",
            "reduction_str": "mean",
            "expected_output": [[2.0, 2.0], [2.0, 1.5]]
        }, {
            "testcase_name": "min",
            "reduction_str": "min",
            "expected_output": [[1.0, 1.0], [1.0, 1.0]]
        }, {
            "testcase_name": "prod",
            "reduction_str": "prod",
            "expected_output": [[6.0, 6.0], [3.0, 2.0]]
        }, {
            "testcase_name": "sum",
            "reduction_str": "sum",
            "expected_output": [[6.0, 6.0], [4.0, 3.0]]
        }, {
            "testcase_name": "sqrtn",
            "reduction_str": "sqrtn",
            "expected_output": [[3.4641016, 3.4641016], [2.8284271, 2.1213203]]
        })
    def test_ragged_reduction(self, reduction_str, expected_output):
        data = tf.ragged.constant([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[3.0, 1.0], [1.0, 2.0]]])
        input_tensor = layers.Input(shape=(None, None), ragged=True)

        output_tensor = Reduction(reduction=reduction_str)(input_tensor)
        model = models.Model(input_tensor, output_tensor)

        output = model.predict(data)

        self.assertAllClose(expected_output, output)

    @parameterized.named_parameters(
        {
            "testcase_name": "max",
            "reduction_str": "max",
            "expected_output": [[3.0, 3.0], [3.0, 2.0]]
        }, {
            "testcase_name": "mean",
            "reduction_str": "mean",
            "expected_output": [[2.0, 2.0], [1.333333, 1.0]]
        }, {
            "testcase_name": "min",
            "reduction_str": "min",
            "expected_output": [[1.0, 1.0], [0.0, 0.0]]
        }, {
            "testcase_name": "prod",
            "reduction_str": "prod",
            "expected_output": [[6.0, 6.0], [0.0, 0.0]]
        }, {
            "testcase_name": "sum",
            "reduction_str": "sum",
            "expected_output": [[6.0, 6.0], [4.0, 3.0]]
        }, {
            "testcase_name": "sqrtn",
            "reduction_str": "sqrtn",
            "expected_output": [[3.4641016, 3.4641016], [2.3094011, 1.7320508]]
        })
    def test_dense_reduction(self, reduction_str, expected_output):
        data = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [[3.0, 1.0], [1.0, 2.0], [0.0, 0.0]]])
        input_tensor = layers.Input(shape=(None, None))

        output_tensor = Reduction(reduction=reduction_str)(input_tensor)
        model = models.Model(input_tensor, output_tensor)

        output = model.predict(data)

        self.assertAllClose(expected_output, output)


if __name__ == "__main__":
    tf.test.main()
