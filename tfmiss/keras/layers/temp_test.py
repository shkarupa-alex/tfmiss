from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfmiss.keras.layers.temp import Reduction


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


if __name__ == "__main__":
    tf.test.main()
