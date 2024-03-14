from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_keras.src.testing_infra import test_combinations, test_utils
from tfmiss.keras.layers.dropout import TimestepDropout


@test_combinations.run_all_keras_modes
class TimestepDropoutTest(test_combinations.TestCase):
    def test_layer(self):
        test_utils.layer_test(
            TimestepDropout,
            kwargs={'rate': .1},
            input_shape=[2, 16, 8],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 16, 8]
        )


if __name__ == "__main__":
    tf.test.main()
