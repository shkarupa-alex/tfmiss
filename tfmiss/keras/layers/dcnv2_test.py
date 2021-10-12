from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import keras_parameterized, testing_utils
from tfmiss.keras.layers.dcnv2 import DCNv2


@keras_parameterized.run_all_keras_modes
class DCNv2Test(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 1, 'kernel_size': 1, 'strides': 1, 'padding': 'valid', 'dilation_rate': 1,
                'deformable_groups': 1, 'use_bias': True},
            input_shape=[2, 3, 4, 2],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 3, 4, 1]
        )
        testing_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 4, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'dilation_rate': 1,
                'deformable_groups': 2, 'use_bias': True},
            input_shape=[2, 3, 4, 3],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 3, 4, 4]
        )
        testing_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 2, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'dilation_rate': 2,
                'deformable_groups': 2, 'use_bias': True},
            input_shape=[2, 3, 4, 3],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 3, 4, 2]
        )
        testing_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 2, 'kernel_size': 3, 'strides': 2, 'padding': 'same', 'dilation_rate': 1,
                'deformable_groups': 2, 'use_bias': True},
            input_shape=[2, 3, 4, 3],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 2, 2, 2]
        )
        testing_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 1, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'dilation_rate': 1,
                'deformable_groups': 1, 'use_bias': True},
            input_shape=[2, 1, 1, 3],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 1, 1, 1]
        )


if __name__ == "__main__":
    tf.test.main()
