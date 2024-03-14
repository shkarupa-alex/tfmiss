from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tf_keras import mixed_precision
from tf_keras.src.testing_infra import test_combinations, test_utils
from tfmiss.keras.layers.dcnv2 import DCNv2
from tfmiss.keras.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class DCNv2Test(test_combinations.TestCase):
    def setUp(self):
        super(DCNv2Test, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(DCNv2Test, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 1, 'kernel_size': 1, 'strides': 1, 'padding': 'valid', 'dilation_rate': 1,
                'deformable_groups': 1, 'use_bias': True},
            input_shape=[2, 3, 4, 2],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 3, 4, 1]
        )
        test_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 4, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'dilation_rate': 1,
                'deformable_groups': 2, 'use_bias': True},
            input_shape=[2, 3, 4, 3],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 3, 4, 4]
        )
        test_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 2, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'dilation_rate': 2,
                'deformable_groups': 2, 'use_bias': True},
            input_shape=[2, 3, 4, 3],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 3, 4, 2]
        )
        test_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 2, 'kernel_size': 3, 'strides': 2, 'padding': 'same', 'dilation_rate': 1,
                'deformable_groups': 2, 'use_bias': True},
            input_shape=[2, 3, 4, 3],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 2, 2, 2]
        )
        test_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 1, 'kernel_size': 1, 'strides': 1, 'padding': 'same', 'dilation_rate': 1,
                'deformable_groups': 1, 'use_bias': True},
            input_shape=[2, 1, 1, 3],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 1, 1, 1]
        )

    def test_layer_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            DCNv2,
            kwargs={
                'filters': 4, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'dilation_rate': 1,
                'deformable_groups': 2, 'use_bias': True},
            input_shape=[2, 3, 4, 3],
            input_dtype='float16',
            expected_output_dtype='float16',
            expected_output_shape=[None, 3, 4, 4]
        )

    def test_custom_alignment(self):
        layer_multi_io_test(
            DCNv2,
            kwargs={
                'filters': 4, 'kernel_size': 3, 'strides': 1, 'padding': 'same', 'dilation_rate': 1,
                'deformable_groups': 2, 'use_bias': True, 'custom_alignment': True},
            input_shapes=[(2, 3, 4, 3), (2, 3, 4, 3)],
            input_dtypes=['float32'] * 2,
            expected_output_dtypes=['float32'],
            expected_output_shapes=[(None, 3, 4, 4)]
        )


if __name__ == "__main__":
    tf.test.main()
