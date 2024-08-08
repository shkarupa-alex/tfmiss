import tensorflow as tf
from keras.src import testing
from keras.src.dtype_policies import dtype_policy

from tfmiss.keras.layers.dcnv2 import DCNv2


class DCNv2Test(testing.TestCase):
    def setUp(self):
        super(DCNv2Test, self).setUp()
        self.default_policy = dtype_policy.dtype_policy()

    def tearDown(self):
        super(DCNv2Test, self).tearDown()
        dtype_policy.set_dtype_policy(self.default_policy)

    def test_layer(self):
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 1,
                "kernel_size": 1,
                "strides": 1,
                "padding": "valid",
                "dilation_rate": 1,
                "deformable_groups": 1,
                "use_bias": True,
            },
            input_shape=(2, 3, 4, 2),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 4, 1),
        )
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "strides": 1,
                "padding": "same",
                "dilation_rate": 1,
                "deformable_groups": 2,
                "use_bias": True,
            },
            input_shape=(2, 3, 4, 3),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 4, 4),
        )
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 2,
                "kernel_size": 3,
                "strides": 1,
                "padding": "same",
                "dilation_rate": 2,
                "deformable_groups": 2,
                "use_bias": True,
            },
            input_shape=(2, 3, 4, 3),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 4, 2),
        )
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 2,
                "kernel_size": 3,
                "strides": 2,
                "padding": "same",
                "dilation_rate": 1,
                "deformable_groups": 2,
                "use_bias": True,
            },
            input_shape=(2, 3, 4, 3),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 2, 2, 2),
        )
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 1,
                "kernel_size": 1,
                "strides": 1,
                "padding": "same",
                "dilation_rate": 1,
                "deformable_groups": 1,
                "use_bias": True,
            },
            input_shape=(2, 1, 1, 3),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 1, 1, 1),
        )

    def test_layer_fp16(self):
        dtype_policy.set_dtype_policy("mixed_float16")
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "strides": 1,
                "padding": "same",
                "dilation_rate": 1,
                "deformable_groups": 2,
                "use_bias": True,
            },
            input_shape=(2, 3, 4, 3),
            input_dtype="float16",
            expected_output_dtype="float16",
            expected_output_shape=(2, 3, 4, 4),
        )

    def test_custom_alignment(self):
        self.run_layer_test(
            DCNv2,
            init_kwargs={
                "filters": 4,
                "kernel_size": 3,
                "strides": 1,
                "padding": "same",
                "dilation_rate": 1,
                "deformable_groups": 2,
                "use_bias": True,
                "custom_alignment": True,
            },
            input_shape=((2, 3, 4, 3), (2, 3, 4, 3)),
            input_dtype=("float32", "float32"),
            expected_output_dtype="float32",
            expected_output_shape=(2, 3, 4, 4),
        )


if __name__ == "__main__":
    tf.test.main()
