import numpy as np
import tensorflow as tf
from tf_keras import layers
from tf_keras.saving import custom_object_scope
from tf_keras.src.testing_infra import test_combinations
from tfmiss.keras.testing_utils import layer_multi_io_test


class OneToManyLayer(layers.Dense):
    def call(self, inputs):
        result = super(OneToManyLayer, self).call(inputs)

        return result, result + result

    def compute_output_shape(self, input_shape):
        result = super(OneToManyLayer, self).compute_output_shape(input_shape)

        return result, result


@test_combinations.run_all_keras_modes
class LayerMultiIOTestTest(test_combinations.TestCase):
    def test_one_to_one(self):
        layer_multi_io_test(
            layers.Dense,
            kwargs={'units': 10},
            input_shapes=[(2, 4)],
            expected_output_shapes=[(None, 10)]
        )
        layer_multi_io_test(
            layers.Dense,
            kwargs={'units': 10},
            input_shapes=[(2, 4)],
            input_dtypes=['float32'],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            layers.Dense,
            kwargs={'units': 10},
            input_shapes=[(2, 4)],
            input_datas=[np.random.random((2, 4)).astype(np.float32)],
            expected_output_dtypes=['float32']
        )

        layer_multi_io_test(
            layers.Dense,
            kwargs={'units': 10, 'dtype': 'float16'},
            input_shapes=[(2, 4)],
            input_dtypes=['float16'],
            input_datas=[np.random.random((2, 4)).astype(np.float16)],
            expected_output_shapes=[(None, 10)],
            expected_output_dtypes=['float16']
        )
        layer_multi_io_test(
            layers.Dense,
            kwargs={'units': 10, 'dtype': 'float16'},
            input_datas=[np.random.random((2, 4))],
            input_dtypes=['float16'],
            expected_output_dtypes=['float16']
        )
        layer_multi_io_test(
            layers.Dense,
            kwargs={'units': 10, 'dtype': 'float16'},
            input_datas=[np.random.random((2, 10)).astype(np.float16)],
            expected_output_dtypes=['float16']
        )

    def test_many_to_one(self):
        layer_multi_io_test(
            layers.Add,
            input_shapes=[(2, 4), (2, 4)],
            expected_output_shapes=[(None, 4)]
        )

    def test_one_to_many(self):
        with custom_object_scope(
                {'OneToManyLayer': OneToManyLayer}):
            layer_multi_io_test(
                OneToManyLayer,
                kwargs={'units': 10},
                input_shapes=[(2, 4)],
                expected_output_dtypes=['float32', 'float32'],
                expected_output_shapes=[(None, 10), (None, 10)]
            )


if __name__ == "__main__":
    tf.test.main()
