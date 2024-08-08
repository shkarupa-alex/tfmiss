import tensorflow as tf
from keras.src import testing

from tfmiss.keras.layers.dropout import TimestepDropout


class TimestepDropoutTest(testing.TestCase):
    def test_layer(self):
        self.run_layer_test(
            TimestepDropout,
            init_kwargs={"rate": 0.1},
            input_shape=(2, 16, 8),
            input_dtype="float32",
            expected_output_dtype="float32",
            expected_output_shape=(2, 16, 8),
        )


if __name__ == "__main__":
    tf.test.main()
