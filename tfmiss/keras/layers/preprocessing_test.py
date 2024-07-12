import numpy as np
import tensorflow as tf
from keras.src.testing_infra import test_combinations, test_utils
from tfmiss.keras.layers.preprocessing import WordShape


@test_combinations.run_all_keras_modes
class WordShapeTest(test_combinations.TestCase):
    def setUp(self):
        super(WordShapeTest, self).setUp()
        self.data = np.array(['1234567', 'low', 'UP', 'Title', 'MiX', '27km']).astype('str')
        self.same = np.array(['1', '2', '2', '2', '3', '4']).astype('str')
        self.ccat = np.array(['1', 'z', ' ', '\n', '💥', '❤️']).astype('str')

    def test_has_case(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_HAS_CASE},
            input_data=self.data,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllEqual([
            [0.], [1.], [1.], [1.], [1.], [1.]
        ], outputs)

    def test_lower_case(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_LOWER_CASE},
            input_data=self.data,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllEqual([
            [0.], [1.], [0.], [0.], [0.], [1.]
        ], outputs)

    def test_upper_case(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_UPPER_CASE},
            input_data=self.data,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllEqual([
            [0.], [0.], [1.], [0.], [0.], [0.]
        ], outputs)

    def test_title_case(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_TITLE_CASE},
            input_data=self.data,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllEqual([
            [0.], [0.], [0.], [1.], [0.], [0.]
        ], outputs)

    def test_mixed_case(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_MIXED_CASE},
            input_data=self.data,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllEqual([
            [0.], [0.], [0.], [0.], [1.], [0.]
        ], outputs)

    def test_same_left(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_LEFT_SAME},
            input_data=self.same,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllEqual([
            [0.], [0.], [1.], [1.], [0.], [0.]
        ], outputs)

    def test_same_right(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_RIGHT_SAME},
            input_data=self.same,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllEqual([
            [0.], [1.], [1.], [0.], [0.], [0.]
        ], outputs)

    def test_same_left2(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_LEFT2_SAME},
            input_data=self.same,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllEqual([
            [0.], [0.], [0.], [1.], [0.], [0.]
        ], outputs)

    def test_same_right2(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_RIGHT2_SAME},
            input_data=self.same,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllEqual([
            [0.], [1.], [0.], [0.], [0.], [0.]
        ], outputs)

    def test_char_cat_first(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_CHAR_CAT_FIRST},
            input_data=self.ccat,
            expected_output_dtype='float32',
            expected_output_shape=(None, 29)
        )
        self.assertAllEqual([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        ], outputs)

    def test_char_cat_short(self):
        test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_CHAR_CAT_FIRST, 'char_cat': ['Lu', 'Ll']},
            input_data=self.ccat,
            expected_output_dtype='float32',
            expected_output_shape=(None, 3)
        )

    def test_char_cat_last(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_CHAR_CAT_LAST},
            input_data=self.ccat,
            expected_output_dtype='float32',
            expected_output_shape=(None, 29)
        )
        self.assertAllEqual([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ], outputs)

    def test_length_norm(self):
        outputs = test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_LENGTH_NORM},
            input_data=self.data,
            expected_output_dtype='float32',
            expected_output_shape=(None, 1)
        )
        self.assertAllClose([
            [0.941857], [-0.27579907], [-0.58021307], [0.33302894], [-0.27579907], [0.02861495]
        ], outputs)

    def test_all(self):
        test_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_ALL},
            input_data=self.data,
            expected_output_dtype='float32',
            expected_output_shape=(None, 68)
        )


if __name__ == "__main__":
    tf.test.main()
