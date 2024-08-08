import tensorflow as tf
from keras.src import testing

from tfmiss.keras.layers.preprocessing import WordShape


class WordShapeTest(testing.TestCase):
    def setUp(self):
        super(WordShapeTest, self).setUp()
        self.data = tf.constant(
            ["1234567", "low", "UP", "Title", "MiX", "27km"]
        )
        self.same = tf.constant(["1", "2", "2", "2", "3", "4"])
        self.ccat = tf.constant(["1", "z", " ", "\n", "💥", "❤️"])

    def test_has_case(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_HAS_CASE},
            input_data=self.data,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_HAS_CASE)(self.data)
        self.assertAllEqual([[0.0], [1.0], [1.0], [1.0], [1.0], [1.0]], outputs)

    def test_lower_case(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_LOWER_CASE},
            input_data=self.data,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_LOWER_CASE)(self.data)
        self.assertAllEqual([[0.0], [1.0], [0.0], [0.0], [0.0], [1.0]], outputs)

    def test_upper_case(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_UPPER_CASE},
            input_data=self.data,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_UPPER_CASE)(self.data)
        self.assertAllEqual([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]], outputs)

    def test_title_case(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_TITLE_CASE},
            input_data=self.data,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_TITLE_CASE)(self.data)
        self.assertAllEqual([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]], outputs)

    def test_mixed_case(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_MIXED_CASE},
            input_data=self.data,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_MIXED_CASE)(self.data)
        self.assertAllEqual([[0.0], [0.0], [0.0], [0.0], [1.0], [0.0]], outputs)

    def test_same_left(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_LEFT_SAME},
            input_data=self.same,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_LEFT_SAME)(self.same)
        self.assertAllEqual([[0.0], [0.0], [1.0], [1.0], [0.0], [0.0]], outputs)

    def test_same_right(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_RIGHT_SAME},
            input_data=self.same,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_RIGHT_SAME)(self.same)
        self.assertAllEqual([[0.0], [1.0], [1.0], [0.0], [0.0], [0.0]], outputs)

    def test_same_left2(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_LEFT2_SAME},
            input_data=self.same,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_LEFT2_SAME)(self.same)
        self.assertAllEqual([[0.0], [0.0], [0.0], [1.0], [0.0], [0.0]], outputs)

    def test_same_right2(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_RIGHT2_SAME},
            input_data=self.same,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_RIGHT2_SAME)(self.same)
        self.assertAllEqual([[0.0], [1.0], [0.0], [0.0], [0.0], [0.0]], outputs)

    def test_char_cat_first(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_CHAR_CAT_FIRST},
            input_data=self.ccat,
            expected_output_dtype="float32",
            expected_output_shape=(6, 29),
        )

        outputs = WordShape(WordShape.SHAPE_CHAR_CAT_FIRST)(self.ccat)
        self.assertAllEqual(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                ],
            ],
            outputs,
        )

    def test_char_cat_short(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={
                "options": WordShape.SHAPE_CHAR_CAT_FIRST,
                "char_cat": ["Lu", "Ll"],
            },
            input_data=self.ccat,
            expected_output_dtype="float32",
            expected_output_shape=(6, 3),
        )

    def test_char_cat_last(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_CHAR_CAT_LAST},
            input_data=self.ccat,
            expected_output_dtype="float32",
            expected_output_shape=(6, 29),
        )

        outputs = WordShape(WordShape.SHAPE_CHAR_CAT_LAST)(self.ccat)
        self.assertAllEqual(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
            ],
            outputs,
        )

    def test_length_norm(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_LENGTH_NORM},
            input_data=self.data,
            expected_output_dtype="float32",
            expected_output_shape=(6, 1),
        )

        outputs = WordShape(WordShape.SHAPE_LENGTH_NORM)(self.data)
        self.assertAllClose(
            [
                [0.941857],
                [-0.27579907],
                [-0.58021307],
                [0.33302894],
                [-0.27579907],
                [0.02861495],
            ],
            outputs,
        )

    def test_all(self):
        self.run_layer_test(
            WordShape,
            init_kwargs={"options": WordShape.SHAPE_ALL},
            input_data=self.data,
            expected_output_dtype="float32",
            expected_output_shape=(6, 68),
        )


if __name__ == "__main__":
    tf.test.main()
