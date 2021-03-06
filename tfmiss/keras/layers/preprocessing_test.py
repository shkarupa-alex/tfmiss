from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tfmiss.keras.layers.preprocessing import CharNgams, WordShape


@keras_parameterized.run_all_keras_modes
class CharNgamsTest(keras_parameterized.TestCase):
    # TODO: can't test ragged output
    # def test_layer(self):
    #     testing_utils.layer_test(
    #         CharNgams,
    #         kwargs={'minn': 2, 'maxn': 4, 'itself': 'alone'},
    #         input_data=np.array([
    #             ['abc', 'defg', 'hi'],
    #             ['a', 'bcdef', 'ghi'],
    #             ['abc', 'defg', 'hi'],
    #             ['a', 'bcdef', 'ghi'],
    #         ]).astype('str'),
    #         expected_output_dtype='string',
    #         expected_output_shape=(None, None, None)
    #     )

    def test_output(self):
        inputs = tf.constant([
            ['abc', 'defg', 'hi'],
            ['a', 'bcdef', 'ghi'],
            ['abc', 'defg', 'hi'],
            ['a', 'bcdef', 'ghi'],
        ], dtype=tf.string)
        layer = CharNgams(minn=2, maxn=4, itself='alone')
        outputs = layer(inputs)
        self.assertListEqual([4, 3, None], outputs.shape.as_list())
        self.assertEqual(tf.string, outputs.dtype)

        outputs = self.evaluate(outputs)
        self.assertTupleEqual((4, None, None), outputs.shape)


@keras_parameterized.run_all_keras_modes
class WordShapeTest(keras_parameterized.TestCase):
    def setUp(self):
        super(WordShapeTest, self).setUp()
        self.data = np.array(['1234567', 'low', 'UP', 'Title', 'MiX', '27km']).astype('str')
        self.same = np.array(['1', '2', '2', '2', '3', '4']).astype('str')
        self.ccat = np.array(['1', 'z', ' ', '\n', '💥', '❤️']).astype('str')

    def test_has_case(self):
        outputs = testing_utils.layer_test(
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
        outputs = testing_utils.layer_test(
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
        outputs = testing_utils.layer_test(
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
        outputs = testing_utils.layer_test(
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
        outputs = testing_utils.layer_test(
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
        outputs = testing_utils.layer_test(
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
        outputs = testing_utils.layer_test(
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
        outputs = testing_utils.layer_test(
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
        outputs = testing_utils.layer_test(
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
        outputs = testing_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_CHAR_CAT_FIRST},
            input_data=self.ccat,
            expected_output_dtype='float32',
            expected_output_shape=(None, 30)
        )
        self.assertAllEqual([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
        ], outputs)

    def test_char_cat_last(self):
        outputs = testing_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_CHAR_CAT_LAST},
            input_data=self.ccat,
            expected_output_dtype='float32',
            expected_output_shape=(None, 30)
        )
        self.assertAllEqual([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        ], outputs)

    def test_length_norm(self):
        outputs = testing_utils.layer_test(
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
        testing_utils.layer_test(
            WordShape,
            kwargs={'options': WordShape.SHAPE_ALL},
            input_data=self.data,
            expected_output_dtype='float32',
            expected_output_shape=(None, 70)
        )


if __name__ == "__main__":
    tf.test.main()
