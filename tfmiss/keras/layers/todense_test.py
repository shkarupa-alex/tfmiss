# Inspired by https://github.com/tensorflow/text/blob/master/tensorflow_text/python/keras/layers/todense_test.py
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.src import testing
from tfmiss.keras.layers import ToDense


class ToDenseTest(testing.TestCase):
    def test_ragged_input_pad_and_mask(self):
        input_data = tf.ragged.constant([[1, 2, 3, 4, 5], []])
        expected_mask = np.array([True, False])

        output = ToDense(pad_value=-1, mask=True)(input_data)
        self.assertTrue(hasattr(output, '_keras_mask'))
        self.assertIsNot(output._keras_mask, None)
        self.assertAllEqual(output._keras_mask, expected_mask)

    # TODO: https://github.com/keras-team/keras/issues/18414
    # def test_ragged_input_with_padding(self):
    #     input_data = tf.data.Dataset.from_tensor_slices(
    #         tf.ragged.constant([[1, 2, 3, 4, 5], [2, 3]], 'float32')).batch(2)
    #     expected_output = np.array([[1, 2, 3, 4, 5], [2, 3, 0, 0, 0]], 'float32')
    #
    #     model = test_utils.get_model_from_layers(
    #         [ToDense(0)],
    #         input_shape=(None,),
    #         input_ragged=True,
    #         input_dtype=tf.float32)
    #     model.compile(
    #         optimizer='sgd',
    #         loss='mse',
    #         metrics=['accuracy'])
    #     output = model.predict(input_data)
    #     self.assertAllEqual(output, expected_output)

    # TODO: https://github.com/keras-team/keras/issues/18414
    # @parameterized.named_parameters(*test_util.generate_combinations_with_testcase_name(
    #     layer=[layers.SimpleRNN, layers.GRU, layers.LSTM]))
    # def test_ragged_input_rnn_layer(self, layer):
    #     inputs = tf.data.Dataset.from_tensor_slices(tf.ragged.constant([[1, 2, 3, 4, 5], [5, 6]])).batch(2)
    #
    #     model = test_utils.get_model_from_layers([
    #         ToDense(7, mask=True),
    #         layers.Embedding(8, 16),
    #         layer(16),
    #         layers.Dense(1, activation='sigmoid')],
    #         input_shape=(None,),
    #         input_ragged=True,
    #         input_dtype=tf.int32)
    #     model.compile(
    #         optimizer='rmsprop',
    #         loss='binary_crossentropy',
    #         metrics=['accuracy'])
    #
    #     output = model.predict(inputs)
    #     self.assertAllEqual(np.zeros((2, 1)).shape, output.shape)

    def test_sparse_input_pad_and_mask(self):
        inputs = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
        expected = np.array([True, True, False])

        output = ToDense(pad_value=-1, mask=True)(inputs)
        self.assertTrue(hasattr(output, '_keras_mask'))
        self.assertIsNot(output._keras_mask, None)
        self.assertAllEqual(output._keras_mask, expected)

    def test_sparse_input_with_padding(self):
        inputs = tf.data.Dataset.from_tensor_slices(
            tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1., 2.], dense_shape=[3, 4])).batch(3)
        expected = np.array([[1., -1., -1., -1.], [-1., -1., 2., -1.], [-1., -1., -1., -1.]])

        model = models.Sequential(
            [layers.Input(shape=(None,), dtype='float32', sparse=True), ToDense(pad_value=-1., trainable=False)])
        model.compile(
            optimizer='sgd',
            loss='mse',
            metrics=['accuracy'])
        output = model.predict(inputs)
        self.assertAllClose(output, expected)


if __name__ == '__main__':
    tf.test.main()
