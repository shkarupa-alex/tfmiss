from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tfmiss.keras.layers.softmax import AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax
from tfmiss.keras.testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class AdaptiveSoftmaxTest(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            AdaptiveSoftmax,
            kwargs={
                'units': 20,
                'cutoff': [3],
            },
            input_shapes=[(10, 5), (10,)],
            input_dtypes=['float32', 'int32'],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            AdaptiveSoftmax,
            kwargs={
                'units': 20,
                'cutoff': [3],
                'dtype': 'float16'
            },
            input_shapes=[(2, 10, 5), (2, 10,)],
            input_dtypes=['float16', 'int32'],
            expected_output_dtypes=['float16']
        )
        layer_multi_io_test(
            AdaptiveSoftmax,
            kwargs={
                'units': 20,
                'cutoff': [3],
                'dtype': 'float16'
            },
            input_shapes=[(3, 2, 10, 5), (3, 2, 10,)],
            input_dtypes=['float32', 'int64'],
            expected_output_dtypes=['float16']
        )

    def test_actual_shape_2d(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3])
        inputs = np.random.rand(10, 5)
        targets = np.arange(10)

        result = layer([inputs, targets], training=True)
        self.assertListEqual([10, 20], list(result.shape))

    def test_actual_shape_3d(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3, 8])
        inputs = np.random.rand(2, 10, 64)
        targets = np.arange(20).reshape([2, 10])

        result = layer([inputs, targets], training=True)
        self.assertListEqual([2, 10, 20], list(result.shape))

    def test_loss_and_output_2d_over_batch(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3], loss_reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        inputs = np.random.rand(10, 5)
        targets = np.arange(10)

        train_result = layer([inputs, targets], training=True)
        train_sum = np.sum(train_result, axis=-1)
        train_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(train_sum), train_sum)

        eval_result = layer([inputs, targets], training=False)
        eval_sum = np.sum(eval_result, axis=-1)
        eval_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(eval_sum), eval_sum)

        self.assertGreater(eval_loss, train_loss)

    def test_loss_and_output_2d_sum(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3], loss_reduction=tf.keras.losses.Reduction.SUM)
        inputs = np.random.rand(10, 5)
        targets = np.arange(10)

        train_result = layer([inputs, targets], training=True)
        train_sum = np.sum(train_result, axis=-1)
        train_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(train_sum), train_sum)

        eval_result = layer([inputs, targets], training=False)
        eval_sum = np.sum(eval_result, axis=-1)
        eval_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(eval_sum), eval_sum)

        self.assertGreater(eval_loss, train_loss)

    def test_no_errors_3d(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3])
        inputs = np.random.rand(2, 4, 5)
        targets = np.arange(2 * 4).reshape([2, 4])

        layer([inputs, targets], training=True)
        layer([inputs, targets], training=False)

    def test_model(self):
        num_samples = 10000
        seq_length = 5
        num_classes = 99
        embed_size = 10
        sample_size = 1000

        xt = [np.random.randint(num_samples - 1, size=(sample_size, seq_length)),
              np.ones((sample_size, seq_length)).astype(np.int32)]
        xv = [np.random.randint(num_samples - 1, size=(sample_size // 100, seq_length)),
              np.ones((sample_size // 100, seq_length)).astype(np.int32)]
        xp = [np.random.randint(num_samples - 1, size=(sample_size // 100, seq_length)),
              np.zeros((sample_size // 100, seq_length)).astype(np.int32)]

        ids = tf.keras.layers.Input(shape=(None,), dtype='int32')
        targets = tf.keras.layers.Input(shape=(None,), dtype='int32')

        embeddings = tf.keras.layers.Embedding(input_dim=num_samples, output_dim=embed_size)(ids)
        logits = tf.keras.layers.Dense(embed_size // 2, activation='relu')(embeddings)
        probs = AdaptiveSoftmax(units=num_classes, cutoff=[3])([logits, targets])
        model = tf.keras.Model(inputs=[ids, targets], outputs=probs)

        model.compile(
            optimizer='Adam',
            loss=None,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function()
        )
        history = model.fit(x=xt, y=None, batch_size=100, epochs=3, validation_data=(xv, None)).history
        predictions = model.predict(x=xp, batch_size=100)
        predictsum = np.sum(predictions, axis=-1)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertEqual([sample_size // 100, seq_length, num_classes], list(predictions.shape))
        self.assertAllClose(np.ones_like(predictsum), predictsum)


@keras_parameterized.run_all_keras_modes
class NoiseContrastiveEstimationTest(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            NoiseContrastiveEstimation,
            kwargs={
                'units': 11,
                'negatives': 2,
            },
            input_shapes=[(10, 5), (10,)],
            input_dtypes=['float32', 'int32'],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            NoiseContrastiveEstimation,
            kwargs={
                'units': 11,
                'negatives': 2,
            },
            input_shapes=[(2, 10, 5), (2, 10,)],
            input_dtypes=['float32', 'int32'],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            NoiseContrastiveEstimation,
            kwargs={
                'units': 11,
                'negatives': 2,
            },
            input_shapes=[(3, 2, 10, 5), (3, 2, 10,)],
            input_dtypes=['float32', 'int64'],
            expected_output_dtypes=['float32']
        )

    def test_model(self):
        num_samples = 10000
        seq_length = 5
        num_classes = 99
        embed_size = 10
        sample_size = 1000

        xt = [np.random.randint(num_samples - 1, size=(sample_size, seq_length)),
              np.ones((sample_size, seq_length)).astype(np.int32)]
        xv = [np.random.randint(num_samples - 1, size=(sample_size // 100, seq_length)),
              np.ones((sample_size // 100, seq_length)).astype(np.int32)]
        xp = [np.random.randint(num_samples - 1, size=(sample_size // 100, seq_length)),
              np.zeros((sample_size // 100, seq_length)).astype(np.int32)]

        ids = tf.keras.layers.Input(shape=(None,), dtype='int32')
        targets = tf.keras.layers.Input(shape=(None,), dtype='int32')

        embeddings = tf.keras.layers.Embedding(input_dim=num_samples, output_dim=embed_size)(ids)
        logits = tf.keras.layers.Dense(embed_size // 2, activation='relu')(embeddings)
        probs = NoiseContrastiveEstimation(units=num_classes, negatives=num_classes // 2)([logits, targets])
        model = tf.keras.Model(inputs=[ids, targets], outputs=probs)

        model.compile(
            optimizer='Adam',
            loss=None,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function()
        )
        history = model.fit(x=xt, y=None, batch_size=100, epochs=3, validation_data=(xv, None)).history
        predictions = model.predict(x=xp, batch_size=100)
        predictsum = np.sum(predictions, axis=-1)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertEqual([sample_size // 100, seq_length, num_classes], list(predictions.shape))
        self.assertAllClose(np.ones_like(predictsum), predictsum)


@keras_parameterized.run_all_keras_modes
class SampledSofmaxTest(keras_parameterized.TestCase):
    def test_layer(self):
        layer_multi_io_test(
            SampledSofmax,
            kwargs={
                'units': 11,
                'negatives': 2,
            },
            input_shapes=[(10, 5), (10,)],
            input_dtypes=['float32', 'int32'],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            SampledSofmax,
            kwargs={
                'units': 11,
                'negatives': 2,
            },
            input_shapes=[(2, 10, 5), (2, 10,)],
            input_dtypes=['float32', 'int32'],
            expected_output_dtypes=['float32']
        )
        layer_multi_io_test(
            SampledSofmax,
            kwargs={
                'units': 11,
                'negatives': 2,
            },
            input_shapes=[(3, 2, 10, 5), (3, 2, 10,)],
            input_dtypes=['float32', 'int64'],
            expected_output_dtypes=['float32']
        )

    def test_model(self):
        num_samples = 10000
        seq_length = 5
        num_classes = 99
        embed_size = 10
        sample_size = 1000

        xt = [np.random.randint(num_samples - 1, size=(sample_size, seq_length)),
              np.ones((sample_size, seq_length)).astype(np.int32)]
        xv = [np.random.randint(num_samples - 1, size=(sample_size // 100, seq_length)),
              np.ones((sample_size // 100, seq_length)).astype(np.int32)]
        xp = [np.random.randint(num_samples - 1, size=(sample_size // 100, seq_length)),
              np.zeros((sample_size // 100, seq_length)).astype(np.int32)]

        ids = tf.keras.layers.Input(shape=(None,), dtype='int32')
        targets = tf.keras.layers.Input(shape=(None,), dtype='int32')

        embeddings = tf.keras.layers.Embedding(input_dim=num_samples, output_dim=embed_size)(ids)
        logits = tf.keras.layers.Dense(embed_size // 2, activation='relu')(embeddings)
        probs = SampledSofmax(units=num_classes, negatives=num_classes // 2)([logits, targets])
        model = tf.keras.Model(inputs=[ids, targets], outputs=probs)

        model.compile(
            optimizer='Adam',
            loss=None,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function()
        )
        history = model.fit(x=xt, y=None, batch_size=100, epochs=3, validation_data=(xv, None)).history
        predictions = model.predict(x=xp, batch_size=100)
        predictsum = np.sum(predictions, axis=-1)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertEqual([sample_size // 100, seq_length, num_classes], list(predictions.shape))
        self.assertAllClose(np.ones_like(predictsum), predictsum)


if __name__ == "__main__":
    tf.test.main()
