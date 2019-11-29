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
        with tf.keras.utils.custom_object_scope({'AdaptiveSoftmax': AdaptiveSoftmax}):
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

    def test_loss_and_output_2d(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3])
        inputs = np.random.rand(10, 5)
        targets = np.arange(10)

        train_result = layer([inputs, targets], training=True)
        train_loss = layer.losses[0]
        self.assertAlmostEqual(1., np.sum(train_result[0]), places=3)

        eval_result = layer([inputs, targets], training=False)
        eval_loss = layer.losses[0]
        self.assertAlmostEqual(1., np.sum(eval_result[0]), places=3)

        self.assertGreater(np.sum(eval_loss), np.sum(train_loss))

    def test_no_errors_3d(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3])
        inputs = np.random.rand(2, 4, 5)
        targets = np.arange(2 * 4).reshape([2, 4])

        layer([inputs, targets], training=True)
        layer([inputs, targets], training=False)

    def test_model(self):
        NUM_SAMPLES = 10000
        SEQ_LENGTH = 5
        NUM_CLASSES = 99
        EMBED_SIZE = 10
        SAMPLE_SIZE = 1000

        Xt = [np.random.randint(NUM_SAMPLES - 1, size=(SAMPLE_SIZE, SEQ_LENGTH)),
              np.ones((SAMPLE_SIZE, SEQ_LENGTH)).astype(np.int32)]
        Xv = [np.random.randint(NUM_SAMPLES - 1, size=(SAMPLE_SIZE // 100, SEQ_LENGTH)),
              np.ones((SAMPLE_SIZE // 100, SEQ_LENGTH)).astype(np.int32)]
        Xp = [np.random.randint(NUM_SAMPLES - 1, size=(SAMPLE_SIZE // 100, SEQ_LENGTH)),
              np.zeros((SAMPLE_SIZE // 100, SEQ_LENGTH)).astype(np.int32)]

        ids = tf.keras.layers.Input(shape=(None,), dtype='int32')
        targets = tf.keras.layers.Input(shape=(None,), dtype='int32')

        embeddings = tf.keras.layers.Embedding(input_dim=NUM_SAMPLES, output_dim=EMBED_SIZE)(ids)
        logits = tf.keras.layers.Dense(EMBED_SIZE // 2, activation='relu')(embeddings)
        probs = AdaptiveSoftmax(units=NUM_CLASSES, cutoff=[3])([logits, targets])
        model = tf.keras.Model(inputs=[ids, targets], outputs=probs)

        model.compile(
            optimizer='Adam',
            loss=None,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function()
        )
        history = model.fit(x=Xt, y=None, batch_size=100, epochs=3, validation_data=(Xv, None)).history
        predictions = model.predict(x=Xp, batch_size=100)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertEqual([SAMPLE_SIZE // 100, SEQ_LENGTH, NUM_CLASSES], list(predictions.shape))
        self.assertAlmostEqual(1., np.sum(predictions[0][0]), places=3)


@keras_parameterized.run_all_keras_modes
class NoiseContrastiveEstimationTest(keras_parameterized.TestCase):
    def test_layer(self):
        with tf.keras.utils.custom_object_scope({'NoiseContrastiveEstimation': NoiseContrastiveEstimation}):
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
        NUM_SAMPLES = 10000
        SEQ_LENGTH = 5
        NUM_CLASSES = 99
        EMBED_SIZE = 10
        SAMPLE_SIZE = 1000

        Xt = [np.random.randint(NUM_SAMPLES - 1, size=(SAMPLE_SIZE, SEQ_LENGTH)),
              np.ones((SAMPLE_SIZE, SEQ_LENGTH)).astype(np.int32)]
        Xv = [np.random.randint(NUM_SAMPLES - 1, size=(SAMPLE_SIZE // 100, SEQ_LENGTH)),
              np.ones((SAMPLE_SIZE // 100, SEQ_LENGTH)).astype(np.int32)]
        Xp = [np.random.randint(NUM_SAMPLES - 1, size=(SAMPLE_SIZE // 100, SEQ_LENGTH)),
              np.zeros((SAMPLE_SIZE // 100, SEQ_LENGTH)).astype(np.int32)]

        ids = tf.keras.layers.Input(shape=(None,), dtype='int32')
        targets = tf.keras.layers.Input(shape=(None,), dtype='int32')

        embeddings = tf.keras.layers.Embedding(input_dim=NUM_SAMPLES, output_dim=EMBED_SIZE)(ids)
        logits = tf.keras.layers.Dense(EMBED_SIZE // 2, activation='relu')(embeddings)
        probs = NoiseContrastiveEstimation(units=NUM_CLASSES, negatives=NUM_CLASSES // 2)([logits, targets])
        model = tf.keras.Model(inputs=[ids, targets], outputs=probs)

        model.compile(
            optimizer='Adam',
            loss=None,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function()
        )
        history = model.fit(x=Xt, y=None, batch_size=100, epochs=3, validation_data=(Xv, None)).history
        predictions = model.predict(x=Xp, batch_size=100)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertEqual([SAMPLE_SIZE // 100, SEQ_LENGTH, NUM_CLASSES], list(predictions.shape))
        self.assertAlmostEqual(1., np.sum(predictions[0][0]), places=3)


@keras_parameterized.run_all_keras_modes
class SampledSofmaxTest(keras_parameterized.TestCase):
    def test_layer(self):
        with tf.keras.utils.custom_object_scope({'SampledSofmax': SampledSofmax}):
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
        NUM_SAMPLES = 10000
        SEQ_LENGTH = 5
        NUM_CLASSES = 99
        EMBED_SIZE = 10
        SAMPLE_SIZE = 1000

        Xt = [np.random.randint(NUM_SAMPLES - 1, size=(SAMPLE_SIZE, SEQ_LENGTH)),
              np.ones((SAMPLE_SIZE, SEQ_LENGTH)).astype(np.int32)]
        Xv = [np.random.randint(NUM_SAMPLES - 1, size=(SAMPLE_SIZE // 100, SEQ_LENGTH)),
              np.ones((SAMPLE_SIZE // 100, SEQ_LENGTH)).astype(np.int32)]
        Xp = [np.random.randint(NUM_SAMPLES - 1, size=(SAMPLE_SIZE // 100, SEQ_LENGTH)),
              np.zeros((SAMPLE_SIZE // 100, SEQ_LENGTH)).astype(np.int32)]

        ids = tf.keras.layers.Input(shape=(None,), dtype='int32')
        targets = tf.keras.layers.Input(shape=(None,), dtype='int32')

        embeddings = tf.keras.layers.Embedding(input_dim=NUM_SAMPLES, output_dim=EMBED_SIZE)(ids)
        logits = tf.keras.layers.Dense(EMBED_SIZE // 2, activation='relu')(embeddings)
        probs = SampledSofmax(units=NUM_CLASSES, negatives=NUM_CLASSES // 2)([logits, targets])
        model = tf.keras.Model(inputs=[ids, targets], outputs=probs)

        model.compile(
            optimizer='Adam',
            loss=None,
            run_eagerly=testing_utils.should_run_eagerly(),
            experimental_run_tf_function=testing_utils.should_run_tf_function()
        )
        history = model.fit(x=Xt, y=None, batch_size=100, epochs=3, validation_data=(Xv, None)).history
        predictions = model.predict(x=Xp, batch_size=100)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertEqual([SAMPLE_SIZE // 100, SEQ_LENGTH, NUM_CLASSES], list(predictions.shape))
        self.assertAlmostEqual(1., np.sum(predictions[0][0]), places=3)


if __name__ == "__main__":
    tf.test.main()
