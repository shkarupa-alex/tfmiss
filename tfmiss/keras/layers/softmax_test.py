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
            expected_output_dtypes=['float32'],
            expected_output_shapes=[(None, 20)]
        )
        layer_multi_io_test(
            AdaptiveSoftmax,
            kwargs={
                'units': 20,
                'cutoff': [3],
            },
            input_shapes=[(2, 10, 5), (2, 10,)],
            input_dtypes=['float16', 'int32'],
            expected_output_dtypes=['float32'],
            expected_output_shapes=[(None, 10, 20)]
        )
        layer_multi_io_test(
            AdaptiveSoftmax,
            kwargs={
                'units': 20,
                'cutoff': [3],
                'dtype': 'float16'
            },
            input_shapes=[(3, 2, 10, 5), (3, 2, 10,)],
            input_dtypes=['float32', 'int32'],
            expected_output_dtypes=['float32'],
            expected_output_shapes=[(None, 2, 10, 20)]
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        mf16_policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(mf16_policy)
        layer_multi_io_test(
            AdaptiveSoftmax,
            kwargs={
                'units': 20,
                'cutoff': [3],
            },
            input_shapes=[(10, 5), (10,)],
            input_dtypes=['float16', 'int32'],
            expected_output_dtypes=['float32'],
            expected_output_shapes=[(None, 20)]
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_actual_shape_2d(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3])
        inputs = np.random.rand(10, 5)
        targets = np.arange(10, dtype=np.int32)

        result = layer([inputs, targets], training=True)
        self.assertListEqual([10, 20], list(result.shape))

    def test_actual_shape_3d(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3, 8])
        inputs = np.random.rand(2, 10, 64)
        targets = np.arange(20, dtype=np.int32).reshape([2, 10])

        result = layer([inputs, targets], training=True)
        self.assertListEqual([2, 10, 20], list(result.shape))

    def test_loss_and_output_2d_over_batch(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3], loss_reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        inputs = np.random.rand(10, 5)
        targets = np.arange(10, dtype=np.int32)

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
        targets = np.arange(10, dtype=np.int32)

        train_result = layer([inputs, targets], training=True)
        train_sum = np.sum(train_result, axis=-1)
        train_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(train_sum), train_sum)

        eval_result = layer([inputs, targets], training=False)
        eval_sum = np.sum(eval_result, axis=-1)
        eval_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(eval_sum), eval_sum)

        self.assertGreater(eval_loss, train_loss)

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

        model.compile(optimizer='Adam', loss=None, run_eagerly=testing_utils.should_run_eagerly())
        history = model.fit(x=xt, y=None, batch_size=100, epochs=3, validation_data=(xv, None)).history
        predictions = model.predict(x=xp, batch_size=100)
        predictsum = np.sum(predictions, axis=-1)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertGreater(history['val_loss'][0], history['loss'][0])
        self.assertGreater(history['val_loss'][-1], history['loss'][-1])
        self.assertEqual([sample_size // 100, seq_length, num_classes], list(predictions.shape))
        self.assertAllClose(np.ones_like(predictsum), predictsum)

    def test_ragged_input(self):
        layer = AdaptiveSoftmax(units=16, cutoff=[1], factor=2)
        logits_data = tf.ragged.constant([
            [[1.], [2.], [2.]],
            [[0.]],
            [[1.], [2.]]
        ], ragged_rank=1)
        targets_data = tf.ragged.constant([
            [1, 2, 3],
            [8],
            [14, 15]
        ], ragged_rank=1)
        layer([logits_data, targets_data])
        layer.set_weights([
            np.array([[1.] * 2]),
            np.array([[2.] * 8]),
            np.array([3.] * 8),
            np.array([[4.] * 15] * 8),
            np.array([5.] * 15),
        ])

        logit_inputs = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32, ragged=True)
        logit_targets = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, ragged=True)
        outputs = layer([logit_inputs, logit_targets])

        model = tf.keras.Model(inputs=[logit_inputs, logit_targets], outputs=outputs)
        model.run_eagerly = testing_utils.should_run_eagerly()
        outputs = model.predict([logits_data, targets_data])
        self.assertAllClose(
            outputs,
            tf.ragged.constant([
                [
                    [0.5] + [0.03333333134651184] * 15,
                    [0.5] + [0.03333333134651184] * 15,
                    [0.5] + [0.03333333134651184] * 15
                ],
                [
                    [0.5] + [0.03333333134651184] * 15
                ],
                [
                    [0.5] + [0.03333333134651184] * 15,
                    [0.5] + [0.03333333134651184] * 15
                ]
            ], ragged_rank=1)
        )


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
            input_dtypes=['float32', 'int32'],
            expected_output_dtypes=['float32']
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        mf16_policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(mf16_policy)
        layer_multi_io_test(
            NoiseContrastiveEstimation,
            kwargs={
                'units': 11,
                'negatives': 2,
            },
            input_shapes=[(10, 5), (10,)],
            input_dtypes=['float16', 'int32'],
            expected_output_dtypes=['float32']
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_actual_shape_2d(self):
        layer = NoiseContrastiveEstimation(units=20, negatives=5)
        inputs = np.random.rand(10, 5)
        targets = np.arange(10, dtype=np.int32)

        result = layer([inputs, targets], training=True)
        self.assertListEqual([10, 20], list(result.shape))

    def test_actual_shape_3d(self):
        layer = NoiseContrastiveEstimation(units=20, negatives=5)
        inputs = np.random.rand(2, 10, 64)
        targets = np.arange(20, dtype=np.int32).reshape([2, 10])

        result = layer([inputs, targets], training=True)
        self.assertListEqual([2, 10, 20], list(result.shape))

    def test_loss_and_output_2d(self):
        layer = NoiseContrastiveEstimation(units=20, negatives=5)
        inputs = np.random.rand(10, 5)
        targets = np.arange(10, dtype=np.int32)

        train_result = layer([inputs, targets], training=True)
        train_sum = np.sum(train_result, axis=-1)
        train_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(train_sum), train_sum)

        eval_result = layer([inputs, targets], training=False)
        eval_sum = np.sum(eval_result, axis=-1)
        eval_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(eval_sum), eval_sum)

        self.assertGreater(eval_loss, train_loss)

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

        model.compile(optimizer='Adam', loss=None, run_eagerly=testing_utils.should_run_eagerly())
        history = model.fit(x=xt, y=None, batch_size=100, epochs=3, validation_data=(xv, None)).history
        predictions = model.predict(x=xp, batch_size=100)
        predictsum = np.sum(predictions, axis=-1)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertGreater(history['val_loss'][0], history['loss'][0])
        self.assertGreater(history['val_loss'][-1], history['loss'][-1])
        self.assertEqual([sample_size // 100, seq_length, num_classes], list(predictions.shape))
        self.assertAllClose(np.ones_like(predictsum), predictsum)

    def test_with_ragged_input(self):
        layer = NoiseContrastiveEstimation(units=16, negatives=8)
        logits_data = tf.ragged.constant([
            [[1.], [2.], [2.]],
            [[0.]],
            [[1.], [2.]]
        ], ragged_rank=1)
        targets_data = tf.ragged.constant([
            [1, 2, 3],
            [8],
            [14, 15]
        ], ragged_rank=1)
        layer([logits_data, targets_data])
        layer.set_weights([
            np.array([[1.] * 16]),
            np.array([2.] * 16),
        ])

        logit_inputs = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32, ragged=True)
        logit_targets = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, ragged=True)
        outputs = layer([logit_inputs, logit_targets])

        model = tf.keras.Model(inputs=[logit_inputs, logit_targets], outputs=outputs)
        model.run_eagerly = testing_utils.should_run_eagerly()
        outputs = model.predict([logits_data, targets_data])
        self.assertAllClose(
            outputs,
            tf.ragged.constant([
                [[0.0625] * 16, [0.0625] * 16, [0.0625] * 16],
                [[0.0625] * 16],
                [[0.0625] * 16, [0.0625] * 16]
            ], ragged_rank=1)
        )


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
            input_dtypes=['float32', 'int32'],
            expected_output_dtypes=['float32']
        )

        glob_policy = tf.keras.mixed_precision.experimental.global_policy()
        mf16_policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
        tf.keras.mixed_precision.experimental.set_policy(mf16_policy)
        layer_multi_io_test(
            SampledSofmax,
            kwargs={
                'units': 11,
                'negatives': 2,
            },
            input_shapes=[(10, 5), (10,)],
            input_dtypes=['float16', 'int32'],
            expected_output_dtypes=['float32']
        )
        tf.keras.mixed_precision.experimental.set_policy(glob_policy)

    def test_actual_shape_2d(self):
        layer = SampledSofmax(units=20, negatives=5)
        inputs = np.random.rand(10, 5)
        targets = np.arange(10, dtype=np.int32)

        result = layer([inputs, targets], training=True)
        self.assertListEqual([10, 20], list(result.shape))

    def test_actual_shape_3d(self):
        layer = SampledSofmax(units=20, negatives=5)
        inputs = np.random.rand(2, 10, 64)
        targets = np.arange(20, dtype=np.int32).reshape([2, 10])

        result = layer([inputs, targets], training=True)
        self.assertListEqual([2, 10, 20], list(result.shape))

    def test_loss_and_output_2d(self):
        layer = SampledSofmax(units=20, negatives=5)
        inputs = np.random.rand(10, 5)
        targets = np.arange(10, dtype=np.int32)

        train_result = layer([inputs, targets], training=True)
        train_sum = np.sum(train_result, axis=-1)
        train_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(train_sum), train_sum)

        eval_result = layer([inputs, targets], training=False)
        eval_sum = np.sum(eval_result, axis=-1)
        eval_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(eval_sum), eval_sum)

        self.assertGreater(eval_loss, train_loss)

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

        model.compile(optimizer='Adam', loss=None, run_eagerly=testing_utils.should_run_eagerly())
        history = model.fit(x=xt, y=None, batch_size=100, epochs=3, validation_data=(xv, None)).history
        predictions = model.predict(x=xp, batch_size=100)
        predictsum = np.sum(predictions, axis=-1)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        # self.assertGreater(history['val_loss'][0], history['loss'][0])
        # self.assertGreater(history['val_loss'][-1], history['loss'][-1])
        self.assertEqual([sample_size // 100, seq_length, num_classes], list(predictions.shape))
        self.assertAllClose(np.ones_like(predictsum), predictsum)

    def test_with_ragged_input(self):
        layer = SampledSofmax(units=16, negatives=8)
        logits_data = tf.ragged.constant([
            [[1.], [2.], [2.]],
            [[0.]],
            [[1.], [2.]]
        ], ragged_rank=1)
        targets_data = tf.ragged.constant([
            [1, 2, 3],
            [8],
            [14, 15]
        ], ragged_rank=1)
        layer([logits_data, targets_data])
        layer.set_weights([
            np.array([[1.] * 16]),
            np.array([2.] * 16),
        ])

        logit_inputs = tf.keras.layers.Input(shape=(None, 1), dtype=tf.float32, ragged=True)
        logit_targets = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, ragged=True)
        outputs = layer([logit_inputs, logit_targets])

        model = tf.keras.Model(inputs=[logit_inputs, logit_targets], outputs=outputs)
        model.run_eagerly = testing_utils.should_run_eagerly()
        outputs = model.predict([logits_data, targets_data])
        self.assertAllClose(
            outputs,
            tf.ragged.constant([
                [[0.0625] * 16, [0.0625] * 16, [0.0625] * 16],
                [[0.0625] * 16],
                [[0.0625] * 16, [0.0625] * 16]
            ], ragged_rank=1)
        )


if __name__ == "__main__":
    tf.test.main()
