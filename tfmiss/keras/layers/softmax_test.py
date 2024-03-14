import numpy as np
import tensorflow as tf
from keras import layers, mixed_precision, models, utils
from keras.src import testing
from tfmiss.keras.layers.softmax import AdaptiveSoftmax, NoiseContrastiveEstimation, SampledSofmax


class AdaptiveSoftmaxTest(testing.TestCase):
    def setUp(self):
        super(AdaptiveSoftmaxTest, self).setUp()
        self.default_policy = mixed_precision.global_policy()
        self.mf16_policy = mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(AdaptiveSoftmaxTest, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        self.run_layer_test(
            AdaptiveSoftmax,
            init_kwargs={'units': 20, 'cutoff': [2], 'return_probs': True},
            input_shape=((10, 5), (10,)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(10, 20)
        )
        self.run_layer_test(
            AdaptiveSoftmax,
            init_kwargs={'units': 20, 'cutoff': [2], 'return_probs': True},
            input_shape=((2, 10, 5), (2, 10,)),
            input_dtype=('float16', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(2, 10, 20)
        )
        self.run_layer_test(
            AdaptiveSoftmax,
            init_kwargs={'units': 20, 'cutoff': [2], 'dtype': 'float16', 'return_probs': True},
            input_shape=((3, 2, 10, 5), (3, 2, 10)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(3, 2, 10, 20)
        )
        self.run_layer_test(
            AdaptiveSoftmax,
            init_kwargs={'units': 20, 'cutoff': [2], 'return_probs': False},
            input_shape=((2, 10, 5), (2, 10,)),
            input_dtype=('float16', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(2, 10, 5)
        )

        mixed_precision.set_global_policy(self.mf16_policy)
        self.run_layer_test(
            AdaptiveSoftmax,
            init_kwargs={'units': 20, 'cutoff': [2], 'return_probs': True},
            input_shape=((10, 5), (10,)),
            input_dtype=('float16', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(10, 20)
        )
        self.run_layer_test(
            AdaptiveSoftmax,
            init_kwargs={'units': 20, 'cutoff': [2], 'return_probs': False},
            input_shape=((10, 5), (10,)),
            input_dtype=('float16', 'int32'),
            expected_output_dtype='float16',
            expected_output_shape=(10, 5)
        )

    def test_actual_shape_2d(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3], return_probs=True)
        inputs = tf.random.uniform((10, 5))
        targets = tf.range(10, dtype='int32')

        result = layer([inputs, targets], training=True)
        self.assertListEqual([10, 20], list(result.shape))

    def test_actual_shape_3d(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3, 8], return_probs=True)
        inputs = tf.random.uniform((2, 10, 64))
        targets = tf.reshape(tf.range(20, dtype='int32'), [2, 10])

        result = layer([inputs, targets], training=True)
        self.assertListEqual([2, 10, 20], list(result.shape))

    def test_loss_and_output_2d_over_batch(self):
        layer = AdaptiveSoftmax(units=20, cutoff=[3], return_probs=True, loss_reduction='sum_over_batch_size')
        inputs = tf.random.uniform((10, 5))
        targets = tf.range(10, dtype='int32')

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
        layer = AdaptiveSoftmax(units=20, cutoff=[3], return_probs=True, loss_reduction='sum')
        inputs = tf.random.uniform((10, 5))
        targets = tf.range(10, dtype='int32')

        train_result = layer([inputs, targets], training=True)
        train_sum = np.sum(train_result, axis=-1)
        train_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(train_sum), train_sum)

        eval_result = layer([inputs, targets], training=False)
        eval_sum = np.sum(eval_result, axis=-1)
        eval_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(eval_sum), eval_sum)

        self.assertGreater(eval_loss, train_loss)

    def test_loss_mask_3d(self):
        inputs = tf.ragged.constant([
            [[1., 2.], [2., 3.], [2., 5.]],
            [[0., 9.]],
            [[1., 1.], [2., 9.]]
        ], ragged_rank=1)
        targets = tf.cast(tf.reduce_max(inputs, axis=-1), tf.int32)
        eval_ones = tf.ones_like(targets).to_tensor()

        layer1 = AdaptiveSoftmax(units=10, cutoff=[3], return_probs=True)
        eval1_result = layer1([inputs, targets], training=False)
        eval1_sum = eval1_result.to_tensor()
        eval1_sum = np.sum(eval1_sum, axis=-1)
        eval1_loss = np.sum(layer1.losses)
        self.assertAllClose(eval_ones, eval1_sum)

        inputs_dense = layers.Masking()(inputs.to_tensor())
        self.assertIsNotNone(inputs_dense._keras_mask)
        targets_dense = targets.to_tensor(0)

        layer2 = AdaptiveSoftmax(units=10, cutoff=[3], return_probs=True)
        layer2([inputs_dense, targets_dense], training=False)
        layer2.set_weights(layer1.get_weights())
        eval2_result = layer2([inputs_dense, targets_dense], training=False)
        self.assertIsNotNone(eval2_result._keras_mask)
        eval2_sum = np.where(eval2_result._keras_mask, np.sum(eval2_result, axis=-1), 0.)
        eval2_loss = np.sum(layer2.losses)
        self.assertAllClose(eval_ones, eval2_sum)

        self.assertEqual(eval1_loss, eval2_loss)

    def test_loss_probs_alone(self):
        inputs = tf.constant([
            [[1., 2.], [2., 3.], [2., 5.]],
            [[0., 9.], [.8, .2], [8., .1]],
            [[1., 1.], [2., 9.], [3., .2]]
        ])
        targets = tf.constant([[1, 0, 1], [0, 1, 0], [1, 0, 1]], 'int32')

        layer1 = AdaptiveSoftmax(units=10, cutoff=[3], return_probs=True)
        layer1([inputs, targets], training=True)
        eval1_loss = layer1.losses[0]

        layer2 = AdaptiveSoftmax(units=10, cutoff=[3], return_probs=False)
        layer2([inputs, targets], training=True)
        layer2.set_weights(layer1.get_weights())
        layer2([inputs, targets], training=True)
        eval2_loss = layer2.losses[0]
        layer2([inputs, targets], training=False)
        eval3_loss = layer2.losses[0]

        self.assertEqual(eval1_loss, eval2_loss)
        self.assertEqual(eval2_loss, eval3_loss)

    def test_model(self):
        utils.set_random_seed(87654321)

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

        ids = layers.Input(shape=(None,), dtype='int32')
        targets = layers.Input(shape=(None,), dtype='int32')

        embeddings = layers.Embedding(input_dim=num_samples, output_dim=embed_size)(ids)
        logits = layers.Dense(embed_size // 2, activation='relu')(embeddings)
        probs = AdaptiveSoftmax(units=num_classes, cutoff=[3], return_probs=True)([logits, targets])
        model = models.Model(inputs=[ids, targets], outputs=probs)

        model.compile(optimizer='Adam', loss=None)
        history = model.fit(x=xt, y=None, batch_size=100, epochs=3, validation_data=(xv, None)).history
        predictions = model.predict(x=xp, batch_size=100)
        predictsum = np.sum(predictions, axis=-1)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertGreater(history['val_loss'][0], history['loss'][0])
        self.assertGreater(history['val_loss'][-1], history['loss'][-1])
        self.assertEqual([sample_size // 100, seq_length, num_classes], list(predictions.shape))
        self.assertAllClose(np.ones_like(predictsum), predictsum)

    # TODO: https://github.com/keras-team/keras/issues/18414
    # def test_ragged_input(self):
    #     layer = AdaptiveSoftmax(units=16, cutoff=[1], return_probs=True, factor=2)
    #     logits_data = tf.ragged.constant([
    #         [[1., 1.], [2., 2.], [2., 2.]],
    #         [[0., 0.]],
    #         [[1., 1.], [2., 2.]]
    #     ], ragged_rank=1)
    #     targets_data = tf.ragged.constant([
    #         [1, 2, 3],
    #         [8],
    #         [14, 15]
    #     ], ragged_rank=1)
    #     layer([logits_data, targets_data])
    #     layer.set_weights([
    #         np.array([[1.] * 2] * 2),
    #         np.array([[2.] * 8] * 2),
    #         np.array([[4.] * 15] * 8),
    #     ])
    #
    #     logit_inputs = layers.Input(shape=(None, 2), dtype=tf.float32, ragged=True)
    #     logit_targets = layers.Input(shape=(None,), dtype=tf.int32, ragged=True)
    #     outputs = layer([logit_inputs, logit_targets])
    #
    #     model = models.Model(inputs=[logit_inputs, logit_targets], outputs=outputs)
    #     outputs = model.predict([logits_data, targets_data])
    #     self.assertAllClose(
    #         outputs,
    #         tf.ragged.constant([
    #             [
    #                 [0.5] + [0.03333333134651184] * 15,
    #                 [0.5] + [0.03333333134651184] * 15,
    #                 [0.5] + [0.03333333134651184] * 15
    #             ],
    #             [
    #                 [0.5] + [0.03333333134651184] * 15
    #             ],
    #             [
    #                 [0.5] + [0.03333333134651184] * 15,
    #                 [0.5] + [0.03333333134651184] * 15
    #             ]
    #         ], ragged_rank=1)
    #     )


class SampledSofmaxTest(testing.TestCase):
    def setUp(self):
        super(SampledSofmaxTest, self).setUp()
        self.default_policy = mixed_precision.global_policy()
        self.mf16_policy = mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(SampledSofmaxTest, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        self.run_layer_test(
            SampledSofmax,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': True},
            input_shape=((10, 5), (10,)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(10, 11)
        )
        self.run_layer_test(
            SampledSofmax,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': True},
            input_shape=((2, 10, 5), (2, 10,)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(2, 10, 11)
        )
        self.run_layer_test(
            SampledSofmax,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': True},
            input_shape=((3, 2, 10, 5), (3, 2, 10)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(3, 2, 10, 11)
        )
        self.run_layer_test(
            SampledSofmax,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': False},
            input_shape=((3, 2, 10, 5), (3, 2, 10)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(3, 2, 10, 5)
        )

        mixed_precision.set_global_policy(self.mf16_policy)
        self.run_layer_test(
            SampledSofmax,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': True},
            input_shape=((10, 5), (10,)),
            input_dtype=('float16', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(10, 11)
        )
        self.run_layer_test(
            SampledSofmax,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': False},
            input_shape=((10, 5), (10,)),
            input_dtype=('float16', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(10, 5)
        )

    def test_actual_shape_2d(self):
        layer = SampledSofmax(units=20, negatives=5, return_probs=True)
        inputs = tf.random.uniform((10, 5))
        targets = tf.range(10, dtype='int32')

        result = layer([inputs, targets], training=True)
        self.assertListEqual([10, 20], list(result.shape))

    def test_actual_shape_3d(self):
        layer = SampledSofmax(units=20, negatives=5, return_probs=True)
        inputs = tf.random.uniform((2, 10, 64))
        targets = tf.reshape(tf.range(20, dtype='int32'), [2, 10])

        result = layer([inputs, targets], training=True)
        self.assertListEqual([2, 10, 20], list(result.shape))

    def test_loss_and_output_2d(self):
        layer = SampledSofmax(units=20, negatives=5, return_probs=True)
        inputs = tf.random.uniform((10, 5))
        targets = tf.range(10, dtype='int32')

        train_result = layer([inputs, targets], training=True)
        train_sum = np.sum(train_result, axis=-1)
        train_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(train_sum), train_sum)

        eval_result = layer([inputs, targets], training=False)
        eval_sum = np.sum(eval_result, axis=-1)
        eval_loss = np.sum(layer.losses)
        self.assertAllClose(np.ones_like(eval_sum), eval_sum)

        self.assertGreater(eval_loss, train_loss)

    def test_loss_mask_3d(self):
        inputs = tf.ragged.constant([
            [[1., 2.], [2., 3.], [2., 5.]],
            [[0., 9.]],
            [[1., 1.], [2., 9.]]
        ], ragged_rank=1)
        targets = tf.cast(tf.reduce_max(inputs, axis=-1), tf.int32)
        eval_ones = tf.ones_like(targets).to_tensor()

        layer1 = SampledSofmax(units=10, negatives=5, return_probs=True)
        eval1_result = layer1([inputs, targets], training=False)
        eval1_sum = eval1_result.to_tensor()
        eval1_sum = np.sum(eval1_sum, axis=-1)
        eval1_loss = np.sum(layer1.losses)
        self.assertAllClose(eval_ones, eval1_sum)

        inputs_dense = layers.Masking()(inputs.to_tensor())
        targets_dense = targets.to_tensor(0)

        layer2 = SampledSofmax(units=10, negatives=5, return_probs=True)
        layer2([inputs_dense, targets_dense], training=False)
        layer2.set_weights(layer1.get_weights())
        eval2_result = layer2([inputs_dense, targets_dense], training=False)
        self.assertIsNotNone(eval2_result._keras_mask)
        eval2_sum = np.where(eval2_result._keras_mask, np.sum(eval2_result, axis=-1), 0.)
        eval2_loss = np.sum(layer2.losses)
        self.assertAllClose(eval_ones, eval2_sum)

        self.assertEqual(eval1_loss, eval2_loss)

    def test_model(self):
        utils.set_random_seed(87654321)

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

        ids = layers.Input(shape=(None,), dtype='int32')
        targets = layers.Input(shape=(None,), dtype='int32')

        embeddings = layers.Embedding(input_dim=num_samples, output_dim=embed_size)(ids)
        logits = layers.Dense(embed_size // 2, activation='relu')(embeddings)
        probs = SampledSofmax(units=num_classes, negatives=num_classes // 2, return_probs=True)([logits, targets])
        model = models.Model(inputs=[ids, targets], outputs=probs)

        model.compile(optimizer='Adam', loss=None)
        history = model.fit(x=xt, y=None, batch_size=100, epochs=3, validation_data=(xv, None)).history
        predictions = model.predict(x=xp, batch_size=100)
        predictsum = np.sum(predictions, axis=-1)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        # self.assertGreater(history['val_loss'][0], history['loss'][0])
        # self.assertGreater(history['val_loss'][-1], history['loss'][-1])
        self.assertEqual([sample_size // 100, seq_length, num_classes], list(predictions.shape))
        self.assertAllClose(np.ones_like(predictsum), predictsum)

    # TODO: https://github.com/keras-team/keras/issues/18414
    # def test_with_ragged_input(self):
    #     layer = SampledSofmax(units=16, negatives=8, return_probs=True)
    #     logits_data = tf.ragged.constant([
    #         [[1.], [2.], [2.]],
    #         [[0.]],
    #         [[1.], [2.]]
    #     ], ragged_rank=1)
    #     targets_data = tf.ragged.constant([
    #         [1, 2, 3],
    #         [8],
    #         [14, 15]
    #     ], ragged_rank=1)
    #     layer([logits_data, targets_data])
    #     layer.set_weights([
    #         np.array([[1.]] * 16),
    #         np.array([2.] * 16),
    #     ])
    #
    #     logit_inputs = layers.Input(shape=(None, 1), dtype=tf.float32, ragged=True)
    #     logit_targets = layers.Input(shape=(None,), dtype=tf.int32, ragged=True)
    #     outputs = layer([logit_inputs, logit_targets])
    #
    #     model = models.Model(inputs=[logit_inputs, logit_targets], outputs=outputs)
    #     outputs = model.predict([logits_data, targets_data])
    #     self.assertAllClose(
    #         outputs,
    #         tf.ragged.constant([
    #             [[0.0625] * 16, [0.0625] * 16, [0.0625] * 16],
    #             [[0.0625] * 16],
    #             [[0.0625] * 16, [0.0625] * 16]
    #         ], ragged_rank=1)
    #     )


class NoiseContrastiveEstimationTest(testing.TestCase):
    def setUp(self):
        super(NoiseContrastiveEstimationTest, self).setUp()
        self.default_policy = mixed_precision.global_policy()
        self.mf16_policy = mixed_precision.Policy('mixed_float16')

    def tearDown(self):
        super(NoiseContrastiveEstimationTest, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        self.run_layer_test(
            NoiseContrastiveEstimation,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': True},
            input_shape=((10, 5), (10,)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(10, 11)
        )
        self.run_layer_test(
            NoiseContrastiveEstimation,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': True},
            input_shape=((2, 10, 5), (2, 10,)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(2, 10, 11)
        )
        self.run_layer_test(
            NoiseContrastiveEstimation,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': True},
            input_shape=((3, 2, 10, 5), (3, 2, 10)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(3, 2, 10, 11)
        )
        self.run_layer_test(
            NoiseContrastiveEstimation,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': False},
            input_shape=((3, 2, 10, 5), (3, 2, 10)),
            input_dtype=('float32', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(3, 2, 10, 5)
        )

        mixed_precision.set_global_policy(self.mf16_policy)
        self.run_layer_test(
            NoiseContrastiveEstimation,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': True},
            input_shape=((10, 5), (10,)),
            input_dtype=('float16', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(10, 11)
        )
        self.run_layer_test(
            NoiseContrastiveEstimation,
            init_kwargs={'units': 11, 'negatives': 2, 'return_probs': False},
            input_shape=((10, 5), (10,)),
            input_dtype=('float16', 'int32'),
            expected_output_dtype='float32',
            expected_output_shape=(10, 5)
        )

    def test_actual_shape_2d(self):
        layer = NoiseContrastiveEstimation(units=20, negatives=5, return_probs=True)
        inputs = tf.random.uniform((10, 5))
        targets = tf.range(10, dtype='int32')

        result = layer([inputs, targets], training=True)
        self.assertListEqual([10, 20], list(result.shape))

    def test_actual_shape_3d(self):
        layer = NoiseContrastiveEstimation(units=20, negatives=5, return_probs=True)
        inputs = tf.random.uniform((2, 10, 64))
        targets = tf.reshape(tf.range(20, dtype='int32'), [2, 10])

        result = layer([inputs, targets], training=True)
        self.assertListEqual([2, 10, 20], list(result.shape))

    def test_loss_and_output_2d(self):
        layer = NoiseContrastiveEstimation(units=20, negatives=5, return_probs=True)
        inputs = tf.random.uniform((10, 5))
        targets = tf.range(10, dtype='int32')

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
        utils.set_random_seed(87654321)

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

        ids = layers.Input(shape=(None,), dtype='int32')
        targets = layers.Input(shape=(None,), dtype='int32')

        embeddings = layers.Embedding(input_dim=num_samples, output_dim=embed_size)(ids)
        logits = layers.Dense(embed_size // 2, activation='relu')(embeddings)
        probs = NoiseContrastiveEstimation(
            units=num_classes, negatives=num_classes // 2, return_probs=True)([logits, targets])
        model = models.Model(inputs=[ids, targets], outputs=probs)

        model.compile(optimizer='Adam', loss=None)
        history = model.fit(x=xt, y=None, batch_size=100, epochs=3, validation_data=(xv, None)).history
        predictions = model.predict(x=xp, batch_size=100)
        predictsum = np.sum(predictions, axis=-1)

        self.assertGreater(history['loss'][0], history['loss'][-1])
        self.assertGreater(history['val_loss'][0], history['val_loss'][-1])
        self.assertGreater(history['val_loss'][0], history['loss'][0])
        self.assertGreater(history['val_loss'][-1], history['loss'][-1])
        self.assertEqual([sample_size // 100, seq_length, num_classes], list(predictions.shape))
        self.assertAllClose(np.ones_like(predictsum), predictsum)

    # TODO: https://github.com/keras-team/keras/issues/18414
    # def test_with_ragged_input(self):
    #     layer = NoiseContrastiveEstimation(units=16, negatives=8, return_probs=True)
    #     logits_data = tf.ragged.constant([
    #         [[1.], [2.], [2.]],
    #         [[0.]],
    #         [[1.], [2.]]
    #     ], ragged_rank=1)
    #     targets_data = tf.ragged.constant([
    #         [1, 2, 3],
    #         [8],
    #         [14, 15]
    #     ], ragged_rank=1)
    #     layer([logits_data, targets_data])
    #     layer.set_weights([
    #         np.array([[1.]] * 16),
    #         np.array([2.] * 16),
    #     ])
    #
    #     logit_inputs = layers.Input(shape=(None, 1), dtype=tf.float32, ragged=True)
    #     logit_targets = layers.Input(shape=(None,), dtype=tf.int32, ragged=True)
    #     outputs = layer([logit_inputs, logit_targets])
    #
    #     model = models.Model(inputs=[logit_inputs, logit_targets], outputs=outputs)
    #     outputs = model.predict([logits_data, targets_data])
    #     self.assertAllClose(
    #         outputs,
    #         tf.ragged.constant([
    #             [[0.0625] * 16, [0.0625] * 16, [0.0625] * 16],
    #             [[0.0625] * 16],
    #             [[0.0625] * 16, [0.0625] * 16]
    #         ], ragged_rank=1)
    #     )


if __name__ == "__main__":
    tf.test.main()
