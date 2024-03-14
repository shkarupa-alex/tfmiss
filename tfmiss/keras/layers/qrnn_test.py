from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tf_keras import layers, mixed_precision, models
from tf_keras.src.testing_infra import test_combinations, test_utils
from tensorflow.python.util import object_identity
from tensorflow.python.checkpoint import checkpoint
from tfmiss.keras.layers.qrnn import QRNN
from tfmiss.keras.testing_utils import layer_multi_io_test


@test_combinations.run_all_keras_modes
class QRNNTest(test_combinations.TestCase):
    def setUp(self):
        super(QRNNTest, self).setUp()
        self.default_policy = mixed_precision.global_policy()

    def tearDown(self):
        super(QRNNTest, self).tearDown()
        mixed_precision.set_global_policy(self.default_policy)

    def test_layer(self):
        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True, 'zero_output_for_mask': True,
                    'return_sequences': False, 'return_state': False, 'go_backwards': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': False, 'go_backwards': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 3, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': False, 'return_state': False, 'go_backwards': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 3, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': False, 'go_backwards': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0.1, 'output_gate': True,
                    'return_sequences': False, 'return_state': False, 'go_backwards': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0.1, 'output_gate': True,
                    'return_sequences': True, 'return_state': False, 'go_backwards': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': False,
                    'return_sequences': False, 'return_state': False, 'go_backwards': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': False,
                    'return_sequences': True, 'return_state': False, 'go_backwards': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': False, 'return_state': False, 'go_backwards': True},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': False, 'go_backwards': True},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

    def test_layer_fp16(self):
        mixed_precision.set_global_policy('mixed_float16')
        test_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True, 'zero_output_for_mask': True,
                    'return_sequences': False, 'return_state': False, 'go_backwards': False},
            input_shape=(10, 5, 3),
            input_dtype='float16',
            expected_output_dtype='float16',
            expected_output_shape=(None, 8)
        )

    def test_layer_state(self):
        layer_multi_io_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': False, 'return_state': True, 'go_backwards': False},
            input_shapes=[(10, 5, 3)],
            input_dtypes=['float32'],
            expected_output_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 8), (None, 8)]
        )
        layer_multi_io_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': True, 'go_backwards': False},
            input_shapes=[(10, 5, 3)],
            input_dtypes=['float32'],
            expected_output_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 5, 8), (None, 8)]
        )
        layer_multi_io_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': False, 'return_state': True, 'go_backwards': True},
            input_shapes=[(10, 5, 3)],
            input_dtypes=['float32'],
            expected_output_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 8), (None, 8)]
        )
        layer_multi_io_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': True, 'go_backwards': True},
            input_shapes=[(10, 5, 3)],
            input_dtypes=['float32'],
            expected_output_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 5, 8), (None, 8)]
        )

    def test_shapes(self):
        data = np.random.random((10, 3, 4))

        layer = QRNN(8, 2, return_state=True)
        h, c = self.evaluate(layer(data))
        self.assertTupleEqual((10, 8), h.shape)
        self.assertTupleEqual((10, 8), c.shape)

        layer = QRNN(8, 2, return_state=True, go_backwards=True)
        h, c = self.evaluate(layer(data))
        self.assertTupleEqual((10, 8), h.shape)
        self.assertTupleEqual((10, 8), c.shape)

        layer = QRNN(8, 2, return_state=True, return_sequences=True)
        h, c = self.evaluate(layer(data))
        self.assertTupleEqual((10, 3, 8), h.shape)
        self.assertTupleEqual((10, 8), c.shape)

    def test_initial_state(self):
        data = np.random.random((10, 3, 4))

        layer = QRNN(8, 2, return_state=True)
        h, c = layer(data)
        h, c = self.evaluate(layer(data, initial_state=c))
        self.assertTupleEqual((10, 8), h.shape)
        self.assertTupleEqual((10, 8), c.shape)

        layer = QRNN(8, 2, return_state=True, return_sequences=True)
        h, c = layer(data)
        h, c = self.evaluate(layer(data, initial_state=c))
        self.assertTupleEqual((10, 3, 8), h.shape)
        self.assertTupleEqual((10, 8), c.shape)

    def test_mask(self):
        data = np.random.random((3, 10, 2))
        data[0, 7:] = 0.
        data[1, 8:] = 0.

        m = layers.Masking()(data)
        h = QRNN(1, 2, output_gate=False, return_sequences=True)(m)
        h = self.evaluate(h)
        self.assertTupleEqual((3, 10, 1), h.shape)
        self.assertEqual(h[0, 6, 0], h[0, 7, 0])
        self.assertEqual(h[0, 6, 0], h[0, 8, 0])
        self.assertEqual(h[0, 6, 0], h[0, 9, 0])
        self.assertEqual(h[1, 7, 0], h[1, 8, 0])
        self.assertEqual(h[1, 7, 0], h[1, 9, 0])

        m = layers.Masking()(data)
        h = QRNN(1, 2, zero_output_for_mask=True, return_sequences=True)(m)
        h = self.evaluate(h)
        self.assertTupleEqual((3, 10, 1), h.shape)
        self.assertEqual(h[0, 7, 0], 0.)
        self.assertEqual(h[0, 8, 0], 0.)
        self.assertEqual(h[0, 9, 0], 0.)
        self.assertEqual(h[1, 8, 0], 0.)
        self.assertEqual(h[1, 9, 0], 0.)

        q_sec = QRNN(1, 2, return_sequences=True)
        q_sec(data)
        q_lst = QRNN(1, 2, return_sequences=False)
        q_lst(data)
        q_lst.set_weights(q_sec.get_weights())
        m = layers.Masking()(data)
        h_sec, h_lst = q_sec(m), q_lst(m)
        h_sec, h_lst = self.evaluate([h_sec, h_lst])
        self.assertTupleEqual((3, 10, 1), h_sec.shape)
        self.assertTupleEqual((3, 1), h_lst.shape)
        self.assertEqual(h_sec[0, 6, 0], h_lst[0, 0])
        self.assertEqual(h_sec[1, 7, 0], h_lst[1, 0])
        self.assertEqual(h_sec[2, 9, 0], h_lst[2, 0])

    def test_zoneout(self):
        data = np.random.random((3, 10, 2))
        init = np.random.random((3, 1))

        h = QRNN(1, 2, zoneout=.9999, output_gate=False, return_sequences=True)(data, initial_state=init, training=True)
        h = self.evaluate(h)
        self.assertTupleEqual((3, 10, 1), h.shape)
        self.assertLess(np.abs(h - np.repeat(init[:, None], 10, axis=1)).max(), 1e-6)

        h = QRNN(1, 2, zoneout=.3, output_gate=False, return_sequences=True)(data, training=True)
        h = self.evaluate(h)
        self.assertBetween(np.mean(h[:, 1:] == h[:, :-1]), 0.1, 0.5)

        h = QRNN(1, 2, zoneout=.7, output_gate=False, return_sequences=True)(data, training=True)
        h = self.evaluate(h)
        self.assertBetween(np.mean(h[:, 1:] == h[:, :-1]), 0.5, 0.9)

    def test_go_backward(self):
        data = np.random.random((3, 10, 2))
        data[0, 7:] = 0.
        data[1, 8:] = 0.

        q_fwd = QRNN(1, 2, go_backwards=False)
        q_fwd(data)
        q_bwd = QRNN(1, 2, go_backwards=True)
        q_bwd(data)
        q_bwd.set_weights(q_fwd.get_weights())
        m_fwd = layers.Masking()(data)
        m_bwd = layers.Masking()(data[:, ::-1, :])
        h_fwd, h_bwd = q_fwd(m_fwd), q_bwd(m_bwd)
        h_fwd, h_bwd = self.evaluate([h_fwd, h_bwd])
        self.assertTupleEqual((3, 1), h_fwd.shape)
        self.assertTupleEqual((3, 1), h_bwd.shape)
        self.assertListEqual(h_fwd.tolist(), h_bwd.tolist())

    def test_model(self):
        model = models.Sequential()
        model.add(layers.Bidirectional(QRNN(
            units=12,
            window=2,
            zoneout=0.2,
            return_sequences=True
        )))
        model.add(QRNN(
            units=2,
            window=1
        ))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=test_utils.should_run_eagerly())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 2)), epochs=1, batch_size=10)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(checkpoint.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


if __name__ == "__main__":
    tf.test.main()
