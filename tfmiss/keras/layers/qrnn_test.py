from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras import keras_parameterized, testing_utils
from tensorflow.python.util import object_identity
from tensorflow.python.training.tracking import util as trackable_util
from tfmiss.keras.layers.qrnn import QRNN
from tfmiss.keras.testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class QRNNTest(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': False, 'return_state': False, 'go_backwards': False, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': False, 'go_backwards': False, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 3, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': False, 'return_state': False, 'go_backwards': False, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 3, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': False, 'go_backwards': False, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0.1, 'output_gate': True,
                    'return_sequences': False, 'return_state': False, 'go_backwards': False, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0.1, 'output_gate': True,
                    'return_sequences': True, 'return_state': False, 'go_backwards': False, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': False,
                    'return_sequences': False, 'return_state': False, 'go_backwards': False, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': False,
                    'return_sequences': True, 'return_state': False, 'go_backwards': False, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': False, 'return_state': False, 'go_backwards': True, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 8)
        )
        testing_utils.layer_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': False, 'go_backwards': True, 'time_major': False},
            input_shape=(10, 5, 3),
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=(None, 5, 8)
        )

        # TODO: can't test time_major=True due to input size check in TF 2.4.0
        # testing_utils.layer_test(
        #     QRNN,
        #     kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
        #             'return_sequences': False, 'return_state': False, 'go_backwards': True, 'time_major': True},
        #     input_shape=(10, 5, 3),
        #     input_dtype='float32',
        #     expected_output_dtype='float32',
        #     expected_output_shape=(None, 8)
        # )
        # testing_utils.layer_test(
        #     QRNN,
        #     kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
        #             'return_sequences': True, 'return_state': False, 'go_backwards': True, 'time_major': True},
        #     input_shape=(10, 5, 3),
        #     input_dtype='float32',
        #     expected_output_dtype='float32',
        #     expected_output_shape=(None, 5, 8)
        # )

    def test_layer_state(self):
        layer_multi_io_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': False, 'return_state': True, 'go_backwards': False, 'time_major': False},
            input_shapes=[(10, 5, 3)],
            input_dtypes=['float32'],
            expected_output_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 8), (None, 8)]
        )
        layer_multi_io_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': True, 'go_backwards': False, 'time_major': False},
            input_shapes=[(10, 5, 3)],
            input_dtypes=['float32'],
            expected_output_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 5, 8), (None, 8)]
        )
        layer_multi_io_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': False, 'return_state': True, 'go_backwards': True, 'time_major': False},
            input_shapes=[(10, 5, 3)],
            input_dtypes=['float32'],
            expected_output_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 8), (None, 8)]
        )
        layer_multi_io_test(
            QRNN,
            kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
                    'return_sequences': True, 'return_state': True, 'go_backwards': True, 'time_major': False},
            input_shapes=[(10, 5, 3)],
            input_dtypes=['float32'],
            expected_output_dtypes=['float32', 'float32'],
            expected_output_shapes=[(None, 5, 8), (None, 8)]
        )

        # TODO: can't test time_major=True due to input size check in TF 2.4.0
        # layer_multi_io_test(
        #     QRNN,
        #     kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
        #             'return_sequences': False, 'return_state': True, 'go_backwards': False, 'time_major': True},
        #     input_shapes=[(10, 5, 3)],
        #     input_dtypes=['float32'],
        #     expected_output_dtypes=['float32', 'float32'],
        #     expected_output_shapes=[(None, 8), (None, 8)]
        # )
        # layer_multi_io_test(
        #     QRNN,
        #     kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
        #             'return_sequences': True, 'return_state': True, 'go_backwards': False, 'time_major': True},
        #     input_shapes=[(10, 5, 3)],
        #     input_dtypes=['float32'],
        #     expected_output_dtypes=['float32', 'float32'],
        #     expected_output_shapes=[(None, 5, 8), (None, 8)]
        # )
        # layer_multi_io_test(
        #     QRNN,
        #     kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
        #             'return_sequences': False, 'return_state': True, 'go_backwards': True, 'time_major': True},
        #     input_shapes=[(10, 5, 3)],
        #     input_dtypes=['float32'],
        #     expected_output_dtypes=['float32', 'float32'],
        #     expected_output_shapes=[(None, 8), (None, 8)]
        # )
        # layer_multi_io_test(
        #     QRNN,
        #     kwargs={'units': 8, 'window': 2, 'zoneout': 0., 'output_gate': True,
        #             'return_sequences': True, 'return_state': True, 'go_backwards': True, 'time_major': True},
        #     input_shapes=[(10, 5, 3)],
        #     input_dtypes=['float32'],
        #     expected_output_dtypes=['float32', 'float32'],
        #     expected_output_shapes=[(None, 5, 8), (None, 8)]
        # )

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

        layer = QRNN(8, 2, return_state=True, time_major=True)
        h, c = self.evaluate(layer(data))
        self.assertTupleEqual((3, 8), h.shape)
        self.assertTupleEqual((3, 8), c.shape)

        layer = QRNN(8, 2, return_state=True, return_sequences=True, time_major=True)
        h, c = self.evaluate(layer(data))
        self.assertTupleEqual((10, 3, 8), h.shape)
        self.assertTupleEqual((3, 8), c.shape)

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

        layer = QRNN(8, 2, return_state=True, time_major=True)
        h, c = layer(data)
        h, c = self.evaluate(layer(data, initial_state=c))
        self.assertTupleEqual((3, 8), h.shape)
        self.assertTupleEqual((3, 8), c.shape)

    def test_model(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Bidirectional(QRNN(
            units=12,
            window=2,
            zoneout=0.2,
            return_sequences=True
        )))
        model.add(QRNN(
            units=2,
            window=1
        ))
        model.compile(optimizer='rmsprop', loss='mse', run_eagerly=testing_utils.should_run_eagerly())
        model.fit(np.random.random((10, 3, 4)), np.random.random((10, 2)), epochs=1, batch_size=10)

        # test config
        model.get_config()

        # check whether the model variables are present in the trackable list of objects
        checkpointed_objects = object_identity.ObjectIdentitySet(trackable_util.list_objects(model))
        for v in model.variables:
            self.assertIn(v, checkpointed_objects)


if __name__ == "__main__":
    tf.test.main()
