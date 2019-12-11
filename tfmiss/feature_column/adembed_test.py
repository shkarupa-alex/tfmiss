from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import numpy as np
import tensorflow as tf
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.client import session
from tensorflow.python.feature_column import dense_features as df
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tfmiss.feature_column.adembed import adaptive_embedding_column, AdaptiveEmbeddingColumn


def _initialized_session(config=None):
    sess = session.Session(config=config)
    sess.run(variables_lib.global_variables_initializer())
    sess.run(lookup_ops.tables_initializer())
    return sess


def _assert_sparse_tensor_value(test_case, expected, actual):
    test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
    test_case.assertAllEqual(expected.indices, actual.indices)

    test_case.assertEqual(np.array(expected.values).dtype, np.array(actual.values).dtype)
    test_case.assertAllEqual(expected.values, actual.values)

    test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
    test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


class _TestStateManager(fc.StateManager):

    def __init__(self, trainable=True):
        # Dict of feature_column to a dict of variables.
        self._all_variables = {}
        self._cols_to_resources_map = collections.defaultdict(lambda: {})
        self._trainable = trainable

    def create_variable(self, feature_column, name, shape,
                        dtype=None, trainable=True, use_resource=True, initializer=None):
        if feature_column not in self._all_variables:
            self._all_variables[feature_column] = {}

        var_dict = self._all_variables[feature_column]
        if name in var_dict:
            return var_dict[name]
        else:
            var = variable_scope.get_variable(
                name=name,
                shape=shape,
                dtype=dtype,
                trainable=self._trainable and trainable,
                use_resource=use_resource,
                initializer=initializer
            )
            var_dict[name] = var

            return var

    def get_variable(self, feature_column, name):
        if feature_column not in self._all_variables:
            raise ValueError('Do not recognize FeatureColumn.')

        if name in self._all_variables[feature_column]:
            return self._all_variables[feature_column][name]

        raise ValueError('Could not find variable.')

    def add_resource(self, feature_column, name, resource):
        self._cols_to_resources_map[feature_column][name] = resource

    def get_resource(self, feature_column, name):
        if name in self._cols_to_resources_map[feature_column]:
            return self._cols_to_resources_map[feature_column][name]
        raise ValueError('Resource does not exist.')


@test_util.run_all_in_graph_and_eager_modes
class EmbeddingColumnTest(tf.test.TestCase):
    def test_defaults(self):
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_dimension = 2
        embedding_column = adaptive_embedding_column(categorical_column, dimension=embedding_dimension, cutoff=[1])
        self.assertIs(categorical_column, embedding_column.categorical_column)
        self.assertEqual((1,), embedding_column.cutoff)
        self.assertEqual(4, embedding_column.factor)
        self.assertEqual(True, embedding_column.mod8)
        self.assertEqual(embedding_dimension, embedding_column.dimension)
        self.assertEqual('mean', embedding_column.combiner)
        # self.assertIsNone(embedding_column.ckpt_to_load_from)
        # self.assertIsNone(embedding_column.tensor_name_in_ckpt)
        self.assertIsNone(embedding_column.max_norm)
        self.assertTrue(embedding_column.trainable)
        self.assertEqual('aaa_adaptive_embedding', embedding_column.name)
        self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
        self.assertEqual({'aaa': parsing_ops.VarLenFeature(tf.string)}, embedding_column.parse_example_spec)
        self.assertTrue(embedding_column._is_v2_column)

    def test_is_v_2_column(self):
        categorical_column = fc_old._categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_dimension = 2
        embedding_column = adaptive_embedding_column(categorical_column, dimension=embedding_dimension, cutoff=[1])
        self.assertFalse(embedding_column._is_v2_column)

    def test_all_constructor_args(self):
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_dimension = 2
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            factor=5,
            combiner='my_combiner',
            initializer=lambda: 'my_initializer',
            # ckpt_to_load_from='my_ckpt',
            # tensor_name_in_ckpt='my_ckpt_tensor',
            max_norm=42.,
            trainable=False
        )
        self.assertIs(categorical_column, embedding_column.categorical_column)
        self.assertEqual((1,), embedding_column.cutoff)
        self.assertEqual(embedding_dimension, embedding_column.dimension)
        self.assertEqual(5, embedding_column.factor)
        self.assertEqual('my_combiner', embedding_column.combiner)
        # self.assertEqual('my_ckpt', embedding_column.ckpt_to_load_from)
        # self.assertEqual('my_ckpt_tensor', embedding_column.tensor_name_in_ckpt)
        self.assertEqual(42., embedding_column.max_norm)
        self.assertFalse(embedding_column.trainable)
        self.assertEqual('aaa_adaptive_embedding', embedding_column.name)
        self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
        self.assertEqual({'aaa': parsing_ops.VarLenFeature(tf.string)}, embedding_column.parse_example_spec)

    def test_deep_copy(self):
        if not tf.executing_eagerly():
            self.skipTest('Deep copy does not work in graph mode')
            return

        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_dimension = 2
        original = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            factor=5,
            mod8=False,
            combiner='my_combiner',
            initializer=lambda: 'my_initializer',
            # ckpt_to_load_from='my_ckpt',
            # tensor_name_in_ckpt='my_ckpt_tensor',
            max_norm=42.,
            trainable=False
        )
        for embedding_column in (original, copy.deepcopy(original)):
            self.assertEqual('aaa', embedding_column.categorical_column.name)
            self.assertEqual(3, embedding_column.categorical_column.num_buckets)
            self.assertEqual({'aaa': parsing_ops.VarLenFeature(tf.string)},
                             embedding_column.categorical_column.parse_example_spec)
            self.assertEqual((1,), embedding_column.cutoff)
            self.assertEqual(embedding_dimension, embedding_column.dimension)
            self.assertEqual(5, embedding_column.factor)
            self.assertEqual(False, embedding_column.mod8)
            self.assertEqual('my_combiner', embedding_column.combiner)
            # self.assertEqual('my_ckpt', embedding_column.ckpt_to_load_from)
            # self.assertEqual('my_ckpt_tensor', embedding_column.tensor_name_in_ckpt)
            self.assertEqual(42., embedding_column.max_norm)
            self.assertFalse(embedding_column.trainable)
            self.assertEqual('aaa_adaptive_embedding', embedding_column.name)
            self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
            self.assertEqual({'aaa': parsing_ops.VarLenFeature(tf.string)}, embedding_column.parse_example_spec)

    def test_invalid_initializer(self):
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        with self.assertRaisesRegexp(ValueError, 'initializer must be callable'):
            adaptive_embedding_column(categorical_column, cutoff=[1], dimension=2, mod8=False, initializer='not_fn')

    def test_parse_example(self):
        a = fc.categorical_column_with_vocabulary_list(key='aaa', default_value=0,
                                                       vocabulary_list=('omar', 'stringer', 'marlo'))
        a_embedded = adaptive_embedding_column(a, cutoff=[1], dimension=2, mod8=False)
        data = example_pb2.Example(features=feature_pb2.Features(feature={
            'aaa': feature_pb2.Feature(bytes_list=feature_pb2.BytesList(value=[b'omar', b'stringer']))}))
        features = parsing_ops.parse_example(
            serialized=[data.SerializeToString()], features=fc.make_parse_example_spec_v2([a_embedded]))
        self.assertIn('aaa', features)

        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=[[0, 0], [0, 1]],
                values=np.array([b'omar', b'stringer'], dtype=np.object_),
                dense_shape=[1, 2]),
            self.evaluate(features['aaa']))

    def test_transform_feature(self):
        # from tensorflow_core.python.feature_column import feature_column_v2 as fc
        a = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        a_embedded = adaptive_embedding_column(a, cutoff=[1], dimension=2, mod8=False)
        features = {'aaa': sparse_tensor.SparseTensor(
            indices=((0, 0), (1, 0), (1, 1)),
            values=('omar', 'stringer', 'omar'),
            dense_shape=(2, 2)
        )}
        outputs = fc._transform_features_v2(features, [a, a_embedded], None)
        output_a = outputs[a]
        output_embedded = outputs[a_embedded]

        self.evaluate(variables_lib.global_variables_initializer())
        self.evaluate(lookup_ops.tables_initializer())

        _assert_sparse_tensor_value(self, self.evaluate(output_a), self.evaluate(output_embedded))

    def test_get_dense_tensor(self):
        # Inputs.
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 4), (3, 0)),
            values=('marlo', 'omar', 'stringer', 'stringer'),
            dense_shape=(4, 5)
        )

        # Embedding variable.
        embedding_dimension = 2
        embedding_values = (
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.),  # id 2
        )

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)

            if 1 == shape[0]:
                return embedding_values[:1]

            return [ew[:1] for ew in embedding_values[1:]]

        # Expected lookup result, using combiner='mean'.
        expected_lookups = (
            # example 0, ids [2], embedding = [7, _] * [1, 1]
            (7., 7.),
            # example 1, ids [0, 1], embedding = mean([1, 2] * [1, 1] + [3, _] * [1, 1]) = [3, 3]
            (2., 2.5),
            # example 2, ids [], embedding = [0, 0]
            (0., 0.),
            # example 3, ids [1], embedding = [3, _] * [1, 1]
            (3., 3.),
        )

        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            initializer=_initializer,
            projection_initializer=tf.compat.v1.ones_initializer(),
        )
        state_manager = _TestStateManager()
        embedding_column.create_state(state_manager)

        # Provide sparse input and get dense result.
        embedding_lookup = embedding_column.get_dense_tensor(
            fc.FeatureTransformationCache({
                'aaa': sparse_input
            }),
            state_manager
        )

        if not tf.executing_eagerly():
            # Assert expected embedding variable and lookups.
            global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.assertItemsEqual(('embedding_weights_0:0',
                                   'embedding_weights_1:0', 'embedding_projections_1:0'),
                                  tuple([v.name for v in global_vars]))

        self.evaluate(variables_lib.global_variables_initializer())
        self.evaluate(lookup_ops.tables_initializer())
        self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

    def test_get_dense_tensor_old_categorical(self):
        # Inputs.
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 4), (3, 0)),
            values=('marlo', 'omar', 'stringer', 'stringer'),
            dense_shape=(4, 5)
        )

        # Embedding variable.
        embedding_dimension = 2
        embedding_values = (
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.),  # id 2
        )

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)

            if 1 == shape[0]:
                return embedding_values[:1]

            return [ew[:1] for ew in embedding_values[1:]]

        # Expected lookup result, using combiner='mean'.
        expected_lookups = (
            # example 0, ids [2], embedding = [7, _] * [1, 1]
            (7., 7.),
            # example 1, ids [0, 1], embedding = mean([1, 2] * [1, 1] + [3, _] * [1, 1]) = [3, 3]
            (3., 3.),
            # example 2, ids [], embedding = [0, 0]
            (0., 0.),
            # example 3, ids [1], embedding = [3, _] * [1, 1]
            (3., 3.),
        )

        # Build columns.
        categorical_column = fc_old._categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            proj0=True,
            initializer=_initializer,
            projection_initializer=tf.compat.v1.ones_initializer(),
        )

        # Provide sparse input and get dense result.
        embedding_lookup = embedding_column._get_dense_tensor(
            fc_old._LazyBuilder({
                'aaa': sparse_input
            }))

        if not tf.executing_eagerly():
            # Assert expected embedding variable and lookups.
            global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.assertItemsEqual(('embedding_weights_0:0', 'embedding_projections_0:0',
                                   'embedding_weights_1:0', 'embedding_projections_1:0'),
                                  tuple([v.name for v in global_vars]))

        self.evaluate(variables_lib.global_variables_initializer())
        self.evaluate(lookup_ops.tables_initializer())
        self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

    def test_get_dense_tensor_3d(self):
        # Inputs.
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0, 0), (1, 1, 0), (1, 1, 4), (3, 0, 0), (3, 1, 2)),
            values=('marlo', 'omar', 'stringer', 'stringer', 'marlo'),
            dense_shape=(4, 2, 5)
        )

        # Embedding variable.
        embedding_dimension = 3
        embedding_values = (
            (1., 2., 4.),  # id 0
            (3., 5., 1.),  # id 1
            (7., 11., 2.),  # id 2
            (2., 7., 12.)  # id 3
        )

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)
            if 1 == shape[0]:
                return embedding_values[:1]

            return [ew[:1] for ew in embedding_values[1:]]

        # Expected lookup result, using combiner='mean'.
        expected_lookups = (
            # example 0, ids [[2], []], embedding = [[7, 11, 2], [0, 0, 0]]
            ((7., 7., 7.), (0., 0., 0.)),
            # example 1, ids [[], [0, 1]], embedding
            # = mean([[], [1, 2, 4] + [3, 5, 1]]) = [[0, 0, 0], [2, 3.5, 2.5]]
            ((0., 0., 0.), (2., 2.5, 3.5)),
            # example 2, ids [[], []], embedding = [[0, 0, 0], [0, 0, 0]]
            ((0., 0., 0.), (0., 0., 0.)),
            # example 3, ids [[1], [2]], embedding = [[3, 5, 1], [7, 11, 2]]
            ((3., 3., 3.), (7., 7., 7.)),
        )

        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            initializer=_initializer,
            projection_initializer=tf.compat.v1.ones_initializer(),
        )
        state_manager = _TestStateManager()
        embedding_column.create_state(state_manager)

        # Provide sparse input and get dense result.
        embedding_lookup = embedding_column.get_dense_tensor(
            fc.FeatureTransformationCache({
                'aaa': sparse_input
            }),
            state_manager
        )

        if not tf.executing_eagerly():
            # Assert expected embedding variable and lookups.
            global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.assertItemsEqual(('embedding_weights_0:0',
                                   'embedding_weights_1:0', 'embedding_projections_1:0'),
                                  tuple([v.name for v in global_vars]))

        self.evaluate(variables_lib.global_variables_initializer())
        self.evaluate(lookup_ops.tables_initializer())
        self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

    def test_get_dense_tensor_placeholder_inputs(self):
        if tf.executing_eagerly():
            self.skipTest('Placeholders not compatible with eager execution')

        # Inputs.
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 4), (3, 0)),
            values=('marlo', 'omar', 'stringer', 'stringer'),
            dense_shape=(4, 5)
        )

        # Embedding variable.
        embedding_dimension = 2
        embedding_values = (
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)
            if 1 == shape[0]:
                return embedding_values[:1]

            return [ew[:1] for ew in embedding_values[1:]]

        # Expected lookup result, using combiner='mean'.
        expected_lookups = (
            # example 0, ids [2], embedding = [7, 11]
            (7., 7.),
            # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
            (2., 2.5),
            # example 2, ids [], embedding = [0, 0]
            (0., 0.),
            # example 3, ids [1], embedding = [3, 5]
            (3., 3.),
        )

        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            initializer=_initializer,
            projection_initializer=tf.compat.v1.ones_initializer(),
        )
        state_manager = _TestStateManager()
        embedding_column.create_state(state_manager)

        # Provide sparse input and get dense result.
        input_indices = array_ops.placeholder(dtype=tf.int64)
        input_values = array_ops.placeholder(dtype=tf.string)
        input_shape = array_ops.placeholder(dtype=tf.int64)
        embedding_lookup = embedding_column.get_dense_tensor(
            fc.FeatureTransformationCache({
                'aaa':
                    sparse_tensor.SparseTensorValue(
                        indices=input_indices,
                        values=input_values,
                        dense_shape=input_shape)
            }),
            state_manager
        )

        if not tf.executing_eagerly():
            # Assert expected embedding variable and lookups.
            global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.assertItemsEqual(('embedding_weights_0:0',
                                   'embedding_weights_1:0', 'embedding_projections_1:0'),
                                  tuple([v.name for v in global_vars]))

        self.evaluate(variables_lib.global_variables_initializer())
        self.evaluate(lookup_ops.tables_initializer())

        with _initialized_session():
            self.assertAllEqual(
                expected_lookups,
                embedding_lookup.eval(
                    feed_dict={
                        input_indices: sparse_input.indices,
                        input_values: sparse_input.values,
                        input_shape: sparse_input.dense_shape,
                    }))

    # def test_get_dense_tensor_restore_from_ckpt(self):
    #     # Inputs.
    #     vocabulary_size = 3
    #     sparse_input = sparse_tensor.SparseTensorValue(
    #         # example 0, ids [2]
    #         # example 1, ids [0, 1]
    #         # example 2, ids []
    #         # example 3, ids [1]
    #         indices=((0, 0), (1, 0), (1, 4), (3, 0)),
    #         values=('marlo', 'omar', 'stringer', 'stringer'),
    #         dense_shape=(4, 5)
    #     )
    #
    #     # Embedding variable. The checkpoint file contains _embedding_values.
    #     embedding_dimension = 2
    #     embedding_values = (
    #         (1., 2.),  # id 0
    #         (3., 5.),  # id 1
    #         (7., 11.)  # id 2
    #     )
    #     ckpt_path = os.path.join(os.path.dirname(__file__), 'testdata', 'adaptive_embedding.ckpt-1')
    #     ckpt_tensor = '/'
    #
    #     # Expected lookup result, using combiner='mean'.
    #     expected_lookups = (
    #         # example 0, ids [2], embedding = [7, 11]
    #         (7., 7.),
    #         # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
    #         (3., 3.),
    #         # example 2, ids [], embedding = [0, 0]
    #         (0., 0.),
    #         # example 3, ids [1], embedding = [3, 5]
    #         (3., 3.),
    #     )
    #
    #     # Build columns.
    #     categorical_column = fc.categorical_column_with_vocabulary_list(
    #         key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
    #     embedding_column = adaptive_embedding_column(
    #         categorical_column,
    #         cutoff=[1],
    #         dimension=embedding_dimension,
    #         ckpt_to_load_from=ckpt_path,
    #         tensor_name_in_ckpt=ckpt_tensor
    #     )
    #     state_manager = _TestStateManager()
    #     embedding_column.create_state(state_manager)
    #
    #     # Provide sparse input and get dense result.
    #     embedding_lookup = embedding_column.get_dense_tensor(
    #         fc.FeatureTransformationCache({
    #             'aaa': sparse_input
    #         }),
    #         state_manager
    #     )
    #
    #     if not tf.executing_eagerly():
    #         # Assert expected embedding variable and lookups.
    #         global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    #         self.assertItemsEqual(('embedding_weights_0:0', 'embedding_projections_0:0',
    #                                'embedding_weights_1:0', 'embedding_projections_1:0'),
    #                               tuple([v.name for v in global_vars]))
    #
    #     self.evaluate(variables_lib.global_variables_initializer())
    #     self.evaluate(lookup_ops.tables_initializer())
    #
    #     self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

    def test_linear_model(self):
        # Inputs.
        batch_size = 4
        vocabulary_size = 3
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 4), (3, 0)),
            values=('marlo', 'omar', 'stringer', 'stringer'),
            dense_shape=(batch_size, 5)
        )

        # Embedding variable.
        embedding_dimension = 2

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)

            return tf.compat.v1.initializers.zeros()(shape, dtype, partition_info)

        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            initializer=_initializer,
            projection_initializer=_initializer,
        )

        # with ops.Graph().as_default():
        model = fc.LinearModel((embedding_column,))
        predictions = model({categorical_column.name: sparse_input})

        if not tf.executing_eagerly():
            expected_var_names = (
                'linear_model/bias_weights:0',
                'linear_model/aaa_adaptive_embedding/weights:0',
                'linear_model/aaa_adaptive_embedding/embedding_weights_0:0',
                'linear_model/aaa_adaptive_embedding/embedding_weights_1:0',
                'linear_model/aaa_adaptive_embedding/embedding_projections_1:0',
            )
            self.assertItemsEqual(
                expected_var_names,
                [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)]
            )

            trainable_vars = {v.name: v for v in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)}
            self.assertItemsEqual(expected_var_names, trainable_vars.keys())

            bias = trainable_vars['linear_model/bias_weights:0']
            embedding_weights0 = trainable_vars['linear_model/aaa_adaptive_embedding/embedding_weights_0:0']
            embedding_weights1 = trainable_vars['linear_model/aaa_adaptive_embedding/embedding_weights_1:0']
            embedding_projections_1 = trainable_vars['linear_model/aaa_adaptive_embedding/embedding_projections_1:0']
            linear_weights = trainable_vars['linear_model/aaa_adaptive_embedding/weights:0']

            self.evaluate(variables_lib.global_variables_initializer())
            self.evaluate(lookup_ops.tables_initializer())

            # Predictions with all zero weights.
            self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
            self.assertAllClose(np.zeros((1, embedding_dimension)), self.evaluate(embedding_weights0))
            self.assertAllClose(np.zeros((vocabulary_size - 1, 1)), self.evaluate(embedding_weights1))
            self.assertAllClose(np.zeros((1, embedding_dimension)), self.evaluate(embedding_projections_1))
            self.assertAllClose(np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
        else:
            self.evaluate(variables_lib.global_variables_initializer())
            self.evaluate(lookup_ops.tables_initializer())

        self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))

    def test_dense_features(self):
        # Inputs.
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 4), (3, 0)),
            values=('marlo', 'omar', 'stringer', 'stringer'),
            dense_shape=(4, 5)
        )

        # Embedding variable.
        embedding_dimension = 2
        embedding_values = (
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)
            if 1 == shape[0]:
                return embedding_values[:1]

            return [ew[:1] for ew in embedding_values[1:]]

        # Expected lookup result, using combiner='mean'.
        expected_lookups = (
            # example 0, ids [2], embedding = [7, 11]
            (7., 7.),
            # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
            (2., 2.5),
            # example 2, ids [], embedding = [0, 0]
            (0., 0.),
            # example 3, ids [1], embedding = [3, 5]
            (3., 3.),
        )

        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            initializer=_initializer,
            projection_initializer=tf.compat.v1.initializers.ones()
        )

        # Provide sparse input and get dense result.
        layer = df.DenseFeatures((embedding_column,))
        dense_features = layer({'aaa': sparse_input})

        if not tf.executing_eagerly():
            # Assert expected embedding variable and lookups.
            global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.assertItemsEqual(('dense_features/aaa_adaptive_embedding/embedding_weights_0:0',
                                   'dense_features/aaa_adaptive_embedding/embedding_weights_1:0',
                                   'dense_features/aaa_adaptive_embedding/embedding_projections_1:0'),
                                  tuple([v.name for v in global_vars]))
            for v in global_vars:
                self.assertIsInstance(v, variables_lib.Variable)
            trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
            self.assertItemsEqual(('dense_features/aaa_adaptive_embedding/embedding_weights_0:0',
                                   'dense_features/aaa_adaptive_embedding/embedding_weights_1:0',
                                   'dense_features/aaa_adaptive_embedding/embedding_projections_1:0'),
                                  tuple([v.name for v in trainable_vars]))

        self.evaluate(variables_lib.global_variables_initializer())
        self.evaluate(lookup_ops.tables_initializer())

        self.assertAllEqual(expected_lookups, self.evaluate(dense_features))

    def test_dense_features_not_trainable(self):
        # Inputs.
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 4), (3, 0)),
            values=('marlo', 'omar', 'stringer', 'stringer'),
            dense_shape=(4, 5)
        )

        # Embedding variable.
        embedding_dimension = 2
        embedding_values = (
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)
            if 1 == shape[0]:
                return embedding_values[:1]

            return [ew[:1] for ew in embedding_values[1:]]

        # Expected lookup result, using combiner='mean'.
        expected_lookups = (
            # example 0, ids [2], embedding = [7, 11]
            (7., 7.),
            # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
            (2., 2.5),
            # example 2, ids [], embedding = [0, 0]
            (0., 0.),
            # example 3, ids [1], embedding = [3, 5]
            (3., 3.),
        )

        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            initializer=_initializer,
            projection_initializer=tf.compat.v1.initializers.ones(),
            trainable=False
        )

        # Provide sparse input and get dense result.
        dense_features = df.DenseFeatures((embedding_column,))({'aaa': sparse_input})

        if not tf.executing_eagerly():
            # Assert expected embedding variable and lookups.
            global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.assertItemsEqual(('dense_features/aaa_adaptive_embedding/embedding_weights_0:0',
                                   'dense_features/aaa_adaptive_embedding/embedding_weights_1:0',
                                   'dense_features/aaa_adaptive_embedding/embedding_projections_1:0'),
                                  tuple([v.name for v in global_vars]))
            self.assertItemsEqual([], ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES))

        self.evaluate(variables_lib.global_variables_initializer())
        self.evaluate(lookup_ops.tables_initializer())
        self.assertAllEqual(expected_lookups, self.evaluate(dense_features))

    def test_input_layer(self):
        # Inputs.
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 4), (3, 0)),
            values=('marlo', 'omar', 'stringer', 'stringer'),
            dense_shape=(4, 5))

        # Embedding variable.
        embedding_dimension = 2
        embedding_values = (
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)
            if 1 == shape[0]:
                return embedding_values[:1]

            return [ew[:1] for ew in embedding_values[1:]]

        # Expected lookup result, using combiner='mean'.
        expected_lookups = (
            # example 0, ids [2], embedding = [7, 11]
            (7., 7.),
            # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
            (2., 2.5),
            # example 2, ids [], embedding = [0, 0]
            (0., 0.),
            # example 3, ids [1], embedding = [3, 5]
            (3., 3.),
        )

        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            initializer=_initializer,
            projection_initializer=tf.compat.v1.initializers.ones()
        )

        # Provide sparse input and get dense result.
        feature_layer = fc_old.input_layer({'aaa': sparse_input}, (embedding_column,))

        if not tf.executing_eagerly():
            # Assert expected embedding variable and lookups.
            global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
            self.assertItemsEqual(('input_layer/aaa_adaptive_embedding/embedding_weights_0:0',
                                   'input_layer/aaa_adaptive_embedding/embedding_weights_1:0',
                                   'input_layer/aaa_adaptive_embedding/embedding_projections_1:0'),
                                  tuple([v.name for v in global_vars]))
            trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
            self.assertItemsEqual(('input_layer/aaa_adaptive_embedding/embedding_weights_0:0',
                                   'input_layer/aaa_adaptive_embedding/embedding_weights_1:0',
                                   'input_layer/aaa_adaptive_embedding/embedding_projections_1:0'),
                                  tuple([v.name for v in trainable_vars]))

        self.evaluate(variables_lib.global_variables_initializer())
        self.evaluate(lookup_ops.tables_initializer())
        self.assertAllEqual(expected_lookups, self.evaluate(feature_layer))

    def test_old_linear_model(self):
        # Inputs.
        batch_size = 4
        vocabulary_size = 3
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 4), (3, 0)),
            values=('marlo', 'omar', 'stringer', 'stringer'),
            dense_shape=(batch_size, 5)
        )

        # Embedding variable.
        embedding_dimension = 2

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)

            return tf.compat.v1.initializers.zeros()(shape, dtype, partition_info)

        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            initializer=_initializer,
            projection_initializer=_initializer,
        )

        # with ops.Graph().as_default():
        predictions = fc_old.linear_model({categorical_column.name: sparse_input}, (embedding_column,))

        if not tf.executing_eagerly():
            expected_var_names = (
                'linear_model/bias_weights:0',
                'linear_model/aaa_adaptive_embedding/weights:0',
                'linear_model/aaa_adaptive_embedding/embedding_weights_0:0',
                'linear_model/aaa_adaptive_embedding/embedding_weights_1:0',
                'linear_model/aaa_adaptive_embedding/embedding_projections_1:0',
            )
            self.assertItemsEqual(
                expected_var_names,
                [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)]
            )
            trainable_vars = {v.name: v for v in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)}
            self.assertItemsEqual(expected_var_names, trainable_vars.keys())

            bias = trainable_vars['linear_model/bias_weights:0']
            embedding_weights0 = trainable_vars['linear_model/aaa_adaptive_embedding/embedding_weights_0:0']
            embedding_weights1 = trainable_vars['linear_model/aaa_adaptive_embedding/embedding_weights_1:0']
            embedding_projections_1 = trainable_vars['linear_model/aaa_adaptive_embedding/embedding_projections_1:0']
            linear_weights = trainable_vars['linear_model/aaa_adaptive_embedding/weights:0']

            self.evaluate(variables_lib.global_variables_initializer())
            self.evaluate(lookup_ops.tables_initializer())

            # Predictions with all zero weights.
            self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
            self.assertAllClose(np.zeros((1, embedding_dimension)), self.evaluate(embedding_weights0))
            self.assertAllClose(np.zeros((vocabulary_size - 1, 1)), self.evaluate(embedding_weights1))
            self.assertAllClose(np.zeros((1, embedding_dimension)), self.evaluate(embedding_projections_1))
            self.assertAllClose(np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
        else:
            self.evaluate(variables_lib.global_variables_initializer())
            self.evaluate(lookup_ops.tables_initializer())

        self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))

    def test_old_linear_model_old_categorical(self):
        # Inputs.
        batch_size = 4
        vocabulary_size = 3
        sparse_input = sparse_tensor.SparseTensorValue(
            # example 0, ids [2]
            # example 1, ids [0, 1]
            # example 2, ids []
            # example 3, ids [1]
            indices=((0, 0), (1, 0), (1, 4), (3, 0)),
            values=('marlo', 'omar', 'stringer', 'stringer'),
            dense_shape=(batch_size, 5)
        )

        # Embedding variable.
        embedding_dimension = 2

        def _initializer(shape, dtype, partition_info=None):
            self.assertEqual(tf.float32, dtype)
            self.assertIsNone(partition_info)

            return tf.compat.v1.initializers.zeros()(shape, dtype, partition_info)

        # Build columns.
        categorical_column = fc_old._categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column,
            cutoff=[1],
            dimension=embedding_dimension,
            mod8=False,
            initializer=_initializer,
            projection_initializer=_initializer,
        )

        # with ops.Graph().as_default():
        predictions = fc_old.linear_model({categorical_column.name: sparse_input}, (embedding_column,))

        if not tf.executing_eagerly():
            expected_var_names = (
                'linear_model/bias_weights:0',
                'linear_model/aaa_adaptive_embedding/weights:0',
                'linear_model/aaa_adaptive_embedding/embedding_weights_0:0',
                'linear_model/aaa_adaptive_embedding/embedding_weights_1:0',
                'linear_model/aaa_adaptive_embedding/embedding_projections_1:0',
            )
            self.assertItemsEqual(
                expected_var_names,
                [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])
            trainable_vars = {v.name: v for v in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)}
            self.assertItemsEqual(expected_var_names, trainable_vars.keys())

            bias = trainable_vars['linear_model/bias_weights:0']
            embedding_weights0 = trainable_vars['linear_model/aaa_adaptive_embedding/embedding_weights_0:0']
            embedding_weights1 = trainable_vars['linear_model/aaa_adaptive_embedding/embedding_weights_1:0']
            embedding_projections_1 = trainable_vars['linear_model/aaa_adaptive_embedding/embedding_projections_1:0']
            linear_weights = trainable_vars['linear_model/aaa_adaptive_embedding/weights:0']

            self.evaluate(variables_lib.global_variables_initializer())
            self.evaluate(lookup_ops.tables_initializer())

            # Predictions with all zero weights.
            self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
            self.assertAllClose(np.zeros((1, embedding_dimension)), self.evaluate(embedding_weights0))
            self.assertAllClose(np.zeros((vocabulary_size - 1, 1)), self.evaluate(embedding_weights1))
            self.assertAllClose(np.zeros((1, embedding_dimension)), self.evaluate(embedding_projections_1))
            self.assertAllClose(np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
            self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))
        else:
            self.evaluate(variables_lib.global_variables_initializer())
            self.evaluate(lookup_ops.tables_initializer())

        self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))

    def test_serialization_with_default_initializer(self):
        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column, cutoff=[1], dimension=2, mod8=False, proj0=True)

        self.assertEqual([categorical_column], embedding_column.parents)

        config = embedding_column._get_config()
        config['initializer']['config']['stddev'] = 0.7071067811865475
        # config['projection_initializer']['config']['stddev'] = 0.7071067811865475
        self.assertEqual({
            'categorical_column': {
                'class_name': 'VocabularyListCategoricalColumn',
                'config': {
                    'key': 'aaa',
                    'vocabulary_list': ('omar', 'stringer', 'marlo'),
                    'dtype': 'string',
                    'default_value': 0,
                    'num_oov_buckets': 0
                }
            },
            # 'ckpt_to_load_from': None,
            'combiner': 'mean',
            'cutoff': (1,),
            'dimension': 2,
            'factor': 4,
            'initializer': {
                'class_name': 'TruncatedNormal',
                'config': {
                    'dtype': 'float32',
                    'stddev': 0.7071067811865475,
                    'seed': None,
                    'mean': 0.0
                }
            },
            'projection_initializer': {
                'class_name': 'GlorotUniform',
                'config': {
                    'dtype': 'float32',
                    'seed': None,
                }
            },
            'max_norm': None,
            'mod8': False,
            'proj0': True,
            # 'tensor_name_in_ckpt': None,
            'trainable': True
        }, config)

        custom_objects = {'TruncatedNormal': init_ops.TruncatedNormal, 'GlorotUniform': init_ops.GlorotUniform}
        new_embedding_column = AdaptiveEmbeddingColumn._from_config(config, custom_objects=custom_objects)
        self.assertEqual(config, new_embedding_column._get_config())
        self.assertIsNot(categorical_column, new_embedding_column.categorical_column)

        new_embedding_column = AdaptiveEmbeddingColumn._from_config(
            config, custom_objects=custom_objects,
            columns_by_name={categorical_column.name: categorical_column})
        self.assertEqual(config, new_embedding_column._get_config())
        self.assertIs(categorical_column, new_embedding_column.categorical_column)

    def test_serialization_with_custom_initializer(self):
        def _initializer(shape, dtype, partition_info=None):
            del shape, dtype, partition_info
            return ValueError('Not expected to be called')

        # Build columns.
        categorical_column = fc.categorical_column_with_vocabulary_list(
            key='aaa', default_value=0, vocabulary_list=('omar', 'stringer', 'marlo'))
        embedding_column = adaptive_embedding_column(
            categorical_column, cutoff=[1], dimension=2, initializer=_initializer,
            projection_initializer=_initializer)

        self.assertEqual([categorical_column], embedding_column.parents)

        config = embedding_column._get_config()
        self.assertEqual({
            'categorical_column': {
                'class_name': 'VocabularyListCategoricalColumn',
                'config': {
                    'key': 'aaa',
                    'vocabulary_list': ('omar', 'stringer', 'marlo'),
                    'dtype': 'string',
                    'default_value': 0,
                    'num_oov_buckets': 0
                }
            },
            # 'ckpt_to_load_from': None,
            'combiner': 'mean',
            'cutoff': (1,),
            'dimension': 2,
            'factor': 4,
            'initializer': '_initializer',
            'projection_initializer': '_initializer',
            'max_norm': None,
            'mod8': True,
            'proj0': False,
            # 'tensor_name_in_ckpt': None,
            'trainable': True
        }, config)

        custom_objects = {
            '_initializer': _initializer,
        }

        new_embedding_column = AdaptiveEmbeddingColumn._from_config(config, custom_objects=custom_objects)
        self.assertEqual(embedding_column, new_embedding_column)
        self.assertIsNot(categorical_column, new_embedding_column.categorical_column)

        new_embedding_column = AdaptiveEmbeddingColumn._from_config(
            config,
            custom_objects=custom_objects,
            columns_by_name={categorical_column.name: categorical_column})
        self.assertEqual(embedding_column, new_embedding_column)
        self.assertIs(categorical_column, new_embedding_column.categorical_column)


if __name__ == "__main__":
    tf.test.main()
