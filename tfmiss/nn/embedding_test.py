from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.nn.embedding import adaptive_embedding_lookup, adaptive_embedding_lookup_sparse
from tfmiss.nn.embedding import safe_adaptive_embedding_lookup_sparse


@test_util.run_all_in_graph_and_eager_modes
class AdaptiveEmbeddingLookupTest(tf.test.TestCase):
    def test_max_norm_constant(self):
        params = [tf.constant([[2.0]]), tf.constant([[2.0]])]
        ids = tf.constant([0], dtype=tf.int32)
        transforms = [lambda embed: _transform_embedding(1, embed)] * len(params)
        embeddings = adaptive_embedding_lookup(params, ids, transforms, max_norm=1.0)
        self.assertAllEqual(self.evaluate(embeddings), [[1.0]])

    def test_max_norm_nontrivial(self):
        params = [tf.constant([[2.0, 4.0], [3.0, 1.0]]), tf.constant([[2.0, 4.0], [3.0, 1.0]])]
        ids = tf.constant([0, 1], dtype=tf.int32)
        transforms = [lambda embed: _transform_embedding(2, embed)] * len(params)
        embeddings = adaptive_embedding_lookup(params, ids, transforms, max_norm=2.0)

        norms = tf.math.sqrt(tf.math.reduce_sum(embeddings * embeddings, axis=1))
        normalized = embeddings / tf.stack([norms, norms], axis=1)
        self.assertAllClose(self.evaluate(embeddings), 2 * self.evaluate(normalized))

    def test_ids_dtype(self):
        params = [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]]
        ids32 = tf.constant([[0], [1]], dtype=tf.int32)
        ids64 = tf.constant([[0], [1]], dtype=tf.int64)
        transforms = [lambda embed: _transform_embedding(4, embed)] * len(params)
        embeddings32 = adaptive_embedding_lookup(params, ids32, transforms)
        embeddings64 = adaptive_embedding_lookup(params, ids64, transforms)
        self.assertAllEqual(self.evaluate(embeddings32), self.evaluate(embeddings64))

    def test_variables(self):
        p = [
            tf.Variable(tf.zeros(shape=[100, 100], dtype=tf.float32)),
            tf.Variable(tf.zeros(shape=[100, 100], dtype=tf.float32)),
        ]
        ids = tf.constant([0, 1, 1, 17], dtype=tf.int32)
        transforms = [lambda embed: _transform_embedding(100, embed)] * 2
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(adaptive_embedding_lookup(p, ids, transforms))

    def test_adaptive_no_error(self):
        params = [
            [[1.0, 1.0, 1.0, 1.0],
             [1.1, 1.1, 1.1, 1.1],
             [1.2, 1.2, 1.2, 1.2],
             [1.3, 1.3, 1.3, 1.3]],
            [[2.0, 2.0, 2.0],
             [2.1, 2.1, 2.1],
             [2.2, 2.2, 2.2]],
            [[3.0, 3.0],
             [3.1, 3.1]],
            [[4.0]]
        ]
        ids = [[0, 8], [3, 9]]

        transforms = [lambda embed: _transform_embedding(4, embed)] * len(params)
        embeddings = adaptive_embedding_lookup(params, ids, transforms)
        expected = [
            [[1.0, 1.0, 1.0, 1.0],
             [3.1, 3.1, 0.0, 0.0]],
            [[1.3, 1.3, 1.3, 1.3],
             [4.0, 0.0, 0.0, 0.0]]
        ]
        self.assertAllClose(self.evaluate(embeddings), expected)

    def test_adaptive_shape(self):
        params = [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]]
        ids = [[0], [1]]
        transforms = [lambda embed: _transform_embedding(4, embed)] * len(params)
        embeddings = adaptive_embedding_lookup(params, ids, transforms)
        self.assertAllEqual(embeddings.shape, [2, 1, 4])

        actual_shape = tf.shape(embeddings)
        self.assertAllEqual(self.evaluate(actual_shape), [2, 1, 4])

    def test_unknown_shape(self):
        params = [tf.constant([[1.0, 1.0, 1.0, 1.0]]), tf.constant([[2.0, 2.0, 2.0]])]
        params[0]._shape_val = tf.TensorShape([None, 4])

        ids = tf.constant([[0], [1]])
        transforms = [lambda embed: _transform_embedding(4, embed)] * len(params)
        embeddings = adaptive_embedding_lookup(params, ids, transforms)
        self.assertAllEqual(embeddings.shape, [2, 1, 4])

        actual_shape = tf.shape(embeddings)
        self.assertAllEqual(self.evaluate(actual_shape), [2, 1, 4])

    def test_adaptive_error(self):
        with self.assertRaisesRegexp(ValueError, '2 variable'):
            self.evaluate(adaptive_embedding_lookup(
                [[0.]],
                [0],
                [lambda embed: _transform_embedding(2, embed)]
            ))
        with self.assertRaisesRegexp(ValueError, 'corresponding transform'):
            self.evaluate(adaptive_embedding_lookup([[[0.]], [[0.]]], [0], []))
        with self.assertRaisesRegexp(ValueError, 'should be callable'):
            self.evaluate(adaptive_embedding_lookup(
                [[[0.]], [[0.]]],
                [0],
                [lambda embed: _transform_embedding(2, embed), None]
            ))
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, 'id should be less'):
            self.evaluate(adaptive_embedding_lookup(
                [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]],
                [3],
                [lambda embed: _transform_embedding(4, embed)] * 2
            ))


@test_util.run_all_in_graph_and_eager_modes
class AdaptiveEmbeddingLookupSparseTest(tf.test.TestCase):
    def _random_ids_weights(self, batch_size, vocab_size):
        max_val_per_entry = 6
        vals_per_batch_entry = np.random.randint(1, max_val_per_entry, size=batch_size)
        num_vals = np.sum(vals_per_batch_entry)

        ids = np.random.randint(vocab_size, size=num_vals)
        weights = 1 + np.random.rand(num_vals)

        indices = []
        for batch_entry, num_val in enumerate(vals_per_batch_entry):
            for val_index in range(num_val):
                indices.append([batch_entry, val_index])

        shape = [batch_size, max_val_per_entry]

        sp_ids = tf.SparseTensor(
            tf.constant(indices, tf.int64),
            tf.constant(ids, tf.int32),
            tf.constant(shape, tf.int64)
        )
        sp_weights = tf.SparseTensor(
            tf.constant(indices, tf.int64),
            tf.constant(weights, tf.float32),
            tf.constant(shape, tf.int64)
        )

        return sp_ids, sp_weights, ids, weights, vals_per_batch_entry

    def test_embedding_lookup_sparse(self):
        vocab_size = 13
        batch_size = 10
        param_shape = [9, 10]

        expected_lookup_result_shape = [batch_size] + param_shape[1:]
        expected_lookup_result_shape_mode = list(expected_lookup_result_shape)
        if not tf.executing_eagerly():
            expected_lookup_result_shape_mode[0] = None

        sp_ids, sp_weights, ids, weights, vals_per_batch_entry = self._random_ids_weights(batch_size, vocab_size)

        for num_shards, combiner, dtype, ignore_weights in itertools.product(
                [2, 5], ['sum', 'mean', 'sqrtn'],
                [tf.float16, tf.bfloat16, tf.float32, tf.float64],
                [True]):  # , False

            params = [
                tf.cast(np.random.rand(param_shape[0], param_shape[-1] - s), dtype=dtype)
                for s in range(num_shards)
            ]
            transforms = [lambda embed: _transform_embedding(param_shape[-1], embed)] * len(params)
            embedding_sum = adaptive_embedding_lookup_sparse(
                params,
                sp_ids,
                None if ignore_weights else sp_weights,
                transforms,
                combiner=combiner
            )
            self.assertEqual(embedding_sum.get_shape().as_list(), expected_lookup_result_shape_mode)

            if dtype in (tf.float16, tf.bfloat16):
                self.assertEqual(embedding_sum.dtype, tf.float32)
            else:
                self.assertEqual(embedding_sum.dtype, dtype)

            tf_embedding_sum = self.evaluate(embedding_sum)
            self.assertEqual(list(tf_embedding_sum.shape), expected_lookup_result_shape)

    def test_incompatible_shapes(self):
        params = [np.random.rand(5, 10), np.random.rand(5, 5)]
        transforms = [lambda embed: _transform_embedding(10, embed)] * len(params)
        sp_ids = tf.SparseTensor(
            tf.constant([[0, 0], [0, 1], [1, 0]], tf.int64),
            tf.constant([0, 1, 2], tf.int32),
            tf.constant([2, 2], tf.int64)
        )
        sp_weights = tf.SparseTensor(
            tf.constant([[0, 0], [0, 1]], tf.int64),
            tf.constant([12.0, 5.0], tf.float32),
            tf.constant([1, 2], tf.int64)
        )

        with self.assertRaises(ValueError):
            adaptive_embedding_lookup_sparse(params, sp_ids, sp_weights, transforms, combiner='mean')


@test_util.run_all_in_graph_and_eager_modes
class SafeAdaptiveEmbeddingLookupSparseTest(tf.test.TestCase):
    def _random_weights(self, vocab_size=4, embed_dim=4, num_shards=2):
        assert vocab_size > 0
        assert embed_dim > 0
        assert num_shards > 0
        assert num_shards <= vocab_size

        initializer = tf.compat.v1.truncated_normal_initializer(
            mean=0.0, stddev=1.0 / tf.math.sqrt(vocab_size * 1.0), dtype=tf.float32)
        embedding_weights = list(tf.compat.v1.get_variable(
            name="embedding_weights",
            shape=[vocab_size, embed_dim],
            partitioner=tf.compat.v1.fixed_size_partitioner(num_shards),
            initializer=initializer)
        )
        for w in embedding_weights:
            self.evaluate(w.initializer)
        embedding_weights = [self.evaluate(w) for w in embedding_weights]
        embedding_transforms = [lambda embed: _transform_embedding(embed_dim, embed)] * len(embedding_weights)

        return embedding_weights, embedding_transforms

    def _ids_and_weights_2d(self):
        # Each row demonstrates a test case:
        #   Row 0: multiple valid ids, 1 invalid id, weighted mean
        #   Row 1: all ids are invalid (leaving no valid ids after pruning)
        #   Row 2: no ids to begin with
        #   Row 3: single id
        #   Row 4: all ids have <=0 weight
        indices = [[0, 0], [0, 1], [0, 2], [1, 0], [3, 0], [4, 0], [4, 1]]
        ids = [0, 1, -1, -1, 2, 0, 1]
        weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
        shape = [5, 4]

        sparse_ids = tf.SparseTensor(
            tf.constant(indices, tf.int64),
            tf.constant(ids, tf.int64),
            tf.constant(shape, tf.int64)
        )

        sparse_weights = tf.SparseTensor(
            tf.constant(indices, tf.int64),
            tf.constant(weights, tf.float32),
            tf.constant(shape, tf.int64)
        )

        return sparse_ids, sparse_weights

    def _ids_and_weights_3d(self):
        # Each (2-D) index demonstrates a test case:
        #   Index 0, 0: multiple valid ids, 1 invalid id, weighted mean
        #   Index 0, 1: all ids are invalid (leaving no valid ids after pruning)
        #   Index 0, 2: no ids to begin with
        #   Index 1, 0: single id
        #   Index 1, 1: all ids have <=0 weight
        #   Index 1, 2: no ids to begin with
        indices = [
            [0, 0, 0], [0, 0, 1], [0, 0, 2],
            [0, 1, 0],
            [1, 0, 0],
            [1, 1, 0], [1, 1, 1]
        ]
        ids = [0, 1, -1, -1, 2, 0, 1]
        weights = [1.0, 2.0, 1.0, 1.0, 3.0, 0.0, -0.5]
        shape = [2, 3, 4]

        sparse_ids = tf.SparseTensor(
            tf.constant(indices, tf.int64),
            tf.constant(ids, tf.int64),
            tf.constant(shape, tf.int64)
        )

        sparse_weights = tf.SparseTensor(
            tf.constant(indices, tf.int64),
            tf.constant(weights, tf.float32),
            tf.constant(shape, tf.int64)
        )

        return sparse_ids, sparse_weights

    def test_safe_embedding_lookup_sparse_return_zero_vector(self):
        embedding_weights, embedding_transforms = self._random_weights()
        sparse_ids, sparse_weights = self._ids_and_weights_2d()

        embedding_lookup_result = self.evaluate(safe_adaptive_embedding_lookup_sparse(
            embedding_weights, sparse_ids, sparse_weights, embedding_transforms))

        print(embedding_weights)
        self.assertAllClose(
            embedding_lookup_result,
            [(1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) /
             3.0, [0] * 4, [0] * 4, embedding_weights[1][0], [0] * 4])

    def test_safe_embedding_lookup_sparse_return_special_vector(self):
        embedding_weights, embedding_transforms = self._random_weights()
        sparse_ids, sparse_weights = self._ids_and_weights_2d()

        embedding_lookup_result = self.evaluate(safe_adaptive_embedding_lookup_sparse(
            embedding_weights, sparse_ids, sparse_weights, embedding_transforms, default_id=3))

        self.assertAllClose(
            embedding_lookup_result,
            [(1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) /
             3.0, embedding_weights[1][1], embedding_weights[1][1],
             embedding_weights[1][0], embedding_weights[1][1]])

    def test_safe_embedding_lookup_sparse_no_weights(self):
        embedding_weights, embedding_transforms = self._random_weights()
        sparse_ids, _ = self._ids_and_weights_2d()

        embedding_lookup_result = self.evaluate(safe_adaptive_embedding_lookup_sparse(
            embedding_weights, sparse_ids, None, embedding_transforms))

        self.assertAllClose(
            embedding_lookup_result,
            [(embedding_weights[0][0] + embedding_weights[0][1]) / 2.0, [0] * 4,
             [0] * 4, embedding_weights[1][0], (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0])

    def test_safe_embedding_lookup_sparse_partitioned(self):
        embedding_weights, embedding_transforms = self._random_weights(num_shards=3)
        sparse_ids, _ = self._ids_and_weights_2d()

        embedding_lookup_result = self.evaluate(safe_adaptive_embedding_lookup_sparse(
            embedding_weights, sparse_ids, None, embedding_transforms))

        embedding_weights = list(itertools.chain(*embedding_weights))
        self.assertAllClose(embedding_lookup_result,
                            [(embedding_weights[0] + embedding_weights[1]) / 2.0,
                             [0] * 4, [0] * 4, embedding_weights[2],
                             (embedding_weights[0] + embedding_weights[1]) / 2.0])

    def test_safe_embedding_lookup_sparse_partitioned_inconsistent_weights(self):
        embedding_weights, embedding_transforms = self._random_weights(num_shards=3)
        sparse_ids, sparse_weights = self._ids_and_weights_2d()

        embedding_weights[1] = embedding_weights[1].astype(np.float64)
        self.assertRaises(TypeError, safe_adaptive_embedding_lookup_sparse,
                          embedding_weights, sparse_ids)
        embedding_weights = [
            tf.constant(w, dtype=tf.float64)
            for w in embedding_weights
        ]
        self.assertRaises(ValueError, safe_adaptive_embedding_lookup_sparse,
                          embedding_weights, sparse_ids, sparse_weights, embedding_transforms)

    def test_safe_embedding_lookup_sparse_3d_return_zero_vector(self):
        embedding_weights, embedding_transforms = self._random_weights()
        sparse_ids, sparse_weights = self._ids_and_weights_3d()

        embedding_lookup_result = self.evaluate(safe_adaptive_embedding_lookup_sparse(
            embedding_weights, sparse_ids, sparse_weights, embedding_transforms))

        self.assertAllClose(embedding_lookup_result, [[
            (1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) / 3.0,
            [0] * 4, [0] * 4
        ], [embedding_weights[1][0], [0] * 4, [0] * 4]])

    def test_safe_embedding_lookup_sparse_3d_return_special_vector(self):
        embedding_weights, embedding_transforms = self._random_weights()
        sparse_ids, sparse_weights = self._ids_and_weights_3d()

        embedding_lookup_result = self.evaluate(
            safe_adaptive_embedding_lookup_sparse(
                embedding_weights, sparse_ids, sparse_weights, embedding_transforms,
                default_id=3))

        self.assertAllClose(
            embedding_lookup_result,
            [[(1.0 * embedding_weights[0][0] + 2.0 * embedding_weights[0][1]) /
              3.0, embedding_weights[1][1], embedding_weights[1][1]], [
                 embedding_weights[1][0], embedding_weights[1][1],
                 embedding_weights[1][1]
             ]])

    def test_safe_embedding_lookup_sparse_3d_no_weights(self):
        embedding_weights, embedding_transforms = self._random_weights()
        sparse_ids, _ = self._ids_and_weights_3d()

        embedding_lookup_result = self.evaluate(safe_adaptive_embedding_lookup_sparse(
            embedding_weights, sparse_ids, None, embedding_transforms))

        self.assertAllClose(
            embedding_lookup_result,
            [[(embedding_weights[0][0] + embedding_weights[0][1]) / 2.0, [0] * 4, [0] * 4],
             [embedding_weights[1][0], (embedding_weights[0][0] + embedding_weights[0][1]) / 2.0, [0] * 4]])

    def test_safe_embedding_lookup_sparse_3d_partitioned(self):
        embedding_weights, embedding_transforms = self._random_weights(num_shards=3)
        sparse_ids, _ = self._ids_and_weights_3d()

        embedding_lookup_result = self.evaluate(safe_adaptive_embedding_lookup_sparse(
                embedding_weights, sparse_ids, None, embedding_transforms))

        embedding_weights = list(itertools.chain(*embedding_weights))
        self.assertAllClose(embedding_lookup_result, [[
            (embedding_weights[0] + embedding_weights[1]) / 2.0, [0] * 4, [0] * 4
        ], [
            embedding_weights[2],
            (embedding_weights[0] + embedding_weights[1]) / 2.0, [0] * 4
        ]])

    def test_safe_embedding_lookup_sparse_3d_partitioned_inconsistent_weights(self):
        embedding_weights, embedding_transforms = self._random_weights(num_shards=3)
        sparse_ids, sparse_weights = self._ids_and_weights_3d()

        # embedding_weights[1] = embedding_weights[1].astype(np.float64)
        # self.assertRaises(TypeError, safe_adaptive_embedding_lookup_sparse,
        #                   embedding_weights, sparse_ids, None, embedding_transforms)
        embedding_weights = [tf.constant(w, dtype=tf.float64) for w in embedding_weights]
        self.assertRaises(ValueError, safe_adaptive_embedding_lookup_sparse,
                          embedding_weights, sparse_ids, sparse_weights, embedding_transforms)


def _transform_embedding(max_dim, embed):
    pad_right = max_dim - tf.shape(embed)[-1]
    pad_res = tf.pad(embed, [[0, 0], [0, pad_right]], 'CONSTANT')
    pad_res.set_shape(embed.get_shape()[:1].concatenate([max_dim]))

    return pad_res


if __name__ == "__main__":
    tf.test.main()
