from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.nn.embedding import embedding_lookup_adaptive


@test_util.run_all_in_graph_and_eager_modes
class EmbeddingLookupAdaptiveTest(tf.test.TestCase):
    def testMaxNormConstant(self):
        params = [tf.constant([[2.0]]), tf.constant([[2.0]])]
        ids = tf.constant([0], dtype=tf.int32)
        transforms = [lambda embed: _transform_embedding(1, embed)] * len(params)
        embeddings = embedding_lookup_adaptive(params, ids, transforms, max_norm=1.0)
        self.assertAllEqual(self.evaluate(embeddings), [[1.0]])

    def testMaxNormNontrivial(self):
        params = [tf.constant([[2.0, 4.0], [3.0, 1.0]]), tf.constant([[2.0, 4.0], [3.0, 1.0]])]
        ids = tf.constant([0, 1], dtype=tf.int32)
        transforms = [lambda embed: _transform_embedding(2, embed)] * len(params)
        embeddings = embedding_lookup_adaptive(params, ids, transforms, max_norm=2.0)

        norms = tf.math.sqrt(tf.math.reduce_sum(embeddings * embeddings, axis=1))
        normalized = embeddings / tf.stack([norms, norms], axis=1)
        self.assertAllClose(self.evaluate(embeddings), 2 * self.evaluate(normalized))

    def testIdsDtype(self):
        params = [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]]
        ids32 = tf.constant([[0], [1]], dtype=tf.int32)
        ids64 = tf.constant([[0], [1]], dtype=tf.int64)
        transforms = [lambda embed: _transform_embedding(4, embed)] * len(params)
        embeddings32 = embedding_lookup_adaptive(params, ids32, transforms)
        embeddings64 = embedding_lookup_adaptive(params, ids64, transforms)
        self.assertAllEqual(self.evaluate(embeddings32), self.evaluate(embeddings64))

    def testVariables(self):
        p = [
            tf.Variable(tf.zeros(shape=[100, 100], dtype=tf.float32)),
            tf.Variable(tf.zeros(shape=[100, 100], dtype=tf.float32)),
        ]
        ids = tf.constant([0, 1, 1, 17], dtype=tf.int32)
        transforms = [lambda embed: _transform_embedding(100, embed)] * 2
        self.evaluate(tf.compat.v1.global_variables_initializer())
        self.evaluate(embedding_lookup_adaptive(p, ids, transforms))

    def testAdaptiveNoError(self):
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
        embeddings = embedding_lookup_adaptive(params, ids, transforms)
        expected = [
            [[1.0, 1.0, 1.0, 1.0],
             [3.1, 3.1, 0.0, 0.0]],
            [[1.3, 1.3, 1.3, 1.3],
             [4.0, 0.0, 0.0, 0.0]]
        ]
        self.assertAllClose(self.evaluate(embeddings), expected)

    def testAdaptiveShape(self):
        params = [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]]
        ids = [[0], [1]]
        transforms = [lambda embed: _transform_embedding(4, embed)] * len(params)
        embeddings = embedding_lookup_adaptive(params, ids, transforms)
        self.assertAllEqual(embeddings.shape, [2, 1, 4])

        actual_shape = tf.shape(embeddings)
        self.assertAllEqual(self.evaluate(actual_shape), [2, 1, 4])

    def testUnknownShape(self):
        params = [tf.constant([[1.0, 1.0, 1.0, 1.0]]), tf.constant([[2.0, 2.0, 2.0]])]
        params[0]._shape_val = tf.TensorShape([None, 4])

        ids = tf.constant([[0], [1]])
        transforms = [lambda embed: _transform_embedding(4, embed)] * len(params)
        embeddings = embedding_lookup_adaptive(params, ids, transforms)
        self.assertAllEqual(embeddings.shape, [2, 1, 4])

        actual_shape = tf.shape(embeddings)
        self.assertAllEqual(self.evaluate(actual_shape), [2, 1, 4])

    def testAdaptiveError(self):
        with self.assertRaisesRegexp(ValueError, 'two params'):
            self.evaluate(embedding_lookup_adaptive(
                [[0.]],
                [0],
                [lambda embed: _transform_embedding(2, embed)]
            ))
        with self.assertRaisesRegexp(ValueError, 'corresponding transform'):
            self.evaluate(embedding_lookup_adaptive([[[0.]], [[0.]]], [0], []))
        with self.assertRaisesRegexp(ValueError, 'should be callable'):
            self.evaluate(embedding_lookup_adaptive(
                [[[0.]], [[0.]]],
                [0],
                [lambda embed: _transform_embedding(2, embed), None]
            ))
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, 'id should be less'):
            self.evaluate(embedding_lookup_adaptive(
                [[[1.0, 1.0, 1.0, 1.0]], [[2.0, 2.0, 2.0]]],
                [3],
                [lambda embed: _transform_embedding(4, embed)] * 2
            ))


def _transform_embedding(max_dim, embed):
    pad_right = max_dim - tf.shape(embed)[-1]

    return tf.pad(embed, [[0, 0], [0, pad_right]], 'CONSTANT')


if __name__ == "__main__":
    tf.test.main()
