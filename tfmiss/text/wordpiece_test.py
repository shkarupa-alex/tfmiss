# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util
from tensorflow.python.ops.lookup_ops import StaticVocabularyTableV1
from tfmiss.text.wordpiece import word_piece


@test_util.run_all_in_graph_and_eager_modes
class WordPieceTest(tf.test.TestCase):
    def _lookup_table(self):
        init = tf.lookup.KeyValueTensorInitializer(
            tf.constant([b'1', b'2', b'3', b'x']), tf.constant([0, 1, 2, 3], 'int64'),
            key_dtype=tf.string, value_dtype=tf.int64)
        lookup = StaticVocabularyTableV1(init, 1, lookup_key_dtype=tf.string)
        self.evaluate(lookup.initializer)

        return lookup

    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = word_piece(source, self._lookup_table())

        self.assertEqual([2, 3, None], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = word_piece(source, self._lookup_table())
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        result = self.evaluate(result)
        self.assertAllEqual((2, 3, 1), result.shape)

    def test_empty(self):
        expected = tf.constant([[]], dtype=tf.string)
        result = word_piece([''], self._lookup_table())
        self.assertIsInstance(result, tf.RaggedTensor)

        expected, result = self.evaluate([expected, result.to_tensor('')])
        self.assertAllEqual(expected, result)

    def test_1d(self):
        expected = tf.constant([['[UNK]']], dtype=tf.string)
        result = word_piece(['12'], self._lookup_table())
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_default_2d(self):
        expected = tf.constant([[['x', '##[UNK]'], ['x', '']]], dtype=tf.string)
        result = word_piece([['xy', 'x']], self._lookup_table(), split_unknown=True)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_ragged(self):
        expected = tf.constant([
            [
                ['[UNK]', '##[UNK]', ''],
                ['[UNK]', '##[UNK]', '##[UNK]'],
                ['', '', '']
            ],
            [
                ['[UNK]', '', ''],
                ['1', '', ''],
                ['1', '##[UNK]', '']
            ]
        ])
        result = word_piece(tf.ragged.constant(
            [['ab', 'c d'], ['e', '1', '19']]), self._lookup_table(), split_unknown=True)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


if __name__ == "__main__":
    tf.test.main()
