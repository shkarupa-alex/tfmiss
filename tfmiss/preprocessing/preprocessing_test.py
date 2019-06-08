from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.preprocessing.preprocessing import cont_bow, skip_gram


@test_util.run_all_in_graph_and_eager_modes
class ContBowTest(tf.test.TestCase):
    def testEmpty(self):
        source = tf.ragged.constant(np.array([]).reshape(0, 2), dtype=tf.string)
        cbows = cont_bow(source, 1)
        target, context = self.evaluate(cbows)

        self.assertAllEqual([], target.tolist())
        self.assertAllEqual([], context.values.tolist())
        self.assertAllEqual([0], context.row_splits.tolist())

    # Disabled due to currently accepting target inclusion into context
    # def testSkipRow(self):
    #     source = tf.ragged.constant([
    #         [],
    #         ['good', 'row'],
    #         ['bad', 'bad'],
    #         [],
    #     ])
    #     target, context = cont_bow(source, 2, seed=1)
    #     target_values, context_values, context_splits = self.evaluate([target, context.values, context.row_splits])
    #
    #     self.assertAllEqual(['good', 'row'], target_values.tolist())
    #     self.assertAllEqual(['row', 'good'], context_values.tolist())
    #     self.assertAllEqual([0, 1, 2], context_splits.tolist())

    def testDense(self):
        source = tf.constant([
            ['the', 'quick', 'brown', 'fox'],
            ['jumped', 'over', 'the', 'dog'],
        ])
        target, context = cont_bow(source, 2, seed=1)
        pairs = self.evaluate(tf.concat([
            tf.expand_dims(target, axis=-1),
            context.to_tensor(default_value='')
        ], axis=-1))

        self.assertAllEqual([
            ['the', 'quick', '', ''],
            ['quick', 'the', 'brown', ''],
            ['brown', 'quick', 'fox', ''],
            ['fox', 'quick', 'brown', ''],
            ['jumped', 'over', '', ''],
            ['over', 'jumped', 'the', 'dog'],
            ['the', 'over', 'dog', ''],
            ['dog', 'over', 'the', '']
        ], pairs.tolist())

    def testRagged(self):
        source = tf.ragged.constant([
            ['', '', ''],
            ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog'],
            [],
            ['tensorflow'],
        ])
        target, context = cont_bow(source, 2, seed=1)
        pairs = self.evaluate(tf.concat([
            tf.expand_dims(target, axis=-1),
            context.to_tensor(default_value='')
        ], axis=-1))

        self.assertAllEqual([
            ['the', 'quick', '', '', ''],
            ['quick', 'the', 'brown', '', ''],
            ['brown', 'quick', 'fox', '', ''],
            ['fox', 'quick', 'brown', 'jumped', 'over'],
            ['jumped', 'fox', 'over', '', ''],
            ['over', 'fox', 'jumped', 'the', 'lazy'],
            ['the', 'over', 'lazy', '', ''],
            ['lazy', 'over', 'the', 'dog', ''],
            ['dog', 'lazy', '', '', '']
        ], pairs.tolist())



@test_util.run_all_in_graph_and_eager_modes
class SkipGramTest(tf.test.TestCase):
    def testEmpty(self):
        source = tf.ragged.constant(np.array([]).reshape(0, 2), dtype=tf.string)
        skipgrams = skip_gram(source, 1)
        target, context = self.evaluate(skipgrams)

        self.assertAllEqual([], target.tolist())
        self.assertAllEqual([], context.tolist())

    def testDense(self):
        source = tf.constant([
            ['the', 'quick', 'brown', 'fox'],
            ['jumped', 'over', 'the', 'dog'],
        ])
        target, context = skip_gram(source, 2, seed=1)
        pairs = self.evaluate(tf.stack([target, context], axis=-1))

        self.assertAllEqual([
            ['the', 'quick'],
            ['quick', 'the'],
            ['quick', 'brown'],
            ['brown', 'quick'],
            ['brown', 'fox'],
            ['fox', 'quick'],
            ['fox', 'brown'],
            ['jumped', 'over'],
            ['over', 'jumped'],
            ['over', 'the'],
            ['over', 'dog'],
            ['the', 'over'],
            ['the', 'dog'],
            ['dog', 'over'],
            ['dog', 'the']
        ], pairs.tolist())

    def testRagged(self):
        source = tf.ragged.constant([
            ['', '', ''],
            ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog'],
            [],
            ['tensorflow'],
        ])
        target, context = skip_gram(source, 2, seed=1)
        pairs = self.evaluate(tf.stack([target, context], axis=-1))

        self.assertAllEqual([
            ['the', 'quick'],
            ['quick', 'the'],
            ['quick', 'brown'],
            ['brown', 'quick'],
            ['brown', 'fox'],
            ['fox', 'quick'],
            ['fox', 'brown'],
            ['fox', 'jumped'],
            ['fox', 'over'],
            ['jumped', 'fox'],
            ['jumped', 'over'],
            ['over', 'fox'],
            ['over', 'jumped'],
            ['over', 'the'],
            ['over', 'lazy'],
            ['the', 'over'],
            ['the', 'lazy'],
            ['lazy', 'over'],
            ['lazy', 'the'],
            ['lazy', 'dog'],
            ['dog', 'lazy']
        ], pairs.tolist())


if __name__ == "__main__":
    tf.test.main()
