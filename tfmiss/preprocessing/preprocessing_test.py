import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.preprocessing.preprocessing import cbow_context, cont_bow, skip_gram, spaces_after
from tfmiss.text.unicode_expand import split_words


@test_util.run_all_in_graph_and_eager_modes
class CbowContextTest(tf.test.TestCase):
    def test_empty(self):
        source = tf.ragged.constant(np.array([]).reshape(0, 2), dtype=tf.string)
        cbows = cbow_context(source, 1, '[PAD]')
        context, position = self.evaluate(cbows)

        self.assertAllEqual([], context.values.tolist())
        self.assertAllEqual([0], context.row_splits.tolist())
        self.assertAllEqual([], position.values.tolist())

    # Disabled due to currently accepting target inclusion into context
    # def test_skip_row(self):
    #     source = tf.ragged.constant([
    #         [],
    #         ['good', 'row'],
    #         ['bad', 'bad'],
    #         [],
    #     ])
    #     context = cbow_context(source, 2, seed=1)
    #     context_values, context_splits = self.evaluate([context.values, context.row_splits])
    #
    #     self.assertAllEqual(['row', 'good'], context_values.tolist())
    #     self.assertAllEqual([0, 1, 2], context_splits.tolist())

    def test_dense(self):
        source = tf.constant([
            ['the', 'quick', 'brown', 'fox'],
            ['jumped', 'over', 'the', 'dog'],
        ])
        context, position = cbow_context(source, 2, '[PAD]')
        context, positions = self.evaluate([context.to_tensor(''), position.to_tensor(0)])

        self.assertAllEqual([
            [b'quick', b'brown', b''],
            [b'the', b'brown', b'fox'],
            [b'the', b'quick', b'fox'],
            [b'quick', b'brown', b''],
            [b'over', b'the', b''],
            [b'jumped', b'the', b'dog'],
            [b'jumped', b'over', b'dog'],
            [b'over', b'the', b'']
        ], context.tolist())
        self.assertAllEqual([
            [1, 2, 0],
            [-1, 1, 2],
            [-2, -1, 1],
            [-2, -1, 0],
            [1, 2, 0],
            [-1, 1, 2],
            [-2, -1, 1],
            [-2, -1, 0]
        ], positions.tolist())

    def test_ragged(self):
        source = tf.ragged.constant([
            ['', '', ''],
            ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog'],
            [],
            ['tensorflow'],
        ])
        context, position = cbow_context(source, 2, '[PAD]')
        context, positions = self.evaluate([context.to_tensor(''), position.to_tensor(0)])

        self.assertAllEqual([
            [b'[PAD]', b'[PAD]', b'', b''],
            [b'[PAD]', b'[PAD]', b'', b''],
            [b'[PAD]', b'[PAD]', b'', b''],
            [b'quick', b'brown', b'', b''],
            [b'the', b'brown', b'fox', b''],
            [b'the', b'quick', b'fox', b'jumped'],
            [b'quick', b'brown', b'jumped', b'over'],
            [b'brown', b'fox', b'over', b'the'],
            [b'fox', b'jumped', b'the', b'lazy'],
            [b'jumped', b'over', b'lazy', b'dog'],
            [b'over', b'the', b'dog', b''],
            [b'the', b'lazy', b'', b''],
            [b'[PAD]', b'[PAD]', b'', b'']
        ], context.tolist())
        self.assertAllEqual([
            [-1, 1, 0, 0],
            [-1, 1, 0, 0],
            [-1, 1, 0, 0],
            [1, 2, 0, 0],
            [-1, 1, 2, 0],
            [-2, -1, 1, 2],
            [-2, -1, 1, 2],
            [-2, -1, 1, 2],
            [-2, -1, 1, 2],
            [-2, -1, 1, 2],
            [-2, -1, 1, 0],
            [-2, -1, 0, 0],
            [-1, 1, 0, 0]
        ], positions.tolist())


@test_util.run_all_in_graph_and_eager_modes
class ContBowTest(tf.test.TestCase):
    def test_empty(self):
        source = tf.ragged.constant(np.array([]).reshape(0, 2), dtype=tf.string)
        cbows = cont_bow(source, 1)
        target, context, position = self.evaluate(cbows)

        self.assertAllEqual([], target.tolist())
        self.assertAllEqual([], context.values.tolist())
        self.assertAllEqual([0], context.row_splits.tolist())
        self.assertAllEqual([], position.values.tolist())

    # Disabled due to currently accepting target inclusion into context
    # def test_skip_row(self):
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

    def test_dense(self):
        source = tf.constant([
            ['the', 'quick', 'brown', 'fox'],
            ['jumped', 'over', 'the', 'dog'],
        ])
        target, context, position = cont_bow(source, 2, seed=1)
        target_context = tf.concat([
            tf.expand_dims(target, axis=-1),
            context.to_tensor(default_value='')
        ], axis=-1)
        pairs, positions = self.evaluate([target_context, position.to_tensor(0)])

        self.assertAllEqual([
            [b'the', b'quick', b'', b''],
            [b'quick', b'the', b'brown', b''],
            [b'brown', b'quick', b'fox', b''],
            [b'fox', b'quick', b'brown', b''],
            [b'jumped', b'over', b'', b''],
            [b'over', b'jumped', b'the', b'dog'],
            [b'the', b'over', b'dog', b''],
            [b'dog', b'over', b'the', b'']
        ], pairs.tolist())
        self.assertAllEqual([
            [1, 0, 0],
            [-1, 1, 0],
            [-1, 1, 0],
            [-2, -1, 0],
            [1, 0, 0],
            [-1, 1, 2],
            [-1, 1, 0],
            [-2, -1, 0],
        ], positions.tolist())

    def test_ragged(self):
        source = tf.ragged.constant([
            ['', '', ''],
            ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog'],
            [],
            ['tensorflow'],
        ])
        target, context, position = cont_bow(source, 2, seed=1)
        target_context = tf.concat([
            tf.expand_dims(target, axis=-1),
            context.to_tensor(default_value='')
        ], axis=-1)
        pairs, positions = self.evaluate([target_context, position.to_tensor(0)])

        self.assertAllEqual([
            [b'the', b'quick', b'', b'', b''],
            [b'quick', b'the', b'brown', b'', b''],
            [b'brown', b'quick', b'fox', b'', b''],
            [b'fox', b'quick', b'brown', b'jumped', b'over'],
            [b'jumped', b'fox', b'over', b'', b''],
            [b'over', b'fox', b'jumped', b'the', b'lazy'],
            [b'the', b'over', b'lazy', b'', b''],
            [b'lazy', b'over', b'the', b'dog', b''],
            [b'dog', b'lazy', b'', b'', b'']
        ], pairs.tolist())
        self.assertAllEqual([
            [1, 0, 0, 0],
            [-1, 1, 0, 0],
            [-1, 1, 0, 0],
            [-2, -1, 1, 2],
            [-1, 1, 0, 0],
            [-2, -1, 1, 2],
            [-1, 1, 0, 0],
            [-2, -1, 1, 0],
            [-1, 0, 0, 0]
        ], positions.tolist())


@test_util.run_all_in_graph_and_eager_modes
class SkipGramTest(tf.test.TestCase):
    def test_empty(self):
        source = tf.ragged.constant(np.array([]).reshape(0, 2), dtype=tf.string)
        skipgrams = skip_gram(source, 1)
        target, context = self.evaluate(skipgrams)

        self.assertAllEqual([], target.tolist())
        self.assertAllEqual([], context.tolist())

    def test_dense(self):
        source = tf.constant([
            ['the', 'quick', 'brown', 'fox'],
            ['jumped', 'over', 'the', 'dog'],
        ])
        target, context = skip_gram(source, 2, seed=1)
        pairs = self.evaluate(tf.stack([target, context], axis=-1))

        self.assertAllEqual([
            [b'the', b'quick'],
            [b'quick', b'the'],
            [b'quick', b'brown'],
            [b'brown', b'quick'],
            [b'brown', b'fox'],
            [b'fox', b'quick'],
            [b'fox', b'brown'],
            [b'jumped', b'over'],
            [b'over', b'jumped'],
            [b'over', b'the'],
            [b'over', b'dog'],
            [b'the', b'over'],
            [b'the', b'dog'],
            [b'dog', b'over'],
            [b'dog', b'the']
        ], pairs.tolist())

    def test_ragged(self):
        source = tf.ragged.constant([
            ['', '', ''],
            ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog'],
            [],
            ['tensorflow'],
        ])
        target, context = skip_gram(source, 2, seed=1)
        pairs = self.evaluate(tf.stack([target, context], axis=-1))

        self.assertAllEqual([
            [b'the', b'quick'],
            [b'quick', b'the'],
            [b'quick', b'brown'],
            [b'brown', b'quick'],
            [b'brown', b'fox'],
            [b'fox', b'quick'],
            [b'fox', b'brown'],
            [b'fox', b'jumped'],
            [b'fox', b'over'],
            [b'jumped', b'fox'],
            [b'jumped', b'over'],
            [b'over', b'fox'],
            [b'over', b'jumped'],
            [b'over', b'the'],
            [b'over', b'lazy'],
            [b'the', b'over'],
            [b'the', b'lazy'],
            [b'lazy', b'over'],
            [b'lazy', b'the'],
            [b'lazy', b'dog'],
            [b'dog', b'lazy']
        ], pairs.tolist())


@test_util.run_all_in_graph_and_eager_modes
class SpacesAfterTest(tf.test.TestCase):
    def test_empty(self):
        source = tf.ragged.constant(np.array([]).reshape(0, 2), dtype=tf.string)
        tokens, spaces = spaces_after(source)
        tokens, spaces = self.evaluate([tokens, spaces])

        self.assertAllEqual([], tokens.values.tolist())
        self.assertAllEqual([0], tokens.row_splits.tolist())
        self.assertAllEqual([], spaces.values.tolist())
        self.assertAllEqual([0], spaces.row_splits.tolist())

    def test_dense(self):
        source = tf.constant([
            ['the', ' ', 'quick', 'brown', ' ', 'fox'],
            ['jumped', 'over', ' ', 'the', 'dog', ' '],
        ])
        tokens, spaces = spaces_after(source)
        tokens, spaces = self.evaluate([tokens.to_tensor(''), spaces.to_tensor('')])

        self.assertAllEqual([
            [b'the', b'quick', b'brown', b'fox'],
            [b'jumped', b'over', b'the', b'dog'],
        ], tokens.tolist())
        self.assertAllEqual([
            [b' ', b'', b' ', b''],
            [b'', b' ', b'', b' '],
        ], spaces.tolist())

    def test_ragged(self):
        source = tf.ragged.constant([
            ['', '', ''],
            ['the', ' ', 'quick', 'brown', 'fox'],
            [],
            ['tensorflow'],
        ])
        tokens, spaces = spaces_after(source)
        tokens, spaces = self.evaluate([tokens.to_tensor(''), spaces.to_tensor('')])

        self.assertAllEqual([
            [b'', b'', b'', b''],
            [b'the', b'quick', b'brown', b'fox'],
            [b'', b'', b'', b''],
            [b'tensorflow', b'', b'', b''],
        ], tokens.tolist())
        self.assertAllEqual([
            [b'', b'', b'', b''],
            [b' ', b'', b'', b''],
            [b'', b'', b'', b''],
            [b'', b'', b'', b''],
        ], spaces.tolist())

    def test_start(self):
        source = tf.constant([
            [' ', 'the', ' ', 'quick'],
            ['\u200b', 'jumped', 'over', '\ufeff'],
            [' ', '\u200b', '', '']
        ])
        tokens, spaces = spaces_after(source)
        tokens, spaces = self.evaluate([tokens.to_tensor(''), spaces.to_tensor('')])

        self.assertAllEqual([
            [b'the', b'quick'],
            [b'jumped', b'over'],
            [b'', b''],
        ], tokens.tolist())
        self.assertAllEqual([
            [b' ', b''],
            [b'', b'\xef\xbb\xbf'],
            [b'', b''],
        ], spaces.tolist())

    def test_space(self):
        sure_spaces = [
            '\t', '\n', '\x0b', '\x0c', '\r', '\x1c', '\x1d', '\x1e', '\x1f', ' ', '\x85', '\xa0', '\u1680', '\u2000',
            '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a',
            '\u2028', '\u2029', '\u200b', '\u202f', '\u205f', '\u2060', '\u2061', '\u2800', '\u3000', '\ufeff']
        source = split_words([' {}W{}W{} {} '.format(s, s, s, s) for s in sure_spaces], extended=True)

        tokens, spaces = spaces_after(source)
        tokens, spaces = self.evaluate([tokens.to_tensor(''), spaces.to_tensor('')])

        self.assertAllEqual([[b'W', b'W']] * 34, tokens.tolist())
        self.assertAllEqual([[s.encode('utf-8'), '{} {} '.format(s, s).encode('utf-8')]
                             for s in sure_spaces], spaces.tolist())


if __name__ == "__main__":
    tf.test.main()
