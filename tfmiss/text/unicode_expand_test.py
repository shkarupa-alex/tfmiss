# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.text.unicode_expand import char_ngrams, split_chars, split_words


@test_util.run_all_in_graph_and_eager_modes
class CharNgramsTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = char_ngrams(source, 1, 1, itself='ALWAYS')

        self.assertEqual([2, 3, None], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = char_ngrams(source, 1, 1, itself='ALWAYS')
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        result = self.evaluate(result)
        self.assertAllEqual((2, 3, 1), result.shape)

    def test_empty(self):
        expected = tf.constant([], dtype=tf.string)
        result = char_ngrams('', 1, 1, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_0d(self):
        expected = tf.constant(['x', 'y'], dtype=tf.string)
        result = char_ngrams('xy', 1, 1, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_1d(self):
        expected = tf.constant([['x', 'y']], dtype=tf.string)
        result = char_ngrams(['xy'], 1, 1, itself='ASIS')
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_default_2d(self):
        expected = tf.constant([[['x', 'y'], ['x', '']]], dtype=tf.string)
        result = char_ngrams([['xy', 'x']], 1, 1, itself='ASIS')
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_ragged(self):
        expected = tf.constant([
            [
                ['a', 'b', 'ab', '', ''],
                ['c', ' ', 'd', 'c ', ' d'],
            ],
            [
                ['e', '', '', '', ''],
                ['', '', '', '', '']
            ]
        ])
        result = char_ngrams(tf.ragged.constant([['ab', 'c d'], ['e']]), 1, 2, itself='ASIS')
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_none(self):
        result = char_ngrams('123', 4, 5, itself='ASIS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([], result.tolist())

    def test_as_is_below(self):
        result = char_ngrams('1234', 2, 3, itself='ASIS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234'], result.tolist())

    def test_as_is_inside(self):
        result = char_ngrams('123', 2, 3, itself='ASIS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'123'], result.tolist())

    def test_as_is_above(self):
        result = char_ngrams('123', 4, 5, itself='ASIS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([], result.tolist())

    def test_never_below(self):
        result = char_ngrams('1234', 2, 3, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234'], result.tolist())

    def test_never_inside(self):
        result = char_ngrams('123', 2, 3, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23'], result.tolist())

    def test_never_above(self):
        result = char_ngrams('123', 4, 5, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([], result.tolist())

    def test_always_below(self):
        result = char_ngrams('1234', 2, 3, itself='ALWAYS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234', b'1234'], result.tolist())

    def test_always_inside(self):
        result = char_ngrams('123', 2, 3, itself='ALWAYS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'123'], result.tolist())

    def test_always_above(self):
        result = char_ngrams('123', 4, 5, itself='ALWAYS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'123'], result.tolist())

    def test_alone_below(self):
        result = char_ngrams('1234', 2, 3, itself='ALONE')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234'], result.tolist())

    def test_alone_inside(self):
        result = char_ngrams('123', 2, 3, itself='ALONE')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23'], result.tolist())

    def test_alone_above(self):
        result = char_ngrams('123', 4, 5, itself='ALONE')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'123'], result.tolist())

    def test_skip(self):
        expected = tf.constant([['x', 'y'], ['[UNK]', '']], dtype=tf.string)
        result = char_ngrams(['xy', '[UNK]'], 1, 1, itself='ASIS', skip=['[UNK]'])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class SplitCharsTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = split_chars(source)

        self.assertEqual([2, 3, None], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = split_chars(source)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        result = self.evaluate(result)
        self.assertEqual((2, 3, 1), result.shape)

    def test_empty(self):
        expected = tf.constant([], dtype=tf.string)
        result = split_chars('')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_0d(self):
        expected = tf.constant(['x', 'y'], dtype=tf.string)

        result = split_chars('xy')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_1d(self):
        expected = tf.constant([['x', 'y']], dtype=tf.string)

        result = split_chars(['xy'])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_2d(self):
        expected = tf.constant([[['x', 'y']]], dtype=tf.string)

        result = split_chars([['xy']])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_ragged(self):
        expected = tf.constant([[['a', 'b', ''], ['c', ' ', 'd']], [['e', '', ''], ['', '', '']]])

        result = split_chars(tf.ragged.constant([['ab', 'c d'], ['e']]))
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = tf.constant([u'ё', u' ', u'е', u'̈', u'2', u'⁵'], dtype=tf.string)

        result = split_chars(u'ё ё2⁵')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_restore(self):
        source = tf.constant('Hey\n\tthere\t«word», !!!')

        splitted = split_chars(source)
        self.assertIsInstance(splitted, tf.Tensor)
        restored = tf.strings.reduce_join(splitted)

        source, restored = self.evaluate(source), self.evaluate(restored)
        self.assertAllEqual(source, restored)

    def test_skip(self):
        expected = tf.constant([['x', 'y'], ['zz', '']], dtype=tf.string)

        result = split_chars(['xy', 'zz'], skip=['zz'])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class SplitWordsTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = split_words(source)

        self.assertEqual([2, 3, None], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = split_words(source)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        result = self.evaluate(result)
        self.assertEqual((2, 3, 1), result.shape)

    def test_empty(self):
        expected = tf.constant([''], dtype=tf.string)
        result = split_words('')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_0d(self):
        expected = tf.constant(['x', '!'], dtype=tf.string)
        result = split_words('x!')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_1d(self):
        expected = tf.constant([['x', '!']], dtype=tf.string)
        result = split_words(['x!'])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_2d(self):
        expected = tf.constant([[['x', '!']]], dtype=tf.string)
        result = split_words([['x!']])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_ragged(self):
        expected = tf.constant([[['ab', '', ''], ['c', ' ', 'd']], [['e', '', ''], ['', '', '']]])

        result = split_words(tf.ragged.constant([['ab', 'c d'], ['e']]))
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_restore(self):
        source = tf.constant('Hey\n\tthere\t«word», !!!')

        splitted = split_words(source)
        self.assertIsInstance(splitted, tf.Tensor)
        restored = tf.strings.reduce_join(splitted)

        source, restored = self.evaluate(source), self.evaluate(restored)
        self.assertAllEqual(source, restored)

    def test_wrapped(self):
        expected = tf.constant([
            [' ', '"', 'word', '"', ' '],
            [' ', u'«', 'word', u'»', ' '],
            [' ', u'„', 'word', u'“', ' '],
            [' ', '{', 'word', '}', ' '],
            [' ', '(', 'word', ')', ' '],
            [' ', '[', 'word', ']', ' '],
            [' ', '<', 'word', '>', ' '],
        ])
        result = split_words([
            ' "word" ',
            u' «word» ',
            u' „word“ ',
            ' {word} ',
            ' (word) ',
            ' [word] ',
            ' <word> ',
        ])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_word_punkt(self):
        expected = tf.constant([
            [' ', 'word', '.', ' ', '', ''],
            [' ', 'word', '.', '.', ' ', ''],
            [' ', 'word', '.', '.', '.', ' '],
            [' ', 'word', u'…', ' ', '', ''],
            [' ', 'word', ',', ' ', '', ''],
            [' ', 'word', '.', ',', ' ', ''],
            [' ', 'word', ':', ' ', '', ''],
            [' ', 'word', ';', ' ', '', ''],
            [' ', 'word', '!', ' ', '', ''],
            [' ', 'word', '?', ' ', '', ''],
            [' ', 'word', '%', ' ', '', ''],
            [' ', '$', 'word', ' ', '', ''],
        ])
        result = split_words([
            ' word. ',
            ' word.. ',
            ' word... ',
            u' word… ',
            ' word, ',
            ' word., ',
            ' word: ',
            ' word; ',
            ' word! ',
            ' word? ',
            ' word% ',
            ' $word ',
        ])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_complex_word(self):
        expected = tf.constant([
            [' ', 'test', '@', 'test.com', ' ', '', '', '', ''],
            [' ', 'www.test.com', ' ', '', '', '', '', '', ''],
            [' ', 'word', '.', '.', 'word', ' ', '', '', ''],
            [' ', 'word', '+', 'word', '-', 'word', ' ', '', ''],
            [' ', 'word', '\\', 'word', '/', 'word', '#', 'word', ' '],
        ])
        result = split_words([
            ' test@test.com ',
            ' www.test.com ',
            ' word..word ',
            ' word+word-word ',
            ' word\\word/word#word ',
        ])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    # def test_icu_word_break(self):
    #     # WORD_BREAK_TEST_URL = 'https://www.unicode.org/Public/UCD/latest/ucd/auxiliary/WordBreakTest.txt'
    #     # test_data = urlopen(WORD_BREAK_TEST_URL).read().decode('utf-8').strip().split('\n')
    #
    #     test_file = os.path.join(os.path.dirname(__file__), 'WordBreakTest.txt')
    #     with open(test_file, 'rb') as ft:
    #         test_data = ft.read().decode('utf-8').strip().split('\n')
    #
    #     expected, source, description = [], [], []
    #     for row, line in enumerate(test_data):
    #         if line.startswith('#'):
    #             continue
    #
    #         example, rule = line.split('#')
    #         example = example.strip().strip(u'÷').strip().replace(u'÷', '00F7').replace(u'×', '00D7').split(' ')
    #         example = [code.zfill(8) if len(code) > 4 else code.zfill(4) for code in example]
    #         example = [u'\\U{}'.format(code) if len(code) > 4 else u'\\u{}'.format(code) for code in example]
    #         example = [code.decode('unicode-escape') for code in example]
    #         example = u''.join(example).replace(u'×', '')
    #
    #         expected.append(example.split(u'÷'))
    #         source.append(example.replace(u'÷', ''))
    #
    #         rule = rule.strip().strip(u'÷').strip()
    #         description.append(u'Row #{}. {}'.format(row + 1, rule))
    #
    #     max_len = len(sorted(expected, key=len, reverse=True)[0])
    #     expected = [e + [''] * (max_len - len(e)) for e in expected]
    #
    #     expected_tensor = tf.constant(expected, dtype=tf.string)
    #     self.assertIsInstance(result_tensor, tf.RaggedTensor)
    #     result_tensor = result_tensor.to_tensor(default_value='')
    #
    #     expected_value, result_value = self.evaluate(expected_tensor), self.evaluate(result_tensor)
    #
    #     for exp, res, desc in zip(expected_value, result_value, description):
    #         self.assertAllEqual(exp, res, desc)

    def test_split_stop(self):
        expected = tf.constant([
            ['.', 'word', ' ', '', ''],
            [' ', 'word', '.', '', ''],
            ['.', 'word', '.', '', ''],
            [' ', 'word', ' ', '', ''],
            [' ', 'word', '.', 'word', ' '],
            [' ', 'word', u'․', 'word', ' '],
            [' ', 'word', u'﹒', 'word', ' '],
            [' ', 'word', u'．', 'word', ' '],
        ], dtype=tf.string)
        result = split_words([
            # left border
            '.word ',

            # right border
            ' word.',

            # both borders
            '.word.',

            # no dot
            ' word ',

            # \u002E
            ' word.word ',

            # \u2024
            u' word․word ',

            # \uFE52
            u' word﹒word ',

            # \uFF0E
            u' word．word ',
        ], extended=True)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_split_extended_space(self):
        expected = tf.constant([
            ['word', '   ', 'word'],
        ], dtype=tf.string)
        result = split_words([
            'word   word',
        ], extended=True)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_split_alnum(self):
        expected = tf.constant([
            [' ', '0', '.', '5', 'word', ' '],
            [' ', '0', ',', '5', 'word', ' '],
        ], dtype=tf.string)
        result = split_words([
            ' 0.5word ', ' 0,5word ',
        ], extended=True)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_complex_extended(self):
        dangerous = u'\',.:;‘’'
        source = []
        for c in dangerous:
            source.append(u' {}00 '.format(c))  # before number
            source.append(u' {}zz '.format(c))  # before letter
            source.append(u' 00{}00 '.format(c))  # inside numbers
            source.append(u' zz{}zz '.format(c))  # inside letters
            source.append(u' 00{} '.format(c))  # after number
            source.append(u' zz{} '.format(c))  # after letter
        expected = tf.constant([
            [' ', u'\'', '00', ' ', ''], [' ', u'\'', 'zz', ' ', ''], [' ', '00', u'\'', '00', ' '],
            [' ', 'zz', u'\'', 'zz', ' '], [' ', '00', u'\'', ' ', ''], [' ', 'zz', u'\'', ' ', ''],

            [' ', u',', '00', ' ', ''], [' ', u',', 'zz', ' ', ''], [' ', '00', u',', '00', ' '],
            [' ', 'zz', u',', 'zz', ' '], [' ', '00', u',', ' ', ''], [' ', 'zz', u',', ' ', ''],

            [' ', u'.', '00', ' ', ''], [' ', u'.', 'zz', ' ', ''], [' ', '00', u'.', '00', ' '],
            [' ', 'zz', u'.', 'zz', ' '], [' ', '00', u'.', ' ', ''], [' ', 'zz', u'.', ' ', ''],

            [' ', u':', '00', ' ', ''], [' ', u':', 'zz', ' ', ''], [' ', '00', u':', '00', ' '],
            [' ', 'zz', u':', 'zz', ' '], [' ', '00', u':', ' ', ''], [' ', 'zz', u':', ' ', ''],

            [' ', u';', '00', ' ', ''], [' ', u';', 'zz', ' ', ''], [' ', '00', u';', '00', ' '],
            [' ', 'zz', u';', 'zz', ' '], [' ', '00', u';', ' ', ''], [' ', 'zz', u';', ' ', ''],

            [' ', u'‘', '00', ' ', ''], [' ', u'‘', 'zz', ' ', ''], [' ', '00', u'‘', '00', ' '],
            [' ', 'zz', u'‘', 'zz', ' '], [' ', '00', u'‘', ' ', ''], [' ', 'zz', u'‘', ' ', ''],

            [' ', u'’', '00', ' ', ''], [' ', u'’', 'zz', ' ', ''], [' ', '00', u'’', '00', ' '],
            [' ', 'zz', u'’', 'zz', ' '], [' ', '00', u'’', ' ', ''], [' ', 'zz', u'’', ' ', ''],
        ], dtype=tf.string)

        result = split_words(source, extended=True)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_skip(self):
        expected = tf.constant([['x', '!'], ['y!', '']], dtype=tf.string)
        result = split_words(['x!', 'y!'], skip=['y!'])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


if __name__ == "__main__":
    tf.test.main()
