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
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_0d(self):
        expected = tf.constant(['x', 'y'], dtype=tf.string)
        result = char_ngrams('xy', 1, 1, itself='NEVER')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

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
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([], result.tolist())

    def test_as_is_below(self):
        result = char_ngrams('1234', 2, 3, itself='ASIS')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234'], result.tolist())

    def test_as_is_inside(self):
        result = char_ngrams('123', 2, 3, itself='ASIS')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'123'], result.tolist())

    def test_as_is_above(self):
        result = char_ngrams('123', 4, 5, itself='ASIS')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([], result.tolist())

    def test_never_below(self):
        result = char_ngrams('1234', 2, 3, itself='NEVER')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234'], result.tolist())

    def test_never_inside(self):
        result = char_ngrams('123', 2, 3, itself='NEVER')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23'], result.tolist())

    def test_never_above(self):
        result = char_ngrams('123', 4, 5, itself='NEVER')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([], result.tolist())

    def test_always_below(self):
        result = char_ngrams('1234', 2, 3, itself='ALWAYS')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234', b'1234'], result.tolist())

    def test_always_inside(self):
        result = char_ngrams('123', 2, 3, itself='ALWAYS')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'123'], result.tolist())

    def test_always_above(self):
        result = char_ngrams('123', 4, 5, itself='ALWAYS')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'123'], result.tolist())

    def test_alone_below(self):
        result = char_ngrams('1234', 2, 3, itself='ALONE')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234'], result.tolist())

    def test_alone_inside(self):
        result = char_ngrams('123', 2, 3, itself='ALONE')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23'], result.tolist())

    def test_alone_above(self):
        result = char_ngrams('123', 4, 5, itself='ALONE')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'123'], result.tolist())

    def test_skip(self):
        expected = tf.constant([[['x', 'y'], ['[UNK]', '']]], dtype=tf.string)
        result = char_ngrams([['xy', '[UNK]']], 1, 1, itself='ASIS', skip=['[UNK]'], name='abc')
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
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_0d(self):
        expected = tf.constant(['x', 'y'], dtype=tf.string)

        result = split_chars('xy')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

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
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_restore(self):
        source = tf.constant('Hey\n\tthere\t«word», !!!')

        splitted = split_chars(source)
        self.assertTrue(tf.is_tensor(splitted))
        self.assertNotIsInstance(splitted, tf.RaggedTensor)
        restored = tf.strings.reduce_join(splitted)

        source, restored = self.evaluate(source), self.evaluate(restored)
        self.assertAllEqual(source, restored)

    def test_skip(self):
        expected = tf.constant([[['x', 'y'], ['zz', '']]], dtype=tf.string)

        result = split_chars([['xy', 'zz']], skip=['zz'])
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
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_0d(self):
        expected = tf.constant(['x', '!'], dtype=tf.string)
        result = split_words('x!')
        self.assertTrue(tf.is_tensor(result))
        self.assertNotIsInstance(result, tf.RaggedTensor)

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
        self.assertTrue(tf.is_tensor(splitted))
        self.assertNotIsInstance(splitted, tf.RaggedTensor)
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

    def test_split_extended_double(self):
        expected = tf.constant([
            [' ', 'word', '.', 'word', '.', 'word', '.', ' '],
            [' ', 'word', '1', 'word', '1', 'word', '1', ' '],
            [' ', '.', '1', '.', '1', '.', '1', ' '],
        ], dtype=tf.string)
        result = split_words([
            ' word.word.word. ',  # wb 6, 7
            ' word1word1word1 ',  # wb 9, 10
            ' .1.1.1 ',  # wb 11, 12
        ], extended=True)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_complex_extended_case(self):
        expected = tf.constant(['Word', '.', 'W', '.', 'O', '.', ' ', 'rd', '.'])
        result = split_words('Word.W.O. rd.', extended=True)
        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_complex_extended_space(self):
        spaces = [
            '\t', '\n', '\x0b', '\x0c', '\r', '\x1c', '\x1d', '\x1e', '\x1f', ' ', '\x85', '\xa0', '\u1680', '\u2000',
            '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008', '\u2009', '\u200a',
            '\u2028', '\u2029', '\u200b', '\u202f', '\u205f', '\u2060', '\u2061', '\u2800', '\u3000', '\ufeff']

        source = [' {}W{}W{} {} '.format(s, s, s, s) for s in spaces]
        expected = tf.constant([
            [' ', '\t', 'W', '\t', 'W', '\t', ' ', '\t', ' '],
            [' ', '\n', 'W', '\n', 'W', '\n', ' ', '\n', ' '],
            [' ', '\x0b', 'W', '\x0b', 'W', '\x0b', ' ', '\x0b', ' '],
            [' ', '\x0c', 'W', '\x0c', 'W', '\x0c', ' ', '\x0c', ' '],
            [' ', '\r', 'W', '\r', 'W', '\r', ' ', '\r', ' '],
            [' ', '\x1c', 'W', '\x1c', 'W', '\x1c', ' ', '\x1c', ' '],
            [' ', '\x1d', 'W', '\x1d', 'W', '\x1d', ' ', '\x1d', ' '],
            [' ', '\x1e', 'W', '\x1e', 'W', '\x1e', ' ', '\x1e', ' '],
            [' ', '\x1f', 'W', '\x1f', 'W', '\x1f', ' ', '\x1f', ' '],
            ['  ', 'W', ' ', 'W', '    ', '', '', '', ''],
            [' ', '\x85', 'W', '\x85', 'W', '\x85', ' ', '\x85', ' '],
            [' ', '\xa0', 'W', '\xa0', 'W', '\xa0', ' ', '\xa0', ' '],
            [' \u1680', 'W', '\u1680', 'W', '\u1680 \u1680 ', '', '', '', ''],
            [' \u2000', 'W', '\u2000', 'W', '\u2000 \u2000 ', '', '', '', ''],
            [' \u2001', 'W', '\u2001', 'W', '\u2001 \u2001 ', '', '', '', ''],
            [' \u2002', 'W', '\u2002', 'W', '\u2002 \u2002 ', '', '', '', ''],
            [' \u2003', 'W', '\u2003', 'W', '\u2003 \u2003 ', '', '', '', ''],
            [' \u2004', 'W', '\u2004', 'W', '\u2004 \u2004 ', '', '', '', ''],
            [' \u2005', 'W', '\u2005', 'W', '\u2005 \u2005 ', '', '', '', ''],
            [' \u2006', 'W', '\u2006', 'W', '\u2006 \u2006 ', '', '', '', ''],
            [' ', '\u2007', 'W', '\u2007', 'W', '\u2007', ' ', '\u2007', ' '],
            [' \u2008', 'W', '\u2008', 'W', '\u2008 \u2008 ', '', '', '', ''],
            [' \u2009', 'W', '\u2009', 'W', '\u2009 \u2009 ', '', '', '', ''],
            [' \u200a', 'W', '\u200a', 'W', '\u200a \u200a ', '', '', '', ''],
            [' ', '\u2028', 'W', '\u2028', 'W', '\u2028', ' ', '\u2028', ' '],
            [' ', '\u2029', 'W', '\u2029', 'W', '\u2029', ' ', '\u2029', ' '],
            [' ', '\u200b', 'W', '\u200b', 'W', '\u200b', ' ', '\u200b', ' '],
            [' ', '\u202f', 'W', '\u202f', 'W', '\u202f', ' ', '\u202f', ' '],
            [' \u205f', 'W', '\u205f', 'W', '\u205f \u205f ', '', '', '', ''],
            [' \u2060', 'W', '\u2060', 'W', '\u2060', ' \u2060', ' ', '', ''],
            [' \u2061', 'W', '\u2061', 'W', '\u2061', ' \u2061', ' ', '', ''],
            [' ', '\u2800', 'W', '\u2800', 'W', '\u2800', ' ', '\u2800', ' '],
            [' \u3000', 'W', '\u3000', 'W', '\u3000 \u3000 ', '', '', '', ''],
            [' \ufeff', 'W', '\ufeff', 'W', '\ufeff', ' \ufeff', ' ', '', '']
        ])

        result = split_words(source, extended=True)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_extended_lr(self):
        expected = tf.constant([
            ['A', ' ', '\u200e', 'B'], ['A\u200eB', '', '', ''],
            ['A', ' ', '\xad', 'B'], ['A\xadB', '', '', ''],
            ['A', ' ', '\ufe0f', 'B'], ['A\ufe0fB', '', '', ''],
        ], dtype=tf.string)
        result = split_words(['A \u200eB', 'A\u200eB', 'A \xadB', 'A\xadB', 'A \ufe0fB', 'A\ufe0fB'], extended=True)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_skip(self):
        expected = tf.constant([[['x', '!'], ['y!', '']]], dtype=tf.string)
        result = split_words([['x!', 'y!']], skip=['y!'])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


if __name__ == "__main__":
    tf.test.main()
