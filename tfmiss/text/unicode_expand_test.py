# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.text.unicode_expand import char_ngrams, split_chars, split_words


@test_util.run_all_in_graph_and_eager_modes
class CharNgramsTest(tf.test.TestCase):
    def testInferenceShape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = char_ngrams(source, 1, 1, itself='ALWAYS')

        self.assertEqual([2, None, None], result.shape.as_list())

    def testActualShape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = char_ngrams(source, 1, 1, itself='ALWAYS')
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        result = self.evaluate(result)
        self.assertAllEqual((2, 3, 1), result.shape)

    def testEmpty(self):
        expected = tf.constant([], dtype=tf.string)
        result = char_ngrams('', 1, 1, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test0D(self):
        expected = tf.constant(['x', 'y'], dtype=tf.string)
        result = char_ngrams('xy', 1, 1, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test1D(self):
        expected = tf.constant([['x', 'y']], dtype=tf.string)
        result = char_ngrams(['xy'], 1, 1, itself='ASIS')
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def testDefault2D(self):
        expected = tf.constant([[['x', 'y'], ['x', '']]], dtype=tf.string)
        result = char_ngrams([['xy', 'x']], 1, 1, itself='ASIS')
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def testRagged(self):
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

    def testNone(self):
        result = char_ngrams('123', 4, 5, itself='ASIS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([], result.tolist())

    def testAsIsBelow(self):
        result = char_ngrams('1234', 2, 3, itself='ASIS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234'], result.tolist())

    def testAsIsInside(self):
        result = char_ngrams('123', 2, 3, itself='ASIS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'123'], result.tolist())

    def testAsIsAbove(self):
        result = char_ngrams('123', 4, 5, itself='ASIS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([], result.tolist())

    def testNeverBelow(self):
        result = char_ngrams('1234', 2, 3, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234'], result.tolist())

    def testNeverInside(self):
        result = char_ngrams('123', 2, 3, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23'], result.tolist())

    def testNeverAbove(self):
        result = char_ngrams('123', 4, 5, itself='NEVER')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([], result.tolist())

    def testAlwaysBelow(self):
        result = char_ngrams('1234', 2, 3, itself='ALWAYS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234', b'1234'], result.tolist())

    def testAlwaysInside(self):
        result = char_ngrams('123', 2, 3, itself='ALWAYS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'123'], result.tolist())

    def testAlwaysAbove(self):
        result = char_ngrams('123', 4, 5, itself='ALWAYS')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'123'], result.tolist())

    def testAloneBelow(self):
        result = char_ngrams('1234', 2, 3, itself='ALONE')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23', b'34', b'123', b'234'], result.tolist())

    def testAloneInside(self):
        result = char_ngrams('123', 2, 3, itself='ALONE')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'12', b'23'], result.tolist())

    def testAloneAbove(self):
        result = char_ngrams('123', 4, 5, itself='ALONE')
        self.assertIsInstance(result, tf.Tensor)

        result = self.evaluate(result)
        self.assertAllEqual([b'123'], result.tolist())


@test_util.run_all_in_graph_and_eager_modes
class LowerCaseTest(tf.test.TestCase):
    def testInferenceShape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = split_chars(source)

        self.assertEqual([2, None, None], result.shape.as_list())

    def testActualShape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = split_chars(source)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        result = self.evaluate(result)
        self.assertEqual((2, 3, 1), result.shape)

    def testEmpty(self):
        expected = tf.constant([], dtype=tf.string)
        result = split_chars('')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test0D(self):
        expected = tf.constant(['x', 'y'], dtype=tf.string)

        result = split_chars('xy')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test1D(self):
        expected = tf.constant([['x', 'y']], dtype=tf.string)

        result = split_chars(['xy'])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test2D(self):
        expected = tf.constant([[['x', 'y']]], dtype=tf.string)

        result = split_chars([['xy']])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def testRagged(self):
        expected = tf.constant([[['a', 'b', ''], ['c', ' ', 'd']], [['e', '', ''], ['', '', '']]])

        result = split_chars(tf.ragged.constant([['ab', 'c d'], ['e']]))
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def testUnicode(self):
        expected = tf.constant([u'ё', u' ', u'е', u'̈', u'2', u'⁵'], dtype=tf.string)

        result = split_chars(u'ё ё2⁵')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def testRestore(self):
        source = tf.constant('Hey\n\tthere\t«word», !!!')

        splitted = split_chars(source)
        self.assertIsInstance(splitted, tf.Tensor)
        restored = tf.strings.reduce_join(splitted)

        source, restored = self.evaluate(source), self.evaluate(restored)
        self.assertAllEqual(source, restored)


@test_util.run_all_in_graph_and_eager_modes
class SplitWordsTest(tf.test.TestCase):
    def testInferenceShape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = split_words(source)

        self.assertEqual([2, None, None], result.shape.as_list())

    def testActualShape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = split_words(source)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        result = self.evaluate(result)
        self.assertEqual((2, 3, 1), result.shape)

    def testEmpty(self):
        expected = tf.constant([''], dtype=tf.string)
        result = split_words('')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test0D(self):
        expected = tf.constant(['x', '!'], dtype=tf.string)
        result = split_words('x!')
        self.assertIsInstance(result, tf.Tensor)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test1D(self):
        expected = tf.constant([['x', '!']], dtype=tf.string)
        result = split_words(['x!'])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test2D(self):
        expected = tf.constant([[['x', '!']]], dtype=tf.string)
        result = split_words([['x!']])
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def testRagged(self):
        expected = tf.constant([[['ab', '', ''], ['c', ' ', 'd']], [['e', '', ''], ['', '', '']]])

        result = split_words(tf.ragged.constant([['ab', 'c d'], ['e']]))
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def testRestore(self):
        source = tf.constant('Hey\n\tthere\t«word», !!!')

        splitted = split_words(source)
        self.assertIsInstance(splitted, tf.Tensor)
        restored = tf.strings.reduce_join(splitted)

        source, restored = self.evaluate(source), self.evaluate(restored)
        self.assertAllEqual(source, restored)

    def testWrapped(self):
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

    def testWordPunkt(self):
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

    def testComplexWord(self):
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

    # def testIcuWordBreak(self):
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

    def testSplitStop(self):
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

    def testComplexStop(self):
        dangerous = u'\',.:;‘’\u00AD'
        source = []
        for c in dangerous:
            source.append(u' {}00 '.format(c))  # before number
            source.append(u' {}zz '.format(c))  # before letter
            source.append(u' 00{}00 '.format(c))  # inside numbers
            source.append(u' zz{}zz '.format(c))  # inside letters
            source.append(u' 00{} '.format(c))  # after number
            source.append(u' zz{} '.format(c))  # after letter
        expected = tf.constant([
            [' ', u'\'', '00', ' ', ''], [' ', u'\'', 'zz', ' ', ''], [' ', u'00\'00', ' ', '', ''],
            [' ', u'zz\'zz', ' ', '', ''], [' ', '00', u'\'', ' ', ''], [' ', 'zz', u'\'', ' ', ''],
            
            [' ', u',', '00', ' ', ''], [' ', u',', 'zz', ' ', ''], [' ', u'00,00', ' ', '', ''],
            [' ', 'zz', u',', 'zz', ' '], [' ', '00', u',', ' ', ''], [' ', 'zz', u',', ' ', ''],
            
            [' ', u'.', '00', ' ', ''], [' ', u'.', 'zz', ' ', ''], [' ', u'00.00', ' ', '', ''],
            [' ', u'zz.zz', ' ', '', ''], [' ', '00', u'.', ' ', ''], [' ', 'zz', u'.', ' ', ''],
            
            [' ', u':', '00', ' ', ''], [' ', u':', 'zz', ' ', ''], [' ', '00', u':', '00', ' '],
            [' ', u'zz:zz', ' ', '', ''], [' ', '00', u':', ' ', ''], [' ', 'zz', u':', ' ', ''],
            
            [' ', u';', '00', ' ', ''], [' ', u';', 'zz', ' ', ''], [' ', u'00;00', ' ', '', ''],
            [' ', 'zz', u';', 'zz', ' '], [' ', '00', u';', ' ', ''], [' ', 'zz', u';', ' ', ''],
            
            [' ', u'‘', '00', ' ', ''], [' ', u'‘', 'zz', ' ', ''], [' ', u'00‘00', ' ', '', ''],
            [' ', u'zz‘zz', ' ', '', ''], [' ', '00', u'‘', ' ', ''], [' ', 'zz', u'‘', ' ', ''],
            
            [' ', u'’', '00', ' ', ''], [' ', u'’', 'zz', ' ', ''], [' ', u'00’00', ' ', '', ''], 
            [' ', u'zz’zz', ' ', '', ''], [' ', '00', u'’', ' ', ''], [' ', 'zz', u'’', ' ', ''],
            
            [u' \xad', '00', ' ', '', ''], [u' \xad', 'zz', ' ', '', ''], [' ', u'00\xad00', ' ', '', ''], 
            [' ', u'zz\xadzz', ' ', '', ''], [' ', u'00\xad', ' ', '', ''], [' ', u'zz\xad', ' ', '', ''], 
        ], dtype=tf.string)

        result = split_words(source, extended=False)
        self.assertIsInstance(result, tf.RaggedTensor)
        result = result.to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        print([[w.decode('utf-8') for w in r] for r in result])
        self.assertAllEqual(expected, result)


if __name__ == "__main__":
    tf.test.main()
