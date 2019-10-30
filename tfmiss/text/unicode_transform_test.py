# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import test_util
from tfmiss.text.unicode_transform import lower_case, normalize_unicode, replace_regex, replace_string
from tfmiss.text.unicode_transform import title_case, upper_case, wrap_with, zero_digits


@test_util.run_all_in_graph_and_eager_modes
class LowerCaseTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = lower_case(source)

        self.assertEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = lower_case(source)
        result = tf.shape(result)

        result = self.evaluate(result)
        self.assertEqual([2, 3], result.tolist())

    def test_empty(self):
        result = lower_case('')

        result = self.evaluate(result)
        self.assertAllEqual(b'', result)

    def test_0d(self):
        result = lower_case('X')

        result = self.evaluate(result)
        self.assertAllEqual(b'x', result)

    def test_1d(self):
        result = lower_case(['X'])

        result = self.evaluate(result)
        self.assertAllEqual([b'x'], result)

    def test_2d(self):
        result = lower_case([['X']])

        result = self.evaluate(result)
        self.assertAllEqual([[b'x']], result)

    def test_ragged(self):
        source = tf.ragged.constant([['X', 'YY'], ['ZZZ ZZZ']])
        expected = tf.constant([['x', 'yy'], ['zzz zzz', '']])
        result = lower_case(source).to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = tf.convert_to_tensor(u'тест', dtype=tf.string)
        result = lower_case(u'ТеСт')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class NormalizeUnicodeTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = normalize_unicode(source, 'NFD')

        self.assertEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = normalize_unicode(source, 'NFD')
        result = tf.shape(result)

        result = self.evaluate(result)
        self.assertEqual([2, 3], result.tolist())

    def test_empty(self):
        result = normalize_unicode('', 'NFD')

        result = self.evaluate(result)
        self.assertAllEqual(b'', result)

    def test_0d(self):
        result = normalize_unicode('X', 'NFD')

        result = self.evaluate(result)
        self.assertAllEqual(b'X', result)

    def test_1d(self):
        result = normalize_unicode(['X'], 'NFD')

        result = self.evaluate(result)
        self.assertAllEqual([b'X'], result)

    def test_2d(self):
        result = normalize_unicode([['X']], 'NFD')

        result = self.evaluate(result)
        self.assertAllEqual([[b'X']], result)

    def test_ragged(self):
        source = tf.ragged.constant([['X', 'Y'], ['Z']])
        expected = tf.constant([['X', 'Y'], ['Z', '']])
        result = normalize_unicode(source, 'NFD').to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_n_f_d(self):
        expected = tf.convert_to_tensor(u'\u0041\u030A', dtype=tf.string)
        result = normalize_unicode(u'\u00C5', 'NFD')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_n_f_c(self):
        expected = tf.convert_to_tensor(u'\u00C5', dtype=tf.string)
        result = normalize_unicode(u'\u0041\u030A', 'NFC')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_n_f_k_d(self):
        expected = tf.convert_to_tensor(u'\u0031', dtype=tf.string)
        result = normalize_unicode(u'\u2460', 'NFKD')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_n_f_k_c(self):
        expected = tf.convert_to_tensor(u'\u1E69', dtype=tf.string)
        result = normalize_unicode(u'\u1E9B\u0323', 'NFKC')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_wrong_alg(self):
        if tf.executing_eagerly():
            with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, 'is not in the list of allowed values'):
                self.evaluate(normalize_unicode(u'', 'ABCD'))
        else:
            with self.assertRaisesRegexp(ValueError, 'string \'ABCD\' not in'):
                self.evaluate(normalize_unicode(u'', 'ABCD'))


@test_util.run_all_in_graph_and_eager_modes
class ReplaceRegexTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = replace_regex(source, ['\\d'], ['0'])

        self.assertEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = replace_regex(source, ['\\d'], ['0'])
        result = tf.shape(result)

        result = self.evaluate(result)
        self.assertEqual([2, 3], result.tolist())

    def test_empty(self):
        result = replace_regex('', ['\\d'], ['0'])

        result = self.evaluate(result)
        self.assertAllEqual(b'', result)

    def test_empty_needle(self):
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, 'Items of "pattern" could not be empty'):
            result = replace_regex('<test>', [''], ['>'])
            result = self.evaluate(result)
            self.assertAllEqual(b'test', result)

    def test_empty_haystack(self):
        result = replace_regex('<test>', ['(<)'], [''])

        result = self.evaluate(result)
        self.assertAllEqual(b'test>', result)

    def test_0d(self):
        result = replace_regex('1test2', ['\\d'], ['0'])

        result = self.evaluate(result)
        self.assertAllEqual(b'0test0', result)

    def test_1d(self):
        result = replace_regex(['1test2'], ['\\d'], ['0'])

        result = self.evaluate(result)
        self.assertAllEqual([b'0test0'], result)

    def test_2d(self):
        result = replace_regex([['1test2']], ['\\d'], ['0'])

        result = self.evaluate(result)
        self.assertAllEqual([[b'0test0']], result)

    def test_ragged(self):
        source = tf.ragged.constant([['1test', 'test2'], ['test']])
        expected = tf.constant([['0test', 'test0'], ['test', '']])
        result = replace_regex(source, ['\\d'], ['0']).to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = u'_ на _ он же _-0 _ _̈_ _'
        result = replace_regex(
            u'тест на юникод он же utf-8 плюс совмещённый символ',
            [u'\\pL{3,}', u'\\d'],
            ['_', u'0']
        )
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class ReplaceStringTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = replace_string(source, ['<'], ['>'])

        self.assertEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = replace_string(source, ['<'], ['>'])
        result = tf.shape(result)

        result = self.evaluate(result)
        self.assertEqual([2, 3], result.tolist())

    def test_empty(self):
        result = replace_string('', ['<'], ['>'])

        result = self.evaluate(result)
        self.assertAllEqual(b'', result)

    def test_empty_needle(self):
        with self.assertRaisesRegexp(tf.errors.InvalidArgumentError, 'Items of "needle" could not be empty'):
            result = replace_string('<test>', [''], ['>'])
            result = self.evaluate(result)
            self.assertAllEqual(b'test', result)

    def test_empty_haystack(self):
        result = replace_string('<test>', ['<'], [''])

        result = self.evaluate(result)
        self.assertAllEqual(b'test>', result)

    def test_0d(self):
        result = replace_string('<test>', ['<'], ['>'])

        result = self.evaluate(result)
        self.assertAllEqual(b'>test>', result)

    def test_1d(self):
        result = replace_string(['<test>'], ['<'], ['>'])

        result = self.evaluate(result)
        self.assertAllEqual([b'>test>'], result)

    def test_2d(self):
        result = replace_string([['<test>']], ['<'], ['>'])

        result = self.evaluate(result)
        self.assertAllEqual([[b'>test>']], result)

    def test_ragged(self):
        source = tf.ragged.constant([['<test', 'test>'], ['test']])
        expected = tf.constant([['>test', 'test>'], ['test', '']])
        result = replace_string(source, ['<'], ['>']).to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = u'тостовый'
        result = replace_string(u'т́ест', [u'́', u'е', u'ост'], ['', u'о', u'остовый'])
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class TitleCaseTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = title_case(source)

        self.assertEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = title_case(source)
        result = tf.shape(result)

        result = self.evaluate(result)
        self.assertEqual([2, 3], result.tolist())

    def test_empty(self):
        result = title_case('')

        result = self.evaluate(result)
        self.assertAllEqual(b'', result)

    def test_0d(self):
        result = title_case('x')

        result = self.evaluate(result)
        self.assertAllEqual(b'X', result)

    def test_1d(self):
        result = title_case(['x'])

        result = self.evaluate(result)
        self.assertAllEqual([b'X'], result)

    def test_2d(self):
        result = title_case([['x']])

        result = self.evaluate(result)
        self.assertAllEqual([[b'X']], result)

    def test_ragged(self):
        source = tf.ragged.constant([['x', 'yy'], ['zzz zzz']])
        expected = tf.constant([['X', 'Yy'], ['Zzz Zzz', '']])
        result = title_case(source).to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_latin(self):
        result = title_case('TeSt')

        result = self.evaluate(result)
        self.assertAllEqual(b'Test', result)

    def test_unicode(self):
        expected = [u'Тест', u'\u01C5']
        result = title_case([u'Тест', u'\u01C6'])
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class UpperCaseUnicodeTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = upper_case(source)

        self.assertEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = upper_case(source)
        result = tf.shape(result)

        result = self.evaluate(result)
        self.assertEqual([2, 3], result.tolist())

    def test_empty(self):
        result = upper_case('')

        result = self.evaluate(result)
        self.assertAllEqual(b'', result)

    def test_0d(self):
        result = upper_case('x')

        result = self.evaluate(result)
        self.assertAllEqual(b'X', result)

    def test_1d(self):
        result = upper_case(['x'])

        result = self.evaluate(result)
        self.assertAllEqual([b'X'], result)

    def test_2d(self):
        result = upper_case([['x']])

        result = self.evaluate(result)
        self.assertAllEqual([[b'X']], result)

    def test_ragged(self):
        source = tf.ragged.constant([['x', 'yy'], ['zzz zzz']])
        expected = tf.constant([['X', 'YY'], ['ZZZ ZZZ', '']])
        result = upper_case(source).to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_latin(self):
        result = upper_case('TeSt')

        result = self.evaluate(result)
        self.assertAllEqual(b'TEST', result)

    def test_unicode(self):
        expected = u'ТЕСТ'
        result = upper_case(u'ТеСт')
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class WrapWithTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = wrap_with(source, '<', '>')

        self.assertEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = wrap_with(source, '<', '>')
        result = tf.shape(result)

        result = self.evaluate(result)
        self.assertEqual([2, 3], result.tolist())

    def test_empty(self):
        result = wrap_with('', '<', '>')

        result = self.evaluate(result)
        self.assertAllEqual(b'<>', result)

    def test_empty_borders(self):
        result = wrap_with('test', '', '')

        result = self.evaluate(result)
        self.assertAllEqual(b'test', result)

    def test_0d(self):
        result = wrap_with('X', '<', '>')

        result = self.evaluate(result)
        self.assertAllEqual(b'<X>', result)

    def test_1d(self):
        result = wrap_with(['X'], '<', '>')

        result = self.evaluate(result)
        self.assertAllEqual([b'<X>'], result)

    def test_2d(self):
        result = wrap_with([['X']], '<', '>')

        result = self.evaluate(result)
        self.assertAllEqual([[b'<X>']], result)

    def test_ragged(self):
        source = tf.ragged.constant([['X', 'X'], ['X']])
        expected = tf.constant([['<X>', '<X>'], ['<X>', '']])
        result = wrap_with(source, '<', '>').to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_unicode(self):
        expected = u'надо'
        result = wrap_with(u'ад', u'н', u'о')
        expected = tf.convert_to_tensor(expected, dtype=tf.string)

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)


@test_util.run_all_in_graph_and_eager_modes
class ZeroDigitsTest(tf.test.TestCase):
    def test_inference_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = zero_digits(source)

        self.assertEqual([2, 3], result.shape.as_list())

    def test_actual_shape(self):
        source = [
            ['1', '2', '3'],
            ['4', '5', '6'],
        ]
        result = zero_digits(source)
        result = tf.shape(result)

        result = self.evaluate(result)
        self.assertEqual([2, 3], result.tolist())

    def test_empty(self):
        result = zero_digits('')

        result = self.evaluate(result)
        self.assertEqual(b'', result)

    def test_0d(self):
        result = zero_digits('7')

        result = self.evaluate(result)
        self.assertEqual(b'0', result)

    def test_1d(self):
        result = zero_digits(['7'])

        result = self.evaluate(result)
        self.assertEqual([b'0'], result)

    def test_2d(self):
        result = zero_digits([['7']])

        result = self.evaluate(result)
        self.assertEqual([[b'0']], result)

    def test_ragged(self):
        source = tf.ragged.constant([['x1', '2x'], ['3x4']])
        expected = tf.constant([['x0', '0x'], ['0x0', '']])
        result = zero_digits(source).to_tensor(default_value='')

        expected, result = self.evaluate([expected, result])
        self.assertAllEqual(expected, result)

    def test_mixed_unicode(self):
        result = zero_digits(u'P.1, АБ1, ЯК12x, м²')
        expected = tf.convert_to_tensor(u'P.0, АБ0, ЯК00x, м²', dtype=tf.string)

        expected, result = self.evaluate([expected, result])
        self.assertEqual(expected, result)


if __name__ == "__main__":
    tf.test.main()
