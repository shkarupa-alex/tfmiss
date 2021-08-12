from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter
from keras import keras_parameterized, testing_utils
from tfmiss.keras.layers.wordvec import WordEmbedding, CharNgramEmbedding, CharBpeEmbedding, CharCnnEmbedding, Highway


@keras_parameterized.run_all_keras_modes
class WordEmbeddingTest(keras_parameterized.TestCase):
    def test_layer(self):
        data = np.array(['[UNK]', 'the', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'])
        vocab = ['the', 'fox', 'jumps']

        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': False, 'max_len': None, 'reserved_words': None, 'embed_type': 'dense_auto',
                    'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'normalize_unicode': None, 'lower_case': False,
                    'zero_digits': False, 'max_len': None, 'reserved_words': None, 'embed_type': 'dense_auto',
                    'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': True,
                    'zero_digits': False, 'max_len': None, 'reserved_words': None, 'embed_type': 'dense_auto',
                    'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': True, 'max_len': 4, 'reserved_words': None, 'embed_type': 'dense_auto',
                    'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': False, 'max_len': None, 'reserved_words': None, 'embed_type': 'dense_auto',
                    'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': ['[UNK]'] + vocab, 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': False, 'max_len': None, 'reserved_words': None, 'embed_type': 'dense_auto',
                    'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': vocab + ['[UNK]'], 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': False, 'max_len': None, 'reserved_words': None, 'embed_type': 'dense_auto',
                    'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': False, 'max_len': None, 'reserved_words': ['[UNK]', '~TesT~'],
                    'embed_type': 'dense_auto', 'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': False, 'max_len': None, 'reserved_words': None, 'embed_type': 'dense_cpu',
                    'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            WordEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': False, 'max_len': None, 'reserved_words': None, 'embed_type': 'adapt',
                    'adapt_cutoff': [2], 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )

    def test_reserved_words(self):
        layer = WordEmbedding([], 4)
        self.assertListEqual(layer._reserved_words, [layer.UNK_MARK])

        layer = WordEmbedding([], 4, reserved_words=['~TesT~'])
        self.assertListEqual(layer._reserved_words, [layer.UNK_MARK, '~TesT~'])

    def test_vocab(self):
        counts = Counter({'the': 4, 'fox': 2, 'jumps': 2, '\u1E9B\u0323': 2, 'dog': 1})
        adapts = [('the', 4), ('fox', 2), ('jumps', 2), ('\u1E69', 2), ('dog', 1)]
        layer = WordEmbedding([], 4)
        self.assertListEqual(layer.vocab(counts).most_common(), adapts)

        vocab = ['the', 'fox', 'jumps', '\u1E69']
        layer = WordEmbedding(vocab, 4)
        self.assertListEqual(layer._vocabulary, [layer.UNK_MARK] + vocab)

        layer = WordEmbedding(vocab, 4, reserved_words=['~TesT~'])
        self.assertListEqual(layer._vocabulary, [layer.UNK_MARK, '~TesT~'] + vocab)

        layer = WordEmbedding(vocab + ['~TesT~'], 4, reserved_words=['~TesT~'])
        self.assertListEqual(layer._vocabulary, [layer.UNK_MARK, '~TesT~'] + vocab)

    def test_adapt(self):
        data = ['[UNK]', 'the', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']

        layer = WordEmbedding([], 4)
        adapted = layer.adapt(data)
        adapted = self.evaluate(adapted)
        adapted = np.char.decode(adapted.astype('S'), 'utf-8').tolist()
        self.assertListEqual(adapted, data)

    def test_adapt_all_on_odd(self):
        data = ['[UNK]', '\u0041\u030A', 'Fox', 'jump99', 'abcdefghik', 'abcdefghikl']
        expected = ['[UNK]', '\u00E5', 'fox', 'jump00', 'abc\uFFFDik', 'abc\uFFFDkl']

        layer = WordEmbedding([], 4, normalize_unicode='NFC', lower_case=True, zero_digits=True, max_len=6)
        adapted = layer.adapt(data)
        adapted = self.evaluate(adapted)
        adapted = np.char.decode(adapted.astype('S'), 'utf-8').tolist()
        self.assertListEqual(adapted, expected)

    def test_adapt_all_on_even(self):
        data = ['[UNK]', '\u0041\u030A', 'Fox', 'jum99', 'abcdefghik', 'abcdefghikl']
        expected = ['[UNK]', '\u00E5', 'fox', 'jum00', 'ab\uFFFDik', 'ab\uFFFDkl']

        layer = WordEmbedding([], 4, normalize_unicode='NFC', lower_case=True, zero_digits=True, max_len=5)
        adapted = layer.adapt(data)
        adapted = self.evaluate(adapted)
        adapted = np.char.decode(adapted.astype('S'), 'utf-8').tolist()
        self.assertListEqual(adapted, expected)


@keras_parameterized.run_all_keras_modes
class CharNgramEmbeddingTest(keras_parameterized.TestCase):
    def test_layer(self):
        data = np.array(['[UNK]', 'the', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'])
        vocab = ['ve', 'do', 'r>', 'er', '<l', 'mp', 'ov', 'la', 'e>', '<d', '<o', 'ju', '<f', 'x>', 'az', '<t', 'y>',
                 'ps', '<j', 'he', 'um', 'fo', 's>', 'th', 'zy', 'ox', 'og', 'g>']

        testing_utils.layer_test(
            CharNgramEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': False, 'max_len': None, 'minn': 2, 'maxn': 5, 'itself': 'always',
                    'reduction': 'mean', 'reserved_words': None, 'embed_type': 'dense_auto', 'adapt_cutoff': None,
                    'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )

    def test_reserved_words(self):
        layer = CharNgramEmbedding([], 4)
        self.assertListEqual(layer._reserved_words, [layer.UNK_MARK])

        layer = CharNgramEmbedding([], 4, reserved_words=['~TesT~'])
        self.assertListEqual(layer._reserved_words, [layer.UNK_MARK, '~TesT~'])

    def test_vocab(self):
        counts = Counter({'the': 4, 'fox': 2, 'jumps': 2, '\u1E9B\u0323': 2, 'dog': 1})
        adapts = [
            ('<th', 4), ('the', 4), ('he>', 4), ('<the', 4), ('the>', 4), ('<the>', 4), ('<fo', 2), ('fox', 2),
            ('ox>', 2), ('<fox', 2), ('fox>', 2), ('<fox>', 2), ('<ju', 2), ('jum', 2), ('ump', 2), ('mps', 2),
            ('ps>', 2), ('<jum', 2), ('jump', 2), ('umps', 2), ('mps>', 2), ('<jump', 2), ('jumps', 2), ('umps>', 2),
            ('<jumps>', 2), ('<\u1E69>', 2), ('<do', 1), ('dog', 1), ('og>', 1), ('<dog', 1), ('dog>', 1), ('<dog>', 1)]
        layer = CharNgramEmbedding([], 4)
        self.assertListEqual(layer.vocab(counts).most_common(), adapts)

        vocab = ['<th', '<the', '<the>', 'he>', 'the', 'the>', '<fo', '<fox', '<fox>', '<ju', '<jum', '<jump',
                 '<jumps>', '<\u1E69>', 'fox', 'fox>', 'jum', 'jump', 'jumps', 'mps', 'mps>', 'ox>', 'ps>', 'ump',
                 'umps', 'umps>']
        layer = CharNgramEmbedding(vocab, 4)
        self.assertListEqual(layer._vocabulary, [layer.UNK_MARK] + vocab)

        layer = CharNgramEmbedding(vocab, 4, reserved_words=['~TesT~'])
        self.assertListEqual(layer._vocabulary, [layer.UNK_MARK, '~TesT~'] + vocab)

    def test_adapt(self):
        data = ['[UNK]', 'the', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
        expected = [
            ['[UNK]'],
            ['<th', 'the', 'he>', '<the', 'the>', '<the>'],
            ['<fo', 'fox', 'ox>', '<fox', 'fox>', '<fox>'],
            ['<ju', 'jum', 'ump', 'mps', 'ps>', '<jum', 'jump', 'umps', 'mps>', '<jump', 'jumps', 'umps>', '<jumps>'],
            ['<ov', 'ove', 'ver', 'er>', '<ove', 'over', 'ver>', '<over', 'over>', '<over>'],
            ['<th', 'the', 'he>', '<the', 'the>', '<the>'],
            ['<la', 'laz', 'azy', 'zy>', '<laz', 'lazy', 'azy>', '<lazy', 'lazy>', '<lazy>'],
            ['<do', 'dog', 'og>', '<dog', 'dog>', '<dog>']]

        layer = CharNgramEmbedding([], 4)
        adapted = layer.adapt(data)
        adapted = self.evaluate(adapted).to_list()
        adapted = [[w_.decode('utf-8') for w_ in w] for w in adapted]
        self.assertListEqual(adapted, expected)


@keras_parameterized.run_all_keras_modes
class CharBpeEmbeddingTest(keras_parameterized.TestCase):
    def test_layer(self):
        data = np.array(['[UNK]', 'the', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'])
        vocab = ['the', '##o', 'f', '##u', '##m', '##s', 'o', '##v', '##er', 'l', '##a', '##y', 'd', '##g']

        testing_utils.layer_test(
            CharBpeEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'reduction': 'mean', 'reserved_words': None,
                    'embed_type': 'dense_auto', 'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )

    def test_reserved_words(self):
        layer = CharBpeEmbedding([], 4)
        self.assertListEqual(layer._reserved_words, [layer.UNK_MARK, layer.UNK_CHAR])

        layer = CharBpeEmbedding([], 4, reserved_words=['~TesT~'])
        self.assertListEqual(layer._reserved_words, [layer.UNK_MARK, layer.UNK_CHAR, '~TesT~'])

    def test_vocab(self):
        counts = Counter({'the': 4, 'fox': 2, 'jumps': 2, '\u1E9B\u0323': 2, 'dog': 1})
        adapts = [
            ('the', 4), ('##o', 3), ('f', 2), ('##x', 2), ('j', 2), ('##u', 2), ('##m', 2), ('##p', 2), ('##s', 2),
            ('[UNK]', 2), ('d', 1), ('##g', 1)]
        layer = CharBpeEmbedding([], 4, vocab_size=4)
        self.assertListEqual(layer.vocab(counts).most_common(), adapts)

        vocab = ['the', '##o', 'f', '##u', '##m', '##s', 'o', '##v', '##er', 'l', '##a', '##y', 'd', '##g']
        layer = CharBpeEmbedding(vocab, 4)
        self.assertListEqual(layer._vocabulary, [layer.UNK_MARK, layer.UNK_CHAR] + vocab)

        layer = CharBpeEmbedding(vocab, 4, reserved_words=['~TesT~'])
        self.assertListEqual(layer._vocabulary, [layer.UNK_MARK, layer.UNK_CHAR, '~TesT~'] + vocab)

    def test_adapt(self):
        data = ['[UNK]', 'the', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
        vocab = ['the', '##o', 'f', '##u', '##m', '##s', 'o', '##v', '##er', 'l', '##a', '##y', 'd', '##g']
        expected = [
            ['[UNK]', '##[UNK]', '##[UNK]', '##[UNK]', '##[UNK]'],  # TODO: fix this somehow
            ['the'],
            ['f', '##o', '##[UNK]'],
            ['[UNK]', '##u', '##m', '##[UNK]', '##s'],
            ['o', '##v', '##er'],
            ['the'],
            ['l', '##a', '##[UNK]', '##y'],
            ['d', '##o', '##g']]

        layer = CharBpeEmbedding(vocab, 4)
        adapted = layer.adapt(data)
        adapted = self.evaluate(adapted).to_list()
        adapted = [[w_.decode('utf-8') for w_ in w] for w in adapted]
        self.assertListEqual(adapted, expected)


@keras_parameterized.run_all_keras_modes
class CharCnnEmbeddingTest(keras_parameterized.TestCase):
    def test_layer(self):
        data = np.array(['[UNK]', 'the', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog'])
        vocab = ['e', 'o', 't', 'h']

        testing_utils.layer_test(
            CharCnnEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'filters': [32, 32, 64, 128, 256, 512, 1024],
                    'kernels': [1, 2, 3, 4, 5, 6, 7], 'char_dim': 16, 'activation': 'tanh', 'highways': 2,
                    'normalize_unicode': 'NFKC', 'lower_case': False, 'zero_digits': False, 'max_len': 50,
                    'reserved_words': None, 'embed_type': 'dense_auto', 'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            CharCnnEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'filters': [3, 1], 'kernels': [8, 16], 'char_dim': 16,
                    'activation': 'tanh', 'highways': 2, 'normalize_unicode': 'NFKC', 'lower_case': False,
                    'zero_digits': False, 'max_len': 50, 'reserved_words': None, 'embed_type': 'dense_auto',
                    'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            CharCnnEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'filters': [32, 32, 64, 128, 256, 512, 1024],
                    'kernels': [1, 2, 3, 4, 5, 6, 7], 'char_dim': 16, 'activation': 'tanh', 'highways': 2,
                    'normalize_unicode': 'NFKC', 'lower_case': False, 'zero_digits': False, 'max_len': 50,
                    'reserved_words': ['EOW', '~TesT~'], 'embed_type': 'dense_auto', 'adapt_cutoff': None,
                    'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            CharCnnEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'filters': [32, 32, 64, 128, 256, 512, 1024],
                    'kernels': [1, 2, 3, 4, 5, 6, 7], 'char_dim': 16, 'activation': 'relu', 'highways': 2,
                    'normalize_unicode': 'NFKC', 'lower_case': False, 'zero_digits': False, 'max_len': 50,
                    'reserved_words': None, 'embed_type': 'dense_auto', 'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )
        testing_utils.layer_test(
            CharCnnEmbedding,
            kwargs={'vocabulary': vocab, 'output_dim': 12, 'filters': [32, 32, 64, 128, 256, 512, 1024],
                    'kernels': [1, 2, 3, 4, 5, 6, 7], 'char_dim': 16, 'activation': 'tanh', 'highways': 0,
                    'normalize_unicode': 'NFKC', 'lower_case': False, 'zero_digits': False, 'max_len': 50,
                    'reserved_words': None, 'embed_type': 'dense_auto', 'adapt_cutoff': None, 'adapt_factor': 4},
            input_data=data,
            expected_output_dtype='float32',
            expected_output_shape=[None, 12]
        )

    def test_reserved_words(self):
        layer = CharCnnEmbedding([], 4, [1], [32])
        self.assertListEqual(layer._reserved_words, [layer.UNK_MARK, layer.BOW_MARK, layer.EOW_MARK])

        layer = CharCnnEmbedding([], 4, [1], [32], reserved_words=['~TesT~'])
        self.assertListEqual(layer._reserved_words, [layer.UNK_MARK, layer.BOW_MARK, layer.EOW_MARK, '~TesT~'])

    def test_vocab(self):
        counts = Counter({'the': 4, 'fox': 2, 'jumps': 2, '\u1E9B\u0323': 2, 'dog': 1})
        adapts = [
            ('[BOW]', 11), ('[EOW]', 11), ('t', 4), ('h', 4), ('e', 4), ('o', 3), ('f', 2), ('x', 2), ('j', 2),
            ('u', 2), ('m', 2), ('p', 2), ('s', 2), ('\u1E69', 2), ('d', 1), ('g', 1)]
        layer = CharCnnEmbedding([], 4, [1], [32])
        self.assertListEqual(layer.vocab(counts).most_common(), adapts)

        vocab = ['t', 'h', 'e', 'f', 'o', 'x', 'j', 'u', 'm', 'p', 's', '\u1E69']
        layer = CharCnnEmbedding(vocab, 4, [1], [32])
        self.assertListEqual(layer._vocabulary, [layer.UNK_MARK, layer.BOW_MARK, layer.EOW_MARK] + vocab)

        layer = CharCnnEmbedding(vocab, 4, [1], [32], reserved_words=['~TesT~'])
        self.assertListEqual(layer._vocabulary, [layer.UNK_MARK, layer.BOW_MARK, layer.EOW_MARK, '~TesT~'] + vocab)

    def test_adapt(self):
        data = ['[UNK]', 'the', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
        expected = [
            ['[BOW]', '[UNK]', '[EOW]'],
            ['[BOW]', 't', 'h', 'e', '[EOW]'],
            ['[BOW]', 'f', 'o', 'x', '[EOW]'],
            ['[BOW]', 'j', 'u', 'm', 'p', 's', '[EOW]'],
            ['[BOW]', 'o', 'v', 'e', 'r', '[EOW]'],
            ['[BOW]', 't', 'h', 'e', '[EOW]'],
            ['[BOW]', 'l', 'a', 'z', 'y', '[EOW]'],
            ['[BOW]', 'd', 'o', 'g', '[EOW]']]

        layer = CharCnnEmbedding([], 4, [1], [32])
        adapted = layer.adapt(data)
        adapted = self.evaluate(adapted).to_list()
        adapted = [[w_.decode('utf-8') for w_ in w] for w in adapted]
        self.assertListEqual(adapted, expected)

    def test_adapt_long_odd(self):
        data = ['[UNK]', 'the', 'fox', '0123456789abcdefghij', '0123456789abcdefghijk']
        expected = [
            ['[BOW]', '[UNK]', '[EOW]'],
            ['[BOW]', 't', 'h', 'e', '[EOW]'],
            ['[BOW]', 'f', 'o', 'x', '[EOW]'],
            ['[BOW]', '0', '1', '2', '3', '�', 'h', 'i', 'j', '[EOW]'],
            ['[BOW]', '0', '1', '2', '3', '�', 'i', 'j', 'k', '[EOW]']]

        layer = CharCnnEmbedding([], 4, [1], [32], max_len=10)
        adapted = layer.adapt(data)
        adapted = self.evaluate(adapted).to_list()
        adapted = [[w_.decode('utf-8') for w_ in w] for w in adapted]
        self.assertListEqual(adapted, expected)

    def test_adapt_long_even(self):
        data = ['[UNK]', 'the', 'fox', '0123456789abcdefghij', '0123456789abcdefghijk']
        expected = [
            ['[BOW]', '[UNK]', '[EOW]'],
            ['[BOW]', 't', 'h', 'e', '[EOW]'],
            ['[BOW]', 'f', 'o', 'x', '[EOW]'],
            ['[BOW]', '0', '1', '2', '�', 'h', 'i', 'j', '[EOW]'],
            ['[BOW]', '0', '1', '2', '�', 'i', 'j', 'k', '[EOW]']]

        layer = CharCnnEmbedding([], 4, [1], [32], max_len=9)
        adapted = layer.adapt(data)
        adapted = self.evaluate(adapted).to_list()
        adapted = [[w_.decode('utf-8') for w_ in w] for w in adapted]
        self.assertListEqual(adapted, expected)


@keras_parameterized.run_all_keras_modes
class HighwayTest(keras_parameterized.TestCase):
    def test_layer(self):
        testing_utils.layer_test(
            Highway,
            kwargs={},
            input_shape=[2, 16, 8],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 16, 8]
        )
        testing_utils.layer_test(
            Highway,
            kwargs={},
            input_shape=[2, 16, 8, 4],
            input_dtype='float32',
            expected_output_dtype='float32',
            expected_output_shape=[None, 16, 8, 4]
        )


if __name__ == "__main__":
    tf.test.main()
