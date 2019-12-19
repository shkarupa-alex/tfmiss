from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import Counter
from tensorflow.python.framework import test_util
from tfmiss.preprocessing.sampling import down_sample, sample_mask


@test_util.run_all_in_graph_and_eager_modes
class DownSampleTest(tf.test.TestCase):
    def test_empty(self):
        source = tf.constant([], dtype=tf.string)
        downsampled = self.evaluate(down_sample(source, ['_'], [1], '', 1., min_freq=0, seed=1))
        self.assertEqual([], downsampled.tolist())

    def test_dense(self):
        freq_vocab = Counter({
            'tensorflow': 99,
            'the': 9,
            'quick': 8,
            'brown': 7,
            'fox': 6,
            'over': 5,
            'dog': 4,
        })
        freq_vocab.update(['unk_{}'.format(i) for i in range(100)])  # noise
        freq_keys, freq_vals = zip(*freq_vocab.most_common())

        source = ['tensorflow', 'the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'tensorflow']
        downsampled = self.evaluate(down_sample(source, freq_keys, freq_vals, '', 1e-2, min_freq=2, seed=1))

        self.assertListEqual(
            [b'', b'', b'quick', b'brown', b'fox', b'', b'over', b'the', b'', b'dog', b''],
            downsampled.tolist()
        )

    def test_ragged_2d(self):
        freq_vocab = Counter({
            'tensorflow': 99,
            'the': 9,
            'quick': 8,
            'brown': 7,
            'fox': 6,
            'over': 5,
            'dog': 4,
        })
        freq_vocab.update(['unk_{}'.format(i) for i in range(100)])  # noise
        freq_keys, freq_vals = zip(*freq_vocab.most_common())

        source = tf.ragged.constant([
            ['tensorflow'],
            ['the', 'quick', 'brown', 'fox', 'jumped'],
            ['over', 'the', 'lazy', 'dog'],
            ['tensorflow']
        ])
        downsampled = self.evaluate(
            down_sample(source, freq_keys, freq_vals, '', 1e-2, min_freq=2, seed=1).to_tensor(default_value=''))

        self.assertListEqual([
            [b'', b'', b'', b'', b''],
            [b'', b'quick', b'brown', b'fox', b''],
            [b'over', b'the', b'', b'dog', b''],
            [b'', b'', b'', b'', b'']
        ], downsampled.tolist())

    def test_ragged_3d(self):
        freq_vocab = Counter({
            'tensorflow': 99,
            'the': 9,
            'quick': 8,
            'brown': 7,
            'fox': 6,
            'over': 5,
            'dog': 4,
        })
        freq_vocab.update(['unk_{}'.format(i) for i in range(100)])  # noise
        freq_keys, freq_vals = zip(*freq_vocab.most_common())

        source = tf.ragged.constant([
            [['tensorflow']],
            [['the', 'quick'],
             ['brown', 'fox', 'jumped']],
            [['over', 'the', 'lazy', 'dog']],
            [['tensorflow']]
        ])
        downsampled = self.evaluate(
            down_sample(source, freq_keys, freq_vals, '', 1e-2, min_freq=2, seed=1).to_tensor(default_value=''))

        self.assertListEqual([
            [[b'', b'', b'', b''],
             [b'', b'', b'', b'']],
            [[b'', b'quick', b'', b''],
             [b'brown', b'fox', b'', b'']],
            [[b'over', b'the', b'', b'dog'],
             [b'', b'', b'', b'']],
            [[b'', b'', b'', b''],
             [b'', b'', b'', b'']]
        ], downsampled.tolist())


@test_util.run_all_in_graph_and_eager_modes
class SampleMaskTest(tf.test.TestCase):
    def test_dense_shape_inference(self):
        mask = sample_mask(['_'], ['_'], [1], 1., min_freq=0, seed=1)
        self.assertEqual([1], mask.shape.as_list())

    def test_empty(self):
        mask = self.evaluate(sample_mask([], ['_'], [1], 1., min_freq=0, seed=1))
        self.assertEqual([], mask.tolist())

    def test_uniq(self):
        freq_vocab = Counter({
            'tensorflow': 99,
            'the': 9,
            'quick': 8,
            'brown': 7,
            'fox': 6,
            'over': 5,
            'dog': 4,
        })
        freq_vocab.update(['unk_{}'.format(i) for i in range(100)])  # noise
        freq_keys, freq_vals = zip(*freq_vocab.most_common())

        samples = ['tensorflow', 'the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'tensorflow']
        mask = self.evaluate(sample_mask(samples, freq_keys, freq_vals, 1e-2, min_freq=2, seed=1))

        self.assertListEqual(
            [False, False, True, True, True, False, True, True, False, True, False],
            mask.tolist()
        )

    def test_no_uniq(self):
        freq_vocab = Counter({
            'tensorflow': 99,
            'the': 9,
            'quick': 8,
            'brown': 7,
            'fox': 6,
            'over': 5,
            'dog': 4,
        })
        freq_vocab.update(['unk_{}'.format(i) for i in range(100)])  # noise
        freq_keys, freq_vals = zip(*freq_vocab.most_common())

        samples = ['tensorflow', 'the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog', 'tensorflow']
        mask = self.evaluate(sample_mask(samples, freq_keys, freq_vals, 1e-2, seed=1))

        self.assertListEqual(
            [False, False, True, True, False, False, False, True, False, False, False],
            mask.tolist()
        )


if __name__ == "__main__":
    tf.test.main()
