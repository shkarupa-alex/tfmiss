from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from collections import Counter
from tensorflow.python.framework import test_util
from tfmiss.preprocessing.sampling import down_sample, sample_mask


@test_util.run_all_in_graph_and_eager_modes
class DownSampleTest(tf.test.TestCase):
    def testEmpty(self):
        source = tf.constant([], dtype=tf.string)
        downsampled = self.evaluate(down_sample(source, ['_'], [1], '', 1., min_freq=0, seed=1))
        self.assertEquals([], downsampled.tolist())

    def testDense(self):
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
            ['', '', 'quick', 'brown', 'fox', '', 'over', 'the', '', 'dog', ''],
            downsampled.tolist()
        )

    def testRagged2D(self):
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
            down_sample(source, freq_keys, freq_vals, '', 1e-2, min_freq=2, seed=1)
                .to_tensor(default_value='')
        )

        self.assertListEqual([
            ['', '', '', '', ''],
            ['', 'quick', 'brown', 'fox', ''],
            ['over', 'the', '', 'dog', ''],
            ['', '', '', '', '']
        ], downsampled.tolist())

    def testRagged3D(self):
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
            down_sample(source, freq_keys, freq_vals, '', 1e-2, min_freq=2, seed=1)
                .to_tensor(default_value='')
        )

        self.assertListEqual([
            [['', '', '', ''],
             ['', '', '', '']],
            [['', 'quick', '', ''],
             ['brown', 'fox', '', '']],
            [['over', 'the', '', 'dog'],
             ['', '', '', '']],
            [['', '', '', ''],
             ['', '', '', '']]
        ], downsampled.tolist())


@test_util.run_all_in_graph_and_eager_modes
class SampleMaskTest(tf.test.TestCase):
    def testDenseShapeInference(self):
        mask = sample_mask(['_'], ['_'], [1], 1., min_freq=0, seed=1)
        self.assertEqual([1], mask.shape.as_list())

    def testEmpty(self):
        mask = self.evaluate(sample_mask([], ['_'], [1], 1., min_freq=0, seed=1))
        self.assertEquals([], mask.tolist())

    def testUniq(self):
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

    def testNoUniq(self):
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
