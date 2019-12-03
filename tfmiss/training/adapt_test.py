from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from collections import Counter
from tfmiss.training.adapt import test_device_matmul, make_matmul_cost, build_zipf_vocab, gen_all_splits
from tfmiss.training.adapt import estimate_split_cost, estimate_best_splits


class TestDeviceMatmulTest(tf.test.TestCase):
    def test_normal_cpu(self):
        device_params = test_device_matmul(
            max_batch=5, max_hidden=9, max_classes=17, repeats=1, device='CPU:0', dtype='float32')
        self.assertListEqual([1, 2, 4, 5], device_params['batch_sizes'].tolist())
        self.assertListEqual([1, 2, 4, 8, 9], device_params['hidden_sizes'].tolist())
        self.assertListEqual([1, 2, 4, 8, 16, 17], device_params['num_classes'].tolist())
        self.assertLen(device_params['cost_values'], 4 * 5 * 6)

    def test_normal_gpu(self):
        if not tf.test.is_gpu_available(cuda_only=True):
            self.skipTest('No GPU available')

        device_params = test_device_matmul(
            max_batch=4, max_hidden=8, max_classes=16, repeats=1, device='CPU:0', dtype='float32')
        self.assertListEqual([1, 2, 4, 8, 16, 24, 32], device_params['batch_sizes'].tolist())
        self.assertListEqual([1, 2, 4, 8], device_params['hidden_sizes'].tolist())
        self.assertListEqual([1, 2, 4, 8, 16], device_params['num_classes'].tolist())
        self.assertLen(device_params['cost_values'], 7 * 4 * 5)


class MakeMatmulCostTest(tf.test.TestCase):
    def test_normal(self):
        device_params = test_device_matmul(
            max_batch=4, max_hidden=8, max_classes=128, repeats=1, device='CPU:0', dtype='float32')
        approx_cost = make_matmul_cost(device_params)

        batch4 = approx_cost(4, 8, 16)
        batch3 = approx_cost(3, 8, 16)
        self.assertAlmostEqual(batch3, batch4, places=1)


class BuildZipfVocabTest(tf.test.TestCase):
    def test_normal(self):
        vocab = build_zipf_vocab(10)
        self.assertIsInstance(vocab, Counter)
        self.assertListEqual([
            (1, 10.0), (2, 5.0), (3, 3.3333333333333335), (4, 2.5), (5, 2.0),
            (6, 1.6666666666666667), (7, 1.4285714285714286), (8, 1.25), (9, 1.1111111111111112), (10, 1.0)
        ], vocab.most_common())


class GenAllSplitsTest(tf.test.TestCase):
    def test_normal_2(self):
        vocab = build_zipf_vocab(30)
        freq = np.array([f for _, f in vocab.most_common()])
        prob = freq / np.sum(freq)
        accum = np.cumsum(prob)

        splits = gen_all_splits(2, accum).tolist()
        self.assertListEqual([
            [5, 8, 17],
            [5, 10, 15],
            [5, 11, 14],
            [6, 10, 14],
            [6, 11, 13],
            [7, 10, 13],
            [8, 10, 12]
        ], splits)

    def test_normal_4(self):
        vocab = build_zipf_vocab(60)
        freq = np.array([f for _, f in vocab.most_common()])
        prob = freq / np.sum(freq)
        accum = np.cumsum(prob)

        splits = gen_all_splits(4, accum).tolist()
        self.assertListEqual([
            [4, 6, 11, 17, 22], [4, 6, 12, 17, 21],
            [5, 8, 11, 15, 21], [5, 8, 11, 17, 19], [5, 8, 12, 15, 20], [5, 9, 11, 15, 20], [5, 9, 12, 15, 19],
            [6, 8, 11, 15, 20], [6, 8, 12, 15, 19], [6, 9, 11, 15, 19], [6, 9, 12, 15, 18],
            [8, 9, 11, 14, 18], [8, 9, 11, 15, 17], [8, 9, 12, 14, 17]
        ], splits)


class EstimateSplitCostTest(tf.test.TestCase):
    def test_normal(self):
        device_params = test_device_matmul(
            max_batch=8, max_hidden=25, max_classes=30, repeats=1, device='CPU:0', dtype='float32')
        approx_cost = make_matmul_cost(device_params)

        freq_vocab = build_zipf_vocab(30)
        all_freq = np.array([f for _, f in freq_vocab.most_common()])
        all_prob = all_freq / np.sum(all_freq)
        prob_accum = np.cumsum(all_prob)

        split_size = np.cumsum([2, 9, 19])
        estimate_split_cost(approx_cost, prob_accum, split_size, batch_size=6, hidden_size=24, factor=2, mod8=True)
        estimate_split_cost(approx_cost, prob_accum, split_size, batch_size=6, hidden_size=24, factor=2, mod8=False)


class EstimateBestSplitsTest(tf.test.TestCase):
    def test_normal(self):
        device_params = test_device_matmul(
            max_batch=8, max_hidden=25, max_classes=30, repeats=1, device='CPU:0', dtype='float32')
        freq_vocab = build_zipf_vocab(30)
        batch_sizes, head_sizes, speed_ups, best_splits = estimate_best_splits(
            device_params, freq_vocab, num_clusters=3, hidden=24, factor=2, mod8=True)

        self.assertListEqual([1, 2, 4, 8], batch_sizes)
        self.assertListEqual([5, 6, 7, 8], head_sizes)
        self.assertLen(speed_ups, 16)
        self.assertLen(best_splits, 16)

if __name__ == '__main__':
    tf.test.main()
