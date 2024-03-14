import json
import numpy as np
import tensorflow as tf
from collections import Counter
from tfmiss.training.adapt import test_device_matmul, interpolate_matmul_cost, build_zipf_vocab
from tfmiss.training.adapt import generate_class_clusters, adaptive_split_cost, estimate_best_splits


class TestDeviceMatmulTest(tf.test.TestCase):
    def test_normal_cpu(self):
        device_params = test_device_matmul(
            max_batch=5, max_hidden=9, max_classes=17, repeats=1, device='CPU:0', dtype='float32')
        self.assertListEqual([1, 2, 3, 4, 5], device_params['batch_sizes'])
        self.assertListEqual([1, 2, 3, 4, 5, 7, 8, 9], device_params['hidden_sizes'])
        self.assertListEqual([1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17], device_params['class_sizes'])
        self.assertLen(device_params['cost_values'], 5 * 8 * 11)

    def test_normal_gpu(self):
        if not tf.config.list_physical_devices('GPU'):
            self.skipTest('No GPU available')

        device_params = test_device_matmul(
            max_batch=32, max_hidden=8, max_classes=16, repeats=1, device='GPU:0', dtype='float32')
        self.assertListEqual([1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 23, 24, 25, 31, 32],
                             device_params['batch_sizes'])
        self.assertListEqual([1, 2, 3, 4, 5, 7, 8], device_params['hidden_sizes'])
        self.assertListEqual([1, 2, 3, 4, 5, 7, 8, 9, 15, 16], device_params['class_sizes'])
        self.assertLen(device_params['cost_values'], 16 * 7 * 10)

    def test_serializable(self):
        device_params = test_device_matmul(
            max_batch=5, max_hidden=9, max_classes=17, repeats=1, device='CPU:0', dtype='float32')
        json.dumps(device_params)


class InterpolateMatmulCostTest(tf.test.TestCase):
    def test_normal(self):
        device_params = test_device_matmul(
            max_batch=4, max_hidden=8, max_classes=128, repeats=1, device='CPU:0', dtype='float32')
        approx_cost = interpolate_matmul_cost(device_params)

        batch4 = approx_cost(4, 8, 16)
        batch3 = approx_cost(3, 8, 16)
        self.assertAlmostEqual(batch3, batch4, places=1)

    def tes_failure(self):
        device_params = test_device_matmul(
            max_batch=4, max_hidden=8, max_classes=128, repeats=1, device='GPU:0', dtype='float32')
        approx_cost = interpolate_matmul_cost(device_params)

        with self.assertRaises(ValueError):
            approx_cost(4, 9, 16)
        with self.assertRaises(ValueError):
            approx_cost(1, 9, 1)


class BuildZipfVocabTest(tf.test.TestCase):
    def test_normal(self):
        vocab = build_zipf_vocab(10)
        self.assertIsInstance(vocab, Counter)
        self.assertListEqual([
            (1, 10.0), (2, 5.0), (3, 3.3333333333333335), (4, 2.5), (5, 2.0),
            (6, 1.6666666666666667), (7, 1.4285714285714286), (8, 1.25), (9, 1.1111111111111112), (10, 1.0)
        ], vocab.most_common())


class GenerateClassClustersTest(tf.test.TestCase):
    def test_normal_1(self):
        vocab = build_zipf_vocab(60)
        freq = np.array([f for _, f in vocab.most_common()])
        prob = freq / np.sum(freq)
        accum = np.cumsum(prob)

        splits = generate_class_clusters(1, accum).tolist()
        self.assertListEqual([
            [15, 45],
            [23, 37]
        ], splits)

    def test_normal_2(self):
        vocab = build_zipf_vocab(120)
        freq = np.array([f for _, f in vocab.most_common()])
        prob = freq / np.sum(freq)
        accum = np.cumsum(prob)

        splits = generate_class_clusters(2, accum).tolist()
        self.assertListEqual([
            [14, 32, 74], [14, 40, 66], [14, 48, 58],
            [22, 32, 66], [22, 40, 58],
            [30, 40, 50]
        ], splits)

    def test_normal_4(self):
        vocab = build_zipf_vocab(240)
        freq = np.array([f for _, f in vocab.most_common()])
        prob = freq / np.sum(freq)
        accum = np.cumsum(prob)

        splits = generate_class_clusters(4, accum).tolist()
        self.assertListEqual([
            [12, 24, 40, 64, 100], [12, 24, 40, 72, 92], [12, 24, 48, 64, 92], [12, 24, 48, 72, 84],
            [12, 24, 56, 64, 84], [12, 32, 40, 64, 92], [12, 32, 48, 64, 84], [12, 32, 56, 64, 76],
            [12, 40, 48, 64, 76],
            [20, 24, 40, 64, 92], [20, 32, 48, 64, 76],
            [28, 40, 48, 56, 68]
        ], splits)


class AdaptiveSplitCostTest(tf.test.TestCase):
    def test_normal(self):
        device_params = test_device_matmul(
            max_batch=8, max_hidden=25, max_classes=30, repeats=1, device='CPU:0', dtype='float32')
        approx_cost = interpolate_matmul_cost(device_params)

        freq_vocab = build_zipf_vocab(30)
        all_freq = np.array([f for _, f in freq_vocab.most_common()])
        all_prob = all_freq / np.sum(all_freq)
        prob_accum = np.cumsum(all_prob)

        split_size = [2, 9, 19]
        adaptive_split_cost(approx_cost, prob_accum, split_size, batch_size=6, hidden_size=24, factor=2)


class EstimateBestSplitsTest(tf.test.TestCase):
    def test_normalize_to_normal_softmax(self):
        device_params = test_device_matmul(
            max_batch=8, max_hidden=25, max_classes=120, repeats=1, device='CPU:0', dtype='float32')
        freq_vocab = build_zipf_vocab(120)
        batch_sizes, head_sizes, speed_ups, best_splits = estimate_best_splits(
            device_params, freq_vocab, num_tails=2, hidden_size=24, factor=2)

        self.assertListEqual([1, 2, 3, 4, 5, 7, 8], batch_sizes)
        self.assertListEqual([14, 22, 30], head_sizes)
        self.assertLen(speed_ups, 7 * 3)
        self.assertLen(best_splits, 7 * 3)

    def test_normalize_to_worst_adaptive(self):
        device_params = test_device_matmul(
            max_batch=8, max_hidden=25, max_classes=110, repeats=1, device='CPU:0', dtype='float32')
        freq_vocab = build_zipf_vocab(120)
        batch_sizes, head_sizes, speed_ups, best_splits = estimate_best_splits(
            device_params, freq_vocab, num_tails=2, hidden_size=24, factor=2)

        self.assertListEqual([1, 2, 3, 4, 5, 7, 8], batch_sizes)
        self.assertListEqual([14, 22, 30], head_sizes)
        self.assertLen(speed_ups, 7 * 3)
        self.assertLen(best_splits, 7 * 3)


if __name__ == '__main__':
    tf.test.main()
