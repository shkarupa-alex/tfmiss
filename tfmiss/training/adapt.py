from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from collections import Counter
from scipy.interpolate import LinearNDInterpolator


def test_device_matmul(max_batch, max_hidden, max_classes, repeats, device, dtype):
    if max_classes < max_hidden:
        raise ValueError('Number of classes should be greater then number of hidden units')

    if 'GPU' not in device.upper():
        tf.get_logger().warn('Target device is not a GPU')

    physical_devices = ', '.join([d.name for d in tf.config.experimental.list_physical_devices()])
    if device.upper() not in physical_devices:
        raise SystemError('Requested device {} is not available ({})'.format(device, physical_devices))

    dtype = tf.dtypes.as_dtype(dtype)
    if not (dtype.is_floating or dtype.is_complex):
        raise TypeError('Unable to test matrix multiplication speed with non-floating dtype {}'.format(dtype))

    cost_values = []

    batch_size0 = np.power(2, np.arange(0, int(np.log2(max_batch)) + 1))
    batch_size1 = (batch_size0[1:] + batch_size0[:-1]) // 2
    batch_size1 = batch_size1[batch_size1 % 8 == 0]
    batch_size = np.unique(np.concatenate((batch_size0, batch_size1, [max_batch])))

    hidden_size0 = np.power(2, np.arange(0, int(np.log2(max_hidden)) + 1))
    hidden_size1 = (hidden_size0[1:] + hidden_size0[:-1]) // 2
    hidden_size1 = hidden_size1[hidden_size1 % 8 == 0]
    hidden_size = np.unique(np.concatenate((hidden_size0, hidden_size1, [max_hidden])))

    num_classes0 = np.power(2, np.arange(0, int(np.log2(max_classes)) + 1))
    num_classes1 = (num_classes0[1:] + num_classes0[:-1]) // 2
    num_classes1 = num_classes1[num_classes1 % 8 == 0]
    num_classes = np.unique(np.concatenate((num_classes0, num_classes1, [max_classes])))

    dense_grid = np.array(np.meshgrid(batch_size, hidden_size, num_classes, indexing='ij')).T.reshape([-1, 3])

    with tf.device(device):
        # Check if device has enough memory
        left = tf.random.normal(shape=(batch_size[-1], hidden_size[-1]), dtype=dtype)
        right = tf.random.normal(shape=(hidden_size[-1], num_classes[-1]), dtype=dtype)
        mult = tf.matmul(left, right)
        val = mult.numpy()

        del left, right, mult, val

        for step, (_batch_size, _hidden_size, _num_classes) in enumerate(dense_grid):
            left = tf.random.normal(shape=(_batch_size, _hidden_size), dtype=dtype)
            right = tf.random.normal(shape=(_hidden_size, _num_classes), dtype=dtype)

            # TensorFlow initializes a GPU the first time it's used,
            # exclude from timing.
            mult = tf.matmul(left, right)
            val = mult.numpy()
            start = time.time()

            for _ in range(repeats):
                mult = tf.matmul(left, right)

            # tf.matmul can return before completing the matrix multiplication
            # (e.g., can return after enqueing the operation on a CUDA stream).
            # The x.numpy() call below will ensure that all enqueued operations
            # have completed (and will also copy the result to host memory,
            # so we're including a little more than just the matmul operation
            # time).
            val = mult.numpy()
            finish = time.time()
            val = mult.numpy()
            over = time.time()
            total = (finish - start - (over - finish)) * 1000 / repeats
            cost_values.append(total)

            del left, right, mult, val

            tf.get_logger().info('Done {} steps of {}'.format(step + 1, dense_grid.shape[0]))

    return {
        'batch_sizes': batch_size,
        'hidden_sizes': hidden_size,
        'num_classes': num_classes,
        'cost_values': cost_values
    }


def estimate_best_splits(device_params, freq_vocab, num_clusters, hidden, factor, mod8):
    approx_cost = make_matmul_cost(device_params)

    all_freq = np.array([f for _, f in freq_vocab.most_common()])
    prob_dist = all_freq / np.sum(all_freq)
    prob_accum = np.cumsum(prob_dist)

    all_splits = gen_all_splits(num_clusters - 1, prob_accum)
    if mod8:
        all_splits[:, 0] = np.ceil(all_splits[:, 0] / 8).astype(np.int32) * 8 - num_clusters + 1
        all_splits[:, 1:-1] = np.ceil(all_splits[:, 1:-1] / 8).astype(np.int32) * 8
        all_splits[:, -1] = len(freq_vocab) - np.sum(all_splits[:, :-1], axis=-1)
    head_sizes = list(np.unique(all_splits[:, 0]))

    best_splits, speed_ups = [], []
    batch_sizes = list(device_params['batch_sizes'])
    for curr_batch in batch_sizes:
        with_costs = {}

        base_cost = approx_cost(curr_batch, hidden, len(prob_accum))
        for curr_split in all_splits:
            curr_cost = estimate_split_cost(approx_cost, prob_accum, curr_split, curr_batch, hidden, factor, mod8)
            speed_up = base_cost / curr_cost
            head_size = curr_split[0]
            if head_size not in with_costs or with_costs[head_size][0] < speed_up:
                split_ids = np.cumsum(curr_split[:-1])
                with_costs[head_size] = speed_up, list(split_ids)

        for hs in head_sizes:
            speed_ups.append(with_costs[hs][0])
            best_splits.append(with_costs[hs][1])

    return batch_sizes, head_sizes, speed_ups, best_splits


def make_matmul_cost(device_params):
    batch_sizes = device_params['batch_sizes']
    hidden_sizes = device_params['hidden_sizes']
    num_classes = device_params['num_classes']
    cost_values = device_params['cost_values']

    point_grid = np.array(np.meshgrid(batch_sizes, hidden_sizes, num_classes, indexing='ij')).T.reshape([-1, 3])
    cost_values = np.array(cost_values)
    approx_cost = LinearNDInterpolator(point_grid, cost_values, fill_value=np.nan, rescale=True)

    def _with_bounds(_batch_size, _hidden_size, _num_classes):
        _batch_size = max(1.0, _batch_size)
        _hidden_size = max(1.0, _hidden_size)
        _num_classes = max(1.0, _num_classes)

        if batch_sizes[0] <= _batch_size <= batch_sizes[-1] and \
                hidden_sizes[0] <= _hidden_size <= hidden_sizes[-1] and \
                num_classes[0] <= _num_classes <= num_classes[-1]:
            return approx_cost(_batch_size, _hidden_size, _num_classes).tolist()

        raise ValueError('Required point is out of mesurements bounds')

    return _with_bounds


def build_zipf_vocab(num_classes):
    freq_keys = np.arange(1, num_classes + 1)
    freq_vals = num_classes / freq_keys

    return Counter({k: v for k, v in zip(freq_keys, freq_vals)})


def gen_all_splits(num_splits, prob_accum, head=None):
    num_classes = len(prob_accum)
    size_space = np.cumsum(np.linspace(1, num_classes, 1000)).astype(np.int32)
    size_space = size_space[size_space > 0.01 * num_classes]
    size_space = size_space[size_space < 0.99 * num_classes]

    if head is None:
        if num_splits < 1:
            raise ValueError('Number of splits should be greater then 2')
        head_split = size_space.reshape([-1, 1])

        return gen_all_splits(num_splits - 1, prob_accum, head_split)

    body = []
    sizes = np.cumsum(head, axis=-1)
    for split, size in zip(head, sizes):
        consumed = size[-1]
        if 0 == num_splits:
            rest_size = num_classes - consumed
            subspace = np.array([rest_size])
        else:
            subspace = size_space[size_space < num_classes - consumed]

        # Next cluster should be at least 10% larger
        last_size = split[-1]
        subspace = subspace[subspace > 1.1 * last_size]

        for bone in subspace:
            left = 0 if len(size) < 2 else size[-2] - 1
            middle = size[-1] - 1
            right = size[-1] + bone - 1

            # Next cluster should have at least 10% lower probability
            if 2.1 * prob_accum[middle] > prob_accum[left] + 1.1 * prob_accum[right]:
                body.append(split.tolist() + [bone])

    body = np.array(body)
    if 0 == num_splits:
        return body

    return gen_all_splits(num_splits - 1, prob_accum, body)


def estimate_split_cost(approx_cost, prob_accum, split_size, batch_size, hidden_size, factor, mod8):
    # root prediction cost
    cost = approx_cost(batch_size, hidden_size, split_size[0] + len(split_size) - 1)

    prev_dim = None
    for i, (tail_start, tail_end) in enumerate(zip(split_size[:-1], split_size[1:])):
        denom = 8 if mod8 else 1
        out = hidden_size // (factor ** (i + 1))
        out = int(np.ceil(out / denom)) * denom
        dim = max(denom, out)
        if dim != prev_dim:
            prev_dim = dim
        else:
            raise ValueError('Some clusters have same internal size. '
                             'Try to decrease number of clusters or `factor`')

        cluster_prob = prob_accum[tail_end - 1] - prob_accum[tail_start - 1]

        # tail projection cost
        cost += approx_cost(batch_size * cluster_prob, hidden_size, dim)

        # tail prediction cost
        cost += approx_cost(batch_size * cluster_prob, dim, tail_end - tail_start)

    return cost
