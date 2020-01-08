from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
from collections import Counter
from scipy.interpolate import LinearNDInterpolator

try:
    from functools import lru_cache
except:
    from repoze.lru import lru_cache


def test_device_matmul(max_batch, max_hidden, max_classes, repeats, device, dtype):
    """Measures matrix multiplication time: [BATCH, HIDDEN] * [HIDDEN, CLASSES].
    In case of sequences this is the same as: [BATCH / TIME, TIME, HIDDEN] * [HIDDEN, CLASSES]

    Args:
        max_batch: Maximum size of batch.
        max_hidden: Maximum size of input logits.
        max_classes: Maximum number of output classes.
        repeats: Number of repeats to average.
        device: Which device to use fo multiplication.
        dtype: Matrices data type.

    Returns:
        A dict with tested `batch_sizes`, `hidden_sizes` and `class_sizes` along with measured `cost_values`
        (time in seconds).
    """
    if max_classes < max_hidden:
        raise ValueError('Number of classes should be greater then input logits size')

    if 'GPU' not in device.upper():
        tf.get_logger().warn('Device matmul estimation is useful for GPUs. '
                             'You ask to measure non-GPU device. Hope you know what you are doing.')

    physical_devices = ', '.join([d.name for d in tf.config.experimental.list_physical_devices()])
    if device.upper() not in physical_devices:
        raise SystemError('Requested device {} is not available: {}'.format(device, physical_devices))

    dtype = tf.dtypes.as_dtype(dtype)
    if not (dtype.is_floating or dtype.is_complex):
        raise TypeError('Unable to test matrix multiplication time with non-floating dtype {}'.format(dtype))

    cost_values = []

    def _matmul_dim_space(max_val):
        log_space = np.power(2, np.arange(0, int(np.log2(max_val)) + 1))
        mean_space = (log_space[:-1] + log_space[1:]) // 2
        mod8_space = mean_space[mean_space % 8 == 0]
        rc_space = np.concatenate((log_space, mod8_space))

        final_space = np.concatenate((rc_space, rc_space - 1, rc_space + 1, [max_val]))
        final_space = np.unique(final_space)
        final_space = final_space[final_space > 0]
        final_space = final_space[final_space <= max_val]

        return np.array(final_space)

    batch_sizes = _matmul_dim_space(max_batch)
    hidden_sizes = _matmul_dim_space(max_hidden)
    class_sizes = _matmul_dim_space(max_classes)

    dense_grid = np.array(np.meshgrid(batch_sizes, hidden_sizes, class_sizes, indexing='ij')).T.reshape([-1, 3])
    with tf.device(device):
        # Check if device has enough memory
        left = tf.random.normal(shape=(batch_sizes[-1], hidden_sizes[-1]), dtype=dtype)
        right = tf.random.normal(shape=(hidden_sizes[-1], class_sizes[-1]), dtype=dtype)
        mult = tf.matmul(left, right)
        mult.numpy()

        del left, right, mult

        for step, (batch_size, hidden_size, class_size) in enumerate(dense_grid):
            left = tf.random.normal(shape=(batch_size, hidden_size), dtype=dtype)
            right = tf.random.normal(shape=(hidden_size, class_size), dtype=dtype)

            # TensorFlow initializes a GPU the first time it's used, exclude from timing.
            mult = tf.matmul(left, right)
            mult.numpy()
            start = time.time()

            for _ in range(repeats):
                mult = tf.matmul(left, right)

            # tf.matmul can return before completing the matrix multiplication (e.g., can return after enqueing the
            # operation on a CUDA stream). The mult.numpy() call below will ensure that all enqueued operations have
            # completed (and will also copy the result to host memory, so we're including a little more than just the
            # matmul operation time).
            mult.numpy()
            finish = time.time()
            mult.numpy()
            over = time.time()
            total = (finish - start - (over - finish)) * 1000 / repeats
            cost_values.append(total)

            del left, right, mult

            tf.get_logger().info('Done {} steps of {}'.format(step + 1, dense_grid.shape[0]))

    return {
        'batch_sizes': batch_sizes.tolist(),
        'hidden_sizes': hidden_sizes.tolist(),
        'class_sizes': class_sizes.tolist(),
        'cost_values': cost_values
    }


def interpolate_matmul_cost(device_params):
    """Interpolates matrix multiplication time according to measurements provided by `test_device_matmul` function.

    Args:
        device_params: A dict with 4 keys (`batch_sizes`, `hidden_sizes`, `class_sizes`, `cost_values`).
            Device performance measurements.

    Returns:
        A linear interpolation function with signature (batch_size, hidden_size, classes_size);
    """
    batch_sizes = device_params['batch_sizes']
    hidden_sizes = device_params['hidden_sizes']
    class_sizes = device_params['class_sizes']
    cost_values = device_params['cost_values']

    point_grid = np.array(np.meshgrid(batch_sizes, hidden_sizes, class_sizes, indexing='ij')).T.reshape([-1, 3])
    cost_values = np.array(cost_values)
    approx_cost = LinearNDInterpolator(point_grid, cost_values, fill_value=np.nan, rescale=True)

    @lru_cache(maxsize=10000)
    def _with_bounds(batch_size, hidden_size, num_classes):
        batch_size = max(1, batch_size)
        hidden_size = max(1, hidden_size)
        num_classes = max(1, num_classes)

        value = approx_cost(batch_size, hidden_size, num_classes)
        if np.isnan(value):
            raise ValueError('Required point ({}, {}, {}) is out of known bounds'.format(
                batch_size, hidden_size, num_classes))

        return value.item()

    return _with_bounds


def build_zipf_vocab(num_classes):
    """Builds frequency vocabulary according to Zipf's law.

    Args:
        num_classes: Total number of classes.

    Returns:
        A `Counter` instance with classes from 0 to num_classes - 1 and corresponding frequencies.
    """
    freq_keys = np.arange(1, num_classes + 1)
    freq_vals = num_classes / freq_keys

    return Counter({k: v for k, v in zip(freq_keys, freq_vals)})


def generate_class_clusters(num_tails, prob_accum, head=None):
    """Generates granular class splits for Adaptive Softmax.

    Args:
        num_tails: Number of tail clusters.
        prob_accum: A list of cumulative probabilities for all classes.
        head: Pre-estimated splits. Reserved for internal purposes.

    Returns:
        A list of possible splits. Each split is a list with cluster sizes.
        All cluster sizes except last one have size evenly dividable by 8.
        Head cluster size + number of clusters is evenly dividable by 8 too.
    """
    num_classes = len(prob_accum)
    size_space = np.cumsum(np.linspace(1, num_classes, 1000)).astype(np.int32)
    size_space = size_space[size_space > 0.01 * num_classes]
    size_space = size_space[size_space < 0.99 * num_classes]

    if head is None:
        if num_tails < 1:
            raise ValueError('There are should be at least one tail cluster')

        head_split = np.floor((size_space + num_tails) / 8).astype(np.int32) * 8 - num_tails
        head_split = np.unique(head_split[head_split > 0]).reshape([-1, 1])

        return generate_class_clusters(num_tails - 1, prob_accum, head_split)

    if 0 == np.size(head):
        raise ValueError('Could not generate required number of clusters. '
                         'Try to decrease number of clusters or increase number of classes.')

    body = []
    sizes = np.cumsum(head, axis=-1)
    for split, size in zip(head, sizes):
        consumed = size[-1]
        if 0 == num_tails:
            rest_size = num_classes - consumed
            subspace = np.array([rest_size])
        else:
            subspace = np.floor(size_space / 8).astype(np.int32) * 8
            subspace = np.unique(subspace)
            subspace = subspace[subspace < num_classes - consumed]

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
    if 0 == num_tails:
        return body

    return generate_class_clusters(num_tails - 1, prob_accum, body)


def adaptive_split_cost(approx_cost, prob_accum, cluster_sizes, batch_size, hidden_size, factor):
    """Estimates computation time for adaptpive softmax split.

    Args:
        approx_cost: Function to estimate matmul for batch-hidden_size-class matrices.
        prob_accum: Per-class appearance probability.
        cluster_sizes: List of cluster sizes
        batch_size: Size of input batch
        hidden_size: Size of input logits
        factor: Scale factor for tail projections.

    Returns:
        Split computation time.
    """
    if np.sum(cluster_sizes).item() != len(prob_accum):
        raise ValueError('Wrong inputs: Sum of cluster sizes should be equal to size of accumulated probabilities.')
    cluster_accum = np.cumsum(cluster_sizes)

    cost = approx_cost(batch_size, hidden_size, cluster_sizes[0] + len(cluster_sizes) - 1)  # Root prediction cost

    prev_dim = None
    for i, tail_size in enumerate(cluster_sizes[1:]):
        dim = hidden_size / (factor ** (i + 1))
        dim = max(1, round(dim / 8)) * 8

        if dim == prev_dim:
            raise ValueError('Some clusters have same internal size. '
                             'Try to decrease number of clusters or `factor`')
        prev_dim = dim

        tail_start, tail_end = cluster_accum[i] - 1, cluster_accum[i + 1] - 1
        clust_prob = prob_accum[tail_end] - prob_accum[tail_start]
        tail_batch = int(batch_size * clust_prob)

        # In most cases we can't guarantee tail batch size evenly dividable by 8. So, for estimation it won't.
        tail_batch = tail_batch + 1 if 0 == tail_batch % 8 else tail_batch

        cost += approx_cost(tail_batch, hidden_size, dim)  # Tail projection cost
        cost += approx_cost(tail_batch, dim, tail_size)  # Tail prediction cost

    return cost


def estimate_best_splits(device_params, freq_vocab, num_tails, hidden_size, factor):
    """Estimates best class splits for Adaptive Softmax.

    Args:
        device_params: A dict with 4 keys (`batch_sizes`, `hidden_sizes`, `class_sizes`, `cost_values`).
            Device performance measurements.
        freq_vocab: Class-to-frequency counter.
        num_tails: Number of tail clusters.
        hidden_size: Size of input logits
        factor: Scale factor for tail projections.

    Returns:
        A tuple of:
            - unique batch sizes
            - unique head sizes
            - speedups for each batch & head
            - split indices for each batch & head
    """
    approx_cost = interpolate_matmul_cost(device_params)

    if not isinstance(freq_vocab, Counter):
        raise ValueError('Frequency vocabulary should be a Counter instance.')
    all_freq = np.array([f for _, f in freq_vocab.most_common()])
    prob_dist = all_freq / np.sum(all_freq)
    prob_accum = np.cumsum(prob_dist)

    all_splits = generate_class_clusters(num_tails, prob_accum)
    head_sizes = list(np.unique(all_splits[:, 0]))

    batch_sizes = [bs for bs in device_params['batch_sizes'] if bs < 8 or bs % 8 == 0]
    batch_sizes = sorted(set(batch_sizes + [max(device_params['batch_sizes'])]))
    try:
        base_costs = [approx_cost(batch, hidden_size, len(prob_accum)) for batch in batch_sizes]
    except ValueError:
        base_costs = None
        tf.get_logger().warning('Can\'t estimate non-adaptive softmax computation time. '
                                'Will use worst split time to compute speedup.')

    best_ids, split_speedups = [], []
    for bi, batch in enumerate(batch_sizes):
        with_costs = {}
        for split in all_splits:
            curr_cost = adaptive_split_cost(approx_cost, prob_accum, split, batch, hidden_size, factor)
            head_size = split[0]
            if head_size not in with_costs or with_costs[head_size][0] > curr_cost:
                split_ids = np.cumsum(split[:-1])
                with_costs[head_size] = curr_cost, list(split_ids)

        max_cost = max([with_costs[hs][0] for hs in head_sizes])
        if base_costs is not None:
            max_cost = base_costs[bi]

        for hs in head_sizes:
            split_speedups.append(max_cost / with_costs[hs][0])
            best_ids.append(with_costs[hs][1])

    return batch_sizes, head_sizes, split_speedups, best_ids
