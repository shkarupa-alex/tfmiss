from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from collections import Counter


def init_buckets(len2freq):
    """Splits length-to-frequency mapping into list of initial buckets.

    Args:
        len2freq: Dict of `sequence length`: `frequency`.

    Returns:
        List of buckets. Each bucket is a tuple (`bucket boundary`, `subset of len2freq`).
    """
    source = Counter(len2freq)

    if not len(source):
        raise ValueError('Empty length-to-frequency map')

    if not all(map(lambda x: isinstance(x, int), source.keys())):
        raise ValueError('Keys of length-to-frequency must be integers')

    if not all(map(lambda x: isinstance(x, int), source.values())):
        raise ValueError('Values of length-to-frequency must be integers')

    denominator = 8
    lengths = sorted(source.keys())

    buckets = []
    for lng in lengths:
        b = int(np.ceil(lng / denominator)) * denominator + 1
        if not len(buckets) or buckets[-1][0] != b:
            buckets.append((b, {}))
        buckets[-1][1][lng] = source[lng]

    return buckets


def waste_frac(bucket):
    """Estimates fraction of `PAD` elements in bucket.

    Args:
        bucket: A bucket to process.

    Returns:
        Float in range [0.0; 1.0): `PAD` fraction.
    """
    if not isinstance(bucket, tuple) or len(bucket) not in {0, 2}:
        raise ValueError('Wrong bucket format')

    if not len(bucket):
        return 0.0

    boundary, len2freq = bucket
    zero_cnt = sum([(boundary - 1 - lng) * f for lng, f in len2freq.items()])
    total_freq = sum([f for _, f in len2freq.items()])

    return zero_cnt / (total_freq * (boundary - 1))


def merge_buckets(bucket1, bucket2):
    """Merges 2 buckets into one.

    Args:
        bucket1: First bucket.
        bucket2: Second bucket.

    Returns:
        New bucket with maximum of boundaries and joined length-to-frequency mappings.
    """
    if not len(bucket1) and not len(bucket2):
        return tuple()
    elif not len(bucket1):
        return bucket2
    elif not len(bucket2):
        return bucket1

    boundary = max([bucket1[0], bucket2[0]])
    len2freq = Counter(bucket1[1]) + Counter(bucket2[1])

    return boundary, len2freq


def merge_allowed(merged, buckets, min_waste, max_waste, min_aggr):
    """Checks if bucket merging allowed.

    Args:
        merged: A merged bucket to test.
        buckets: All existing buckets.
        min_waste: Minimum waste fraction.
        max_waste: Maximum waste fraction
        min_aggr: Minimum aggregate fraction.

    Returns:
        Boolean flag of allowing merge
    """
    if not len(merged):
        return False

    total_freq = sum([f for (_, l2f) in buckets for _, f in l2f.items()])
    curr_aggr = sum([f for _, f in merged[1].items()]) * 1.0 / total_freq
    curr_waste = waste_frac(merged)

    return curr_waste < min_waste or curr_waste < max_waste and curr_aggr < min_aggr


def group_buckets(before, middle, after, min_waste, max_waste, min_aggr):
    """Merges buckets one by one from `before` and `after` into middle until merging allowed.

    Args:
        before: List of buckets before `middle`.
        middle: Current bucket to expand.
        after: List of buckets after `middle`.
        min_waste: Minimum waste fraction.
        max_waste: Maximum waste fraction
        min_aggr: Minimum aggregate fraction.

    Returns:
        Re-groupped `before`, `middle` and `after` buckets.
    """
    last_size = 0

    while len(middle[1]) > last_size:
        last_size = len(middle[1])

        left = before[-1] if len(before) else tuple()
        right = after[0] if len(after) else tuple()

        with_left = merge_buckets(left, middle)
        with_right = merge_buckets(right, middle)

        waste_left = waste_frac(with_left)
        waste_right = waste_frac(with_right)

        all_buckets = before + [middle] + after
        allow_left = merge_allowed(with_left, all_buckets, min_waste, max_waste, min_aggr)
        allow_right = merge_allowed(with_right, all_buckets, min_waste, max_waste, min_aggr)

        if allow_left and (not allow_right or waste_left < waste_right):
            before = before[:-1]
            middle = with_left
        elif allow_right and (not allow_left or waste_right < waste_left):
            middle = with_right
            after = after[1:]

    return before, middle, after


def estimate_bucket_boundaries(len2freq, min_waste=0.01, max_waste=0.1, min_aggr=0.01):
    """Estimates and merges buckets from the most common (middle).
    By default tries to make buckets with more then 1% of samples and no more then 1% of paddings
    or at least  no more then 10% of paddings.

    Args:
        len2freq: Dict of `sequence length`: `frequency`.
        min_waste: Minimum waste fraction.
        max_waste: Maximum waste fraction
        min_aggr: Minimum aggregate fraction.

    Returns:
        List of integer bucket boundaries.
    """
    buckets = init_buckets(len2freq)

    sizes = [sum(l2f.values()) for _, l2f in buckets]
    start = sizes.index(max(sizes))

    before = buckets[:start]
    middle = buckets[start]
    after = buckets[start + 1:]

    before, middle, after = group_buckets(before, middle, after, min_waste, max_waste, min_aggr)
    result = [middle]

    while len(before):
        middle = before[-1]
        before = before[:-1]
        before, middle, _ = group_buckets(before, middle, result + after, min_waste, max_waste, min_aggr)

        result = [middle] + result

    while len(after):
        middle = after[0]
        after = after[1:]
        _, middle, after = group_buckets(result, middle, after, min_waste, max_waste, min_aggr)

        result = result + [middle]

    original = Counter(len2freq)
    restored = sum([Counter(r[1]) for r in result], Counter())
    assert set(original.keys()) == set(restored.keys())
    assert set(original.values()) == set(restored.values())

    return [r[0] for r in result]


def estimate_bucket_pipeline(bucket_boundaries, num_samples, safe=True):
    """Estimates bach sizes and reduces bucket boundaries to fit required number of samples per batch.

    Args:
        bucket_boundaries: pre-estimated bucket boundaries (see `estimate_bucket_boundaries`).
        num_samples: number of samples per batch (same as `batch size` / `sequence length`).
        safe: Do not allow maximum number of samples to be greater then `num_samples`.

    Returns:
        A tuple of (`reduced bucket boundaries`, `batch sizes`, `maximum boundary`).
        Bucket boundaries and batch sizes must be supplied to `tf.data.experimental.bucket_by_sequence_length`.
        Maximum boundary should be used to filter out too long sequences
        with `tf.data.Dataset.filter` (`length` < `max_boundary`).
    """
    if len(bucket_boundaries) < 2:
        raise ValueError('Bucket boundaries must contain at least 2 values')

    batch_step = 8

    batch_sizes = []
    for boundary in bucket_boundaries:
        batch_size = num_samples / (boundary - 1)
        batch_size = np.floor(batch_size / batch_step) if safe \
            else np.round(batch_size / batch_step)
        batch_size = batch_step * batch_size

        if safe and batch_size < batch_step:
            if len(batch_sizes) < 2:
                raise ValueError('Too few samples per batch')

            return bucket_boundaries[:len(batch_sizes) - 1], batch_sizes, bucket_boundaries[len(batch_sizes) - 1]

        batch_sizes.append(max(batch_step, batch_size.astype(np.int)))

    return bucket_boundaries[:-1], batch_sizes, bucket_boundaries[-1]
