from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import tensorflow as tf
from nlpvocab import Vocabulary
from tabulate import tabulate
from tfmiss.training import build_zipf_vocab, estimate_best_splits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Find best vocabulary splits for `AdaptiveSoftmax` layer with chosen batch and head sizes')
    parser.add_argument(
        'dev_perf',
        type=argparse.FileType('r'),
        help='File to load device matmul measurements')
    parser.add_argument(
        'num_tails',
        type=int,
        default=5,
        help='Number of tail clusters')
    parser.add_argument(
        'batch_size',
        type=int,
        help='Maximum batch size')
    parser.add_argument(
        'hidden_size',
        type=int,
        help='Hidden size')
    parser.add_argument(
        '--classes_vocab',
        type=argparse.FileType('r'),
        default=None,
        help='Vocabulary with classes and corresponding frequencies (should be preferred over num_classes)')
    parser.add_argument(
        '--vocab_format',
        choices=[
            Vocabulary.FORMAT_BINARY_PICKLE,
            Vocabulary.FORMAT_TSV_WITH_HEADERS,
            Vocabulary.FORMAT_TSV_WITHOUT_HEADERS
        ],
        default=Vocabulary.FORMAT_BINARY_PICKLE,
        help='Vocabulary file format')
    parser.add_argument(
        '--num_classes',
        type=int,
        default=0,
        help='Number of classes')
    parser.add_argument(
        '--factor',
        type=int,
        default=4,
        help='Scale factor for tail projections')

    argv, _ = parser.parse_known_args()
    tf.get_logger().setLevel(logging.INFO)

    device_params = json.load(argv.dev_perf)

    if 0 == argv.num_classes and argv.classes_vocab is None:
        raise ValueError('Classes vocabulary or at least number of classes should be provided')
    if argv.classes_vocab:
        vocab_file = argv.classes_vocab.name
        argv.classes_vocab.close()
        freq_vocab = Vocabulary.load(vocab_file, format=argv.vocab_format)
    else:
        freq_vocab = build_zipf_vocab(argv.num_classes)

    batch_sizes, head_sizes, speed_ups, best_splits = estimate_best_splits(
        device_params=device_params,
        freq_vocab=freq_vocab,
        num_tails=argv.num_tails,
        hidden_size=argv.hidden_size,
        factor=argv.factor
    )

    # Batch-head-speedup table
    speedup_headers = ['BATCH\\HEAD'] + head_sizes
    speedup_table = []
    heads_len = len(head_sizes)
    for i, batch in enumerate(batch_sizes):
        row = [batch] + speed_ups[heads_len * i: heads_len * (i + 1)]
        speedup_table.append(row)
    print(tabulate(speedup_table, headers=speedup_headers, tablefmt='presto', floatfmt='.1f'))

    # Batch-head-split table
    splits_table = []
    for i, batch in enumerate(batch_sizes):
        for j, head in enumerate(head_sizes):
            split = best_splits[i * heads_len + j]
            row = [batch, head, str(split)]
            splits_table.append(row)
    print(tabulate(splits_table, headers=['BATCH', 'HEAD', 'SPLIT'], tablefmt='presto'))
