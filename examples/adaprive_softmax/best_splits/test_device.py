from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import tensorflow as tf
from tfmiss.training import test_device_matmul

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Measure matrix multiplication time: '
                    '[BATCH_SIZE, HIDDEN_SIZE] * [HIDDEN_SIZE, NUM_CLASSES]')
    parser.add_argument(
        'dev_perf',
        type=argparse.FileType('w'),
        help='File to store device matmul measurements')
    parser.add_argument(
        '--max_batch',
        type=int,
        default=2 ** 11,
        help='Maximum batch size')
    parser.add_argument(
        '--max_hidden',
        type=int,
        default=2 ** 11,
        help='Maximum hidden size')
    parser.add_argument(
        '--max_classes',
        type=int,
        default=2 ** 16,
        help='Maximum number of classes + maximum number of clusters (including head one)')
    parser.add_argument(
        '--repeats',
        type=int,
        default=1000,
        help='Number repeats to average')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU:0',
        help='Device name')
    parser.add_argument(
        '--dtype',
        type=str,
        default='float32',
        help='Data type')

    argv, _ = parser.parse_known_args()
    tf.get_logger().setLevel(logging.INFO)

    device_params = test_device_matmul(
        max_batch=argv.max_batch,
        max_hidden=argv.max_hidden,
        max_classes=argv.max_classes,
        repeats=argv.repeats,
        device=argv.device,
        dtype=argv.dtype
    )
    json.dump(device_params, argv.dev_perf, allow_nan=False, indent=2)
