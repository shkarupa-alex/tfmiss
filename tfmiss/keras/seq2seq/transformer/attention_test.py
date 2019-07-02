from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras import keras_parameterized
from tfmiss.keras.seq2seq.transformer.attention import ScaledDotProductAttention, MultiHeadAttention, SelfAttention
from tfmiss.keras.testing_utils import layer_multi_io_test


@keras_parameterized.run_all_keras_modes
class ScaledDotProductAttentionTest(keras_parameterized.TestCase):
    def testLayer(self):
        qa, qb = 14, 32
        ka, kb = 19, qb
        va, vb = ka, 33
        ma, mb = qa, ka

        with tf.keras.utils.custom_object_scope({'ScaledDotProductAttention': ScaledDotProductAttention}):
            layer_multi_io_test(
                ScaledDotProductAttention,
                kwargs={
                    'dropout_rate': 0.1,
                },
                input_shapes=[
                    (64, qa, qb),
                    (64, ka, kb),
                    (64, va, vb),
                    (64, ma, mb),
                ]
            )

            layer_multi_io_test(
                ScaledDotProductAttention,
                kwargs={
                    'dropout_rate': 0.1,
                },
                input_shapes=[
                    (8, 64, qa, qb),
                    (8, 64, ka, kb),
                    (8, 64, va, vb),
                    (8, 64, ma, mb),
                ]
            )


@keras_parameterized.run_all_keras_modes
class MultiHeadAttentionTest(keras_parameterized.TestCase):
    def testLayer(self):
        num_heads = 8
        qa, qb = 14, 32
        ka, kb = 19, qb
        va, vb = ka, 33
        ma, mb, mc = num_heads, qa, ka

        with tf.keras.utils.custom_object_scope({'MultiHeadAttention': MultiHeadAttention}):
            layer_multi_io_test(
                MultiHeadAttention,
                kwargs={
                    'hidden_size': 16,
                    'num_heads': num_heads,
                    'dropout_rate': 0.1,
                },
                input_shapes=[
                    (64, qa, qb),
                    (64, ka, kb),
                    (64, va, vb),
                    (64, ma, mb, mc),
                ]
            )


@keras_parameterized.run_all_keras_modes
class SelfAttentionTest(keras_parameterized.TestCase):
    def testLayer(self):
        num_heads = 8
        qa, qb = 14, 32
        ma, mb, mc = num_heads, qa, qa

        with tf.keras.utils.custom_object_scope({'SelfAttention': SelfAttention}):
            layer_multi_io_test(
                SelfAttention,
                kwargs={
                    'hidden_size': 16,
                    'num_heads': num_heads,
                    'dropout_rate': 0.1,
                },
                input_shapes=[
                    (64, qa, qb),
                    (64, ma, mb, mc),
                ]
            )


if __name__ == "__main__":
    tf.test.main()
