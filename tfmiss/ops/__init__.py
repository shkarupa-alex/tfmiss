from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

tfmiss_ops = tf.load_op_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tfmiss_ops.so'))

# preprocessing
tf.no_gradient('Miss>SampleMask')
tf.no_gradient('Miss>SkipGram')
tf.no_gradient('Miss>ContBow')


# qrnn
@tf.RegisterGradient('Miss>TimeMajorFoPool')
def _fo_pool_time_grad(op, grad):
    return tfmiss_ops.miss_time_major_bwd_fo_pool(h=op.outputs[0], x=op.inputs[0], forget=op.inputs[1], gh=grad)


@tf.RegisterGradient('Miss>BatchMajorFoPool')
def _fo_pool_batch_grad(op, grad):
    return tfmiss_ops.miss_batch_major_bwd_fo_pool(h=op.outputs[0], x=op.inputs[0], forget=op.inputs[1], gh=grad)


# unicode transform
tf.no_gradient('Miss>LowerCase')
tf.no_gradient('Miss>NormalizeUnicode')
tf.no_gradient('Miss>ReplaceRegex')
tf.no_gradient('Miss>ReplaceString')
tf.no_gradient('Miss>TitleCase')
tf.no_gradient('Miss>UpperCase')
tf.no_gradient('Miss>WrapWith')
tf.no_gradient('Miss>ZeroDigits')

# unicode expand
tf.no_gradient('Miss>CharNgrams')
tf.no_gradient('Miss>SplitWords')
tf.no_gradient('Miss>SplitChars')
