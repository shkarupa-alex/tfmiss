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


# dcn v2
@tf.RegisterGradient('Miss>ModulatedDeformableColumn')
def _modulated_deformable_column_grad(op, grad):
    return tfmiss_ops.miss_modulated_deformable_column_backward(
        input=op.inputs[0],
        offset=op.inputs[1],
        mask=op.inputs[2],
        grad=grad,
        kernel_h=op.get_attr('kernel_h'),
        kernel_w=op.get_attr('kernel_w'),
        stride_h=op.get_attr('stride_h'),
        stride_w=op.get_attr('stride_w'),
        pad_hb=op.get_attr('pad_hb'),
        pad_ha=op.get_attr('pad_ha'),
        pad_wb=op.get_attr('pad_wb'),
        pad_wa=op.get_attr('pad_wa'),
        dilation_h=op.get_attr('dilation_h'),
        dilation_w=op.get_attr('dilation_w'),
        deformable_group=op.get_attr('deformable_group')
    )


# qrnn
@tf.RegisterGradient('Miss>TimeMajorFoPool')
def _time_major_bwd_fo_pool_grad(op, grad):
    # TODO: same input for h and x?
    return tfmiss_ops.miss_time_major_bwd_fo_pool(h=op.outputs[0], x=op.inputs[0], forget=op.inputs[1], gh=grad)


@tf.RegisterGradient('Miss>BatchMajorFoPool')
def _batch_major_bwd_fo_pool_grad(op, grad):
    return tfmiss_ops.miss_batch_major_bwd_fo_pool(h=op.outputs[0], x=op.inputs[0], forget=op.inputs[1], gh=grad)


# unicode transform
tf.no_gradient('Miss>CharCategory')
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
