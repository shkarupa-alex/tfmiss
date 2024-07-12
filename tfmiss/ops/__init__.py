import os
import tensorflow as tf

tfmiss_ops = tf.load_op_library(os.path.join(os.path.dirname(os.path.abspath(__file__)), '_tfmiss_ops.so'))

# image
tf.no_gradient('Miss>ConnectedComponents')
tf.no_gradient('Miss>EuclideanDistance')

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
        column=op.outputs[0],
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
@tf.RegisterGradient('Miss>FoPool')
def _fo_pool_grad(op, grad):
    return tfmiss_ops.miss_fo_pool_backward(input=op.inputs[0], forget=op.inputs[1], hidden=op.outputs[0], grad=grad)


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
