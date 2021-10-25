from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras.utils.conv_utils import normalize_tuple
from tfmiss.ops import tfmiss_ops


def modulated_deformable_column(
        inputs, offset, mask, kernel_size, strides, padding, dilation_rate, deformable_groups, name=None):
    """Samples intermediate "column" values for DCNv2 layer

    Args:
        inputs: Tensor, input features with shape [N, Hin, Win, C]
        offset: Tensor, offset values with shape [N, Hout, Wout, deformable_group * kernel_height * kernel_width * 2]
        mask: Tensor, mask values with shape [N, Hout, Wout, deformable_group * kernel_height * kernel_width]
        kernel_size: An integer or tuple/list of 2 integers, the height and width of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers, the strides of the convolution along the height and width.
        padding: An integer or tuple/list of 2 integers, the explicit paddings for the input along the height and width.
        dilation_rate: An integer or tuple/list of 2 integers, the dilation rate to use for dilated convolution.
        deformable_groups: A positive integer, the number of groups in which the input is split along the channel axis.
            Each group uses it's own offset and mask values.
        name: A name for the operation (optional).
    Returns:
        Tensor: values of shape [N, Hout * Wout, C * kernel_height * kernel_width] sampled from input with respect to
            offset, mask and other convolution parameters.
    """
    with tf.name_scope(name or 'modulated_deformable_column'):
        inputs = tf.convert_to_tensor(inputs)
        offset = tf.convert_to_tensor(offset, dtype=inputs.dtype)
        mask = tf.convert_to_tensor(mask, dtype=inputs.dtype)

        kernel_size = normalize_tuple(kernel_size, 2, 'kernel_size')
        strides = normalize_tuple(strides, 2, 'strides')
        padding = normalize_tuple(padding, 4, 'padding')
        dilation_rate = normalize_tuple(dilation_rate, 2, 'dilation_rate')

        outputs = tfmiss_ops.miss_modulated_deformable_column(
            input=inputs,
            offset=offset,
            mask=mask,
            kernel_h=kernel_size[0],
            kernel_w=kernel_size[1],
            stride_h=strides[0],
            stride_w=strides[1],
            pad_hb=padding[0],
            pad_ha=padding[1],
            pad_wb=padding[2],
            pad_wa=padding[3],
            dilation_h=dilation_rate[0],
            dilation_w=dilation_rate[1],
            deformable_group=deformable_groups)

    return outputs
