from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper


@tf.keras.utils.register_keras_serializable(package='Miss')
def macro_soft_f1(y_true, y_pred, from_logits=False, double=True):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_true.shape.assert_is_compatible_with(y_pred.shape)

    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)

    tp = tf.reduce_sum(y_pred * y_true, axis=0)
    fp = tf.reduce_sum(y_pred * (1. - y_true), axis=0)
    fn = tf.reduce_sum((1. - y_pred) * y_true, axis=0)
    epsilon_ = tf.constant(tf.keras.backend.epsilon(), y_pred.dtype.base_dtype)

    loss = 2 * tp / (2 * tp + fn + fp + epsilon_)
    loss = 1 - loss

    if double:
        # reduce inverted class (0 -> 1) loss
        tn = tf.reduce_sum((1. - y_pred) * (1. - y_true), axis=0)
        loss_ = 2 * tn / (2 * tn + fn + fp + 1e-16)
        loss_ = 1 - loss_
        loss = (loss + loss_) / 2

    return loss


@tf.keras.utils.register_keras_serializable(package='Miss')
def binary_soft_f1(y_true, y_pred, from_logits=False, double=True):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    y_true.shape.assert_is_compatible_with(y_pred.shape)

    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)

    tp = y_pred * y_true
    fp = y_pred * (1. - y_true)
    fn = (1. - y_pred) * y_true
    epsilon_ = tf.constant(tf.keras.backend.epsilon(), y_pred.dtype.base_dtype)

    loss = 2 * tp / (2 * tp + fn + fp + epsilon_)
    loss = 1. - loss

    if double:
        # reduce inverted class (0 -> 1) loss
        tn = (1. - y_pred) * (1. - y_true)
        loss_ = 2 * tn / (2 * tn + fn + fp + epsilon_)
        loss_ = 1. - loss_
        loss = (loss + loss_) / 2

    return tf.reduce_mean(loss, axis=-1)


@tf.keras.utils.register_keras_serializable(package='Miss')
class MacroSoftF1(LossFunctionWrapper):
    """Computes macro soft F1 loss."""

    def __init__(self, double=True, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name='macro_soft_f1'):
        super(MacroSoftF1, self).__init__(
            macro_soft_f1, name=name, reduction=reduction, from_logits=from_logits, double=double)


@tf.keras.utils.register_keras_serializable(package='Miss')
class BinarySoftF1(LossFunctionWrapper):
    """Computes binary soft F1 loss."""

    def __init__(self, double=True, from_logits=False,
                 reduction=tf.keras.losses.Reduction.AUTO, name='binary_soft_f1'):
        super(BinarySoftF1, self).__init__(
            binary_soft_f1, name=name, reduction=reduction, from_logits=from_logits, double=double)
