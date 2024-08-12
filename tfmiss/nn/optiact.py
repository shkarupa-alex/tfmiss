# Adapted from https://github.com/PgLoLo/optiacts
# TODO: check memory consumption
import numpy as np
import tensorflow as tf


def _unpack_bool(packed, shape, size):
    with tf.name_scope("unpack_bool"):
        mask = (
            tf.bitwise.right_shift(
                packed[..., None], np.arange(8, dtype="uint8")
            )
            & 1
        )
        mask = tf.reshape(mask, [-1])[:size]
        mask = tf.cast(mask, "bool")
        mask = tf.reshape(mask, shape)

        return mask


def _pack_bool(mask):
    with tf.name_scope("pack_bool"):
        shape = tf.shape(mask)
        size = tf.size(mask)

        packed = tf.reshape(mask, [-1])
        packed = tf.pad(packed, [[0, (8 - size % 8) % 8]])
        packed = tf.reshape(packed, [-1, 8])
        packed = tf.cast(packed, "uint8")
        packed = tf.bitwise.left_shift(packed, np.arange(8, dtype="uint8"))
        packed = tf.reduce_sum(packed, axis=-1)

        return packed, shape, size


def _gelu_left(y):
    with tf.name_scope("gelu_left"):
        y_min = -0.16997120747990369
        y_delta = 0.16997120747990369
        alpha = 0.44236329315571304
        beta = 0.8312705290392689
        a = -0.2964121012745849
        b = 0.0012176597919768537

        y = tf.clip_by_value((y - y_min) * (1 / y_delta), 0, 1)
        g = b + a * y**alpha * (1 - y) ** beta

        return g


def _gelu_right(y):
    with tf.name_scope("gelu_right"):
        y_min = -0.16997120747990369
        exp_mean = -2.1198850898439496
        exp_std = 0.03214673676937606
        bias = -1.383717971214795
        sqrt = 1.5584201843500274
        linear = 0.04404574801811055

        y = tf.maximum(y - y_min, 0)
        exp = tf.exp((exp_mean - y) ** 3 * exp_std)
        poly = bias + sqrt * tf.sqrt(y) + linear * y
        g = 1 + poly * exp

        return g


@tf.custom_gradient
def gelu(features, name=None):
    with tf.name_scope(name or "gelu"):
        features = tf.stop_gradient(features)
        min_point = -0.751791524693564457457904947
        is_left_packed, shape, size = _pack_bool(features < min_point)

        y = tf.nn.gelu(features)

        def grad(upstream):
            with tf.name_scope(f'{name or "gelu"}_grad'):
                is_left_mask = _unpack_bool(is_left_packed, shape, size)
                upstream *= tf.where(
                    is_left_mask, _gelu_left(y), _gelu_right(y)
                )

                return upstream

        return y, grad


def _silu_left(y):
    y_min = -0.2784645427610738
    sqrt = -0.5076843705082636
    poly0 = 0.3574942048593756
    poly1 = 0.07963139766917564
    poly2 = 0.21717700759576886

    y -= y_min
    g = sqrt * tf.sqrt(y) + (poly0 * y + poly1) * y + poly2

    return g


def _silu_right(y):
    y_min = -0.2784645427610738
    exp_mean = -5.770613302664509
    exo_std = 0.0026961639850448835
    bias = -1.3108564021309803
    sqrt = 0.8485896470316523
    linear = -0.16299051259510922

    y -= y_min
    exp = tf.exp((exp_mean - y) ** 3 * exo_std)
    poly = bias + sqrt * tf.sqrt(y) + linear * y
    g = 1 + poly * exp

    return g


@tf.custom_gradient
def silu(features, name=None):
    with tf.name_scope(name or "silu"):
        features = tf.stop_gradient(features)
        min_point = -1.278464542761073795109358739022980155439
        is_left_packed, shape, size = _pack_bool(features < min_point)
        is_left_packed = tf.stop_gradient(is_left_packed)

        y = tf.nn.silu(features)

        def grad(upstream):
            is_left_mask = _unpack_bool(is_left_packed, shape, size)
            upstream *= (
                tf.where(is_left_mask, _silu_left(y), _silu_right(y)) * (1 - y)
                + y
            )

            return upstream

        return y, grad
