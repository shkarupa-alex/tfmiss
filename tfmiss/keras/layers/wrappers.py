from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.utils import generic_utils, tf_utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables


@tf.keras.utils.register_keras_serializable(package='Miss')
class WeightNorm(tf.keras.layers.Wrapper):
    """ Applies weight normalization to a layer. Weight normalization is a reparameterization that decouples the
    magnitude of a weight tensor from its direction. This speeds up convergence by improving the conditioning of the
    optimization problem.

    Reference: https://arxiv.org/abs/1602.07868
    Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
    Tim Salimans, Diederik P. Kingma (2016)

    Arguments:
      layer: The `Layer` instance to be wrapped.
      weight_name: name of weights `Tensor` (or list of names) in wrapped layer to be normalized
    """

    def __init__(self, layer, weight_names='kernel', **kwargs):
        super(WeightNorm, self).__init__(layer, **kwargs)
        self.input_spec = layer.input_spec
        self.supports_masking = layer.supports_masking
        self._supports_ragged_inputs = layer._supports_ragged_inputs

        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `WeightNorm` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))

        if not hasattr(layer, '_weight_norm') or not layer._weight_norm:
            layer._weight_norm = True
        else:
            raise ValueError(
                'Weight normalization already applied to layer {}'.format(layer))

        if not isinstance(weight_names, (list, tuple)):
            weight_names = [weight_names]
        self.weight_names = weight_names
        if not len(self.weight_names):
            raise ValueError('Weight names could not be empty.')

        for name in self.weight_names:
            setattr(self, '{}_v'.format(name), None)
            setattr(self, '{}_g'.format(name), None)
            setattr(self, '{}_norm_axes'.format(name), None)

    @tf_utils.shape_type_conversion
    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)

        self.input_spec = self.layer.input_spec

        for name in self.weight_names:
            if not hasattr(self.layer, name):
                raise ValueError(
                    'Weights with name '
                    '{} not found in layer {}'.format(name, self.layer))

            v = getattr(self.layer, name)
            setattr(self, '{}_v'.format(name), v)

            v_depth = int(v.shape[-1])
            setattr(self, '{}_v_depth'.format(name), v_depth)

            v_axes = list(range(v.shape.rank - 1))
            setattr(self, '{}_v_axes'.format(name), v_axes)

            g = self.add_weight(
                name='{}_g'.format(name),
                shape=(v.shape[-1],),
                initializer='ones',
                dtype=v.dtype,
                trainable=True,
            )
            setattr(self, '{}_g'.format(name), g)

            g_init = self.add_weight(
                name='{}_g_init'.format(name),
                shape=None,
                initializer='zeros',
                dtype=tf.dtypes.bool,
                trainable=False
            )
            setattr(self, '{}_g_init'.format(name), g_init)

            g_mutex = tf.CriticalSection(name='{}_g_mutex'.format(name))
            setattr(self, '{}_g_mutex'.format(name), g_mutex)

            setattr(self.layer, name, None)

        super(WeightNorm, self).build()

    def compute_weights(self):
        updates = []

        for name in self.weight_names:
            v = getattr(self, '{}_v'.format(name))
            v_depth = getattr(self, '{}_v_depth'.format(name))
            v_axes = getattr(self, '{}_v_axes'.format(name))
            g = getattr(self, '{}_g'.format(name))
            g_init = getattr(self, '{}_g_init'.format(name))
            g_mutex = getattr(self, '{}_g_mutex'.format(name))

            def _read_g():
                return tf.identity(g)

            def _init_g():
                # Ensure we read `g` after _update_weights.
                assert_init = tf.debugging.assert_equal(g_init, False, message='The layer already initialized')
                with tf.control_dependencies([assert_init]):
                    v_flat = tf.reshape(v, [-1, v_depth])
                    v_norm = tf.norm(v_flat, axis=0)
                    g_assign = g.assign(tf.reshape(v_norm, (v_depth,)))
                    g_init_assign = g_init.assign(True)

                    with tf.control_dependencies([g_assign, g_init_assign]):
                        return tf.identity(g)

            g = g_mutex.execute(lambda: tf.cond(g_init, _read_g, _init_g))

            # Replace kernel by normalized weight variable.
            w = tf.nn.l2_normalize(v, axis=v_axes) * g
            setattr(self.layer, name, w)
            updates.append(tf.identity(w))

        return updates

    def call(self, inputs, **kwargs):
        kernel_updates = []
        if kwargs.pop('compute_weights', True):
            kernel_updates = self.compute_weights()

        for name in self.weight_names:
            if getattr(self.layer, name) is None:
                raise ValueError('You need to compute weights before using WeightNorm')

        layer_kwargs = {}
        for key in kwargs.keys():
            if generic_utils.has_arg(self.layer.call, key):
                layer_kwargs[key] = kwargs[key]

        # Ensure we calculate result after updating kernel.
        with tf.control_dependencies(kernel_updates):
            return self.layer.call(inputs, **layer_kwargs)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def get_config(self):
        config = super(WeightNorm, self).get_config()
        config['weight_names'] = self.weight_names

        return config
