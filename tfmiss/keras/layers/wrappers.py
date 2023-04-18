from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import layers
from keras.saving import register_keras_serializable
from keras.src.backend import convert_inputs_if_ragged, maybe_convert_to_ragged
from keras.src.utils.generic_utils import has_arg
from keras.src.utils.tf_utils import shape_type_conversion


@register_keras_serializable(package='Miss')
class WithRagged(layers.Wrapper):
    """ Passes ragged tensor to layer that accepts only dense one.

    Arguments:
      layer: The `Layer` instance to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        super(WithRagged, self).__init__(layer, **kwargs)
        self.input_spec = layer.input_spec
        self.supports_masking = layer.supports_masking
        self._supports_ragged_inputs = True

        if not isinstance(layer, layers.Layer):
            raise ValueError(
                'Please initialize `WithRagged` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)

        self.input_spec = self.layer.input_spec

        zero = '' if self.layer.dtype == tf.string else 0
        self.masking_layer = layers.Masking(mask_value=zero)

        super(WithRagged, self).build()

    def call(self, inputs, **kwargs):
        layer_kwargs = {}
        for key in kwargs.keys():
            if has_arg(self.layer.call, key):
                layer_kwargs[key] = kwargs[key]

        inputs_dense, row_lengths = convert_inputs_if_ragged(inputs)
        inputs_dense = self.masking_layer(inputs_dense)
        outputs_dense = self.layer.call(inputs_dense, **layer_kwargs)
        outputs = maybe_convert_to_ragged(row_lengths is not None, outputs_dense, row_lengths)

        return outputs

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def get_config(self):
        return super(WithRagged, self).get_config()


@register_keras_serializable(package='Miss')
class MapFlat(layers.Wrapper):
    """ Calls layer on the flat values of ragged tensor.

    Arguments:
      layer: The `Layer` instance to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
        self._supports_ragged_inputs = True

    def build(self, input_shape=None):
        super(MapFlat, self).build([None])

    def call(self, inputs, **kwargs):
        return tf.ragged.map_flat_values(self.layer, inputs)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape + self.layer.compute_output_shape([None])[1:]


@register_keras_serializable(package='Miss')
class WeightNorm(layers.Wrapper):
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
        if hasattr(layer, '_supports_ragged_inputs'):
            self._supports_ragged_inputs = layer._supports_ragged_inputs

        if not isinstance(layer, layers.Layer):
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

    @shape_type_conversion
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

            def _read_g():
                return tf.identity(g)

            def _init_g():
                # Ensure we read `g` after _update_weights.
                assert_init = tf.debugging.assert_equal(
                    g_init, False, message='The layer already initialized')
                with tf.control_dependencies([assert_init]):
                    v_flat = tf.reshape(v, [-1, v_depth])
                    v_norm = tf.norm(v_flat, axis=0)
                    g_assign = g.assign(tf.reshape(v_norm, (v_depth,)))
                    g_init_assign = g_init.assign(True)

                    with tf.control_dependencies([g_assign, g_init_assign]):
                        return tf.identity(g)

            g = tf.cond(g_init, _read_g, _init_g)

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
            if has_arg(self.layer.call, key):
                layer_kwargs[key] = kwargs[key]

        # Ensure we calculate result after updating kernel.
        with tf.control_dependencies(kernel_updates):
            return self.layer.call(inputs, **layer_kwargs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def get_config(self):
        config = super(WeightNorm, self).get_config()
        config['weight_names'] = self.weight_names

        return config
