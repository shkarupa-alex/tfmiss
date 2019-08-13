from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.utils import generic_utils


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

        super(WeightNorm, self).__init__(layer, **kwargs)
        self.input_spec = layer.input_spec
        self.supports_masking = layer.supports_masking

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
            norm_axes = list(range(v.shape.rank - 1))

            def g_init(*args, **kwargs):
                v_init = v.read_value()
                c = tf.shape(v_init)[-1]
                v_init = tf.reshape(v_init, [-1, c])

                return tf.norm(v_init, ord=2, axis=0)

            g = self.add_variable(
                name='{}_g'.format(name),
                shape=(v.shape[-1],),
                initializer=g_init,
                dtype=v.dtype,
                trainable=True,
                aggregation=tf.VariableAggregation.MEAN
            )

            setattr(self, '{}_v'.format(name), v)
            setattr(self, '{}_g'.format(name), g)
            setattr(self, '{}_norm_axes'.format(name), norm_axes)
            setattr(self.layer, name, None)

        super(WeightNorm, self).build()

    def compute_weights(self):
        for name in self.weight_names:
            v = getattr(self, '{}_v'.format(name))
            g = getattr(self, '{}_g'.format(name))
            norm_axes = getattr(self, '{}_norm_axes'.format(name))

            w = tf.nn.l2_normalize(v, axis=norm_axes) * g
            setattr(self.layer, name, w)

    def call(self, inputs, **kwargs):
        compute_weights = kwargs.pop('compute_weights', True)
        if compute_weights:
            self.compute_weights()

        for name in self.weight_names:
            if getattr(self.layer, name) is None:
                raise ValueError('You need to compute weights before using WeightNorm')

        layer_kwargs = {}
        for key in kwargs.keys():
            if generic_utils.has_arg(self.layer.call, key):
                layer_kwargs[key] = kwargs[key]

        return self.layer.call(inputs, **layer_kwargs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def get_config(self):
        config = super(WeightNorm, self).get_config()
        config['weight_names'] = self.weight_names

        return config
