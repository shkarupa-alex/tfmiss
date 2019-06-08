from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.python.keras import backend as K


class WeightNorm(keras.layers.Wrapper):
    """ Applies weight normalization to a layer. Weight normalization is a reparameterization that decouples the
    magnitude of a weight tensor from its direction. This speeds up convergence by improving the conditioning of the
    optimization problem.

    Reference: https://arxiv.org/abs/1602.07868
    Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks
    Tim Salimans, Diederik P. Kingma (2016)

    Arguments:
      layer: The `Layer` instance to be wrapped.
      weight_name: name of weights `Tensor` in wrapped layer to be normalized
    """

    def __init__(self, layer, weight_names='kernel', **kwargs):
        if not isinstance(layer, keras.layers.Layer):
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

        self.v = None
        self.norm_axes = None
        self.g = None

        super(WeightNorm, self).__init__(layer, **kwargs)
        self.supports_masking = layer.supports_masking
        self.input_spec = layer.input_spec

    def build(self, input_shape=None):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)

        if not self.layer.built:
            self.layer.build(input_shape)

        for name in self.weight_names:
            if not hasattr(self.layer, name):
                raise ValueError(
                    'Weights with name '
                    '{} not found in layer {}'.format(name, self.layer))

            v = getattr(self.layer, name)
            norm_axes = list(range(v.shape.ndims - 1))

            def g_init(*args, **kwargs):
                v_init = v.read_value()
                c = K.shape(v_init)[-1]
                v_init = K.reshape(v_init, [-1, c])

                return K.linalg_ops.norm(v_init, ord=2, axis=0)

            g = self.add_variable(
                name='{}_g'.format(name),
                shape=(v.shape[-1],),
                initializer=g_init,
                dtype=v.dtype,
                trainable=True,
                aggregation=K.variables_module.VariableAggregation.MEAN
            )

            setattr(self, '{}_v'.format(name), v)
            setattr(self, '{}_g'.format(name), g)
            setattr(self, '{}_norm_axes'.format(name), norm_axes)

        super(WeightNorm, self).build()

    def call(self, inputs, **kwargs):
        for name in self.weight_names:
            v = getattr(self, '{}_v'.format(name))
            g = getattr(self, '{}_g'.format(name))
            norm_axes = getattr(self, '{}_norm_axes'.format(name))

            w = K.nn.l2_normalize(v, axis=norm_axes) * g
            setattr(self.layer, name, w)

        return self.layer.call(inputs, **kwargs)

    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)

    def compute_mask(self, inputs, mask=None):
        return self.layer.compute_mask(inputs, mask)

    def get_config(self):
        config = super(WeightNorm, self).get_config()
        config['weight_names'] = self.weight_names

        return config
