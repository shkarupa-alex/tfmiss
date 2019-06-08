from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
# from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras import backend as K
from tfmiss.keras.layers.wrappers import WeightNorm


# class WeightShareConv1d(keras.layers.Layer):
#     # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
#     # Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
#     _STRIDES = 1
#
#     def __init__(self,
#                  filters,
#                  kernel_size,
#                  dilation,
#                  dropout,
#                  padding='causal',
#                  activation='relu',
#                  use_bias=True,
#                  kernel_initializer='glorot_uniform',
#                  bias_initializer='zeros',
#                  kernel_regularizer=None,
#                  bias_regularizer=None,
#                  activity_regularizer=None,
#                  kernel_constraint=None,
#                  bias_constraint=None,
#                  *args,
#                  **kwargs):
#         if padding not in {'causal', 'same'}:
#             raise ValueError('Only "causal" and "same" padding are compatible with this layer.')
#
#         self.filters = filters
#         self.kernel_size = kernel_size
#         self.dilation = dilation
#         self.dropout = dropout
#         self.padding = padding
#
#         self.activation = keras.activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = keras.initializers.get(kernel_initializer)
#         self.bias_initializer = keras.initializers.get(bias_initializer)
#         self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
#         self.bias_regularizer = keras.regularizers.get(bias_regularizer)
#         self.kernel_constraint = keras.constraints.get(kernel_constraint)
#         self.bias_constraint = keras.constraints.get(bias_constraint)
#
#         self.conv1d1 = None
#         self.conv1d2 = None
#         self.dropout1 = None
#         self.dropout2 = None
#         self.downsample = None
#         self.add = None
#         self.act = None
#
#         super(WeightShareConv1d, self).__init__(
#             activity_regularizer=keras.regularizers.get(activity_regularizer), *args, **kwargs)
#         self.input_spec = keras.layers.InputSpec(ndim=3)
#
#     # def build(self, input_shape):
#     #     if len(input_shape) != 3:
#     #         raise ValueError('Shape {} must have rank 3'.format(input_shape))
#     #
#     #     num_channels = input_shape[-1]
#     #     if num_channels is None:
#     #         raise ValueError('The last dimension of the inputs should be defined. Found `None`.')
#     #
#     #     self.input_spec = keras.layers.InputSpec(ndim=3, axes={-1: num_channels})
#     #
#     #     self.conv1d1 = WeightNorm(keras.layers.Conv1D(
#     #         filters=self.filters,
#     #         kernel_size=self.kernel_size,
#     #         strides=self._STRIDES,
#     #         padding=self.padding,
#     #         data_format='channels_last',
#     #         dilation_rate=self.dilation,
#     #         activation=self.activation,
#     #         use_bias=self.use_bias,
#     #         kernel_initializer=self.kernel_initializer,
#     #         bias_initializer=self.bias_initializer,
#     #         kernel_regularizer=self.kernel_regularizer,
#     #         bias_regularizer=self.bias_regularizer,
#     #         kernel_constraint=self.kernel_constraint,
#     #         bias_constraint=self.bias_constraint,
#     #     ))
#     #     self.conv1d2 = WeightNorm(keras.layers.Conv1D(
#     #         filters=self.filters,
#     #         kernel_size=self.kernel_size,
#     #         strides=self._STRIDES,
#     #         padding=self.padding,
#     #         data_format='channels_last',
#     #         dilation_rate=self.dilation,
#     #         activation=self.activation,
#     #         use_bias=self.use_bias,
#     #         kernel_initializer=self.kernel_initializer,
#     #         bias_initializer=self.bias_initializer,
#     #         kernel_regularizer=self.kernel_regularizer,
#     #         bias_regularizer=self.bias_regularizer,
#     #         kernel_constraint=self.kernel_constraint,
#     #         bias_constraint=self.bias_constraint,
#     #     ))
#     #
#     #     self.dropout1 = keras.layers.SpatialDropout1D(rate=self.dropout)
#     #     self.dropout2 = keras.layers.SpatialDropout1D(rate=self.dropout)
#     #
#     #     if num_channels != self.filters:
#     #         self.downsample = keras.layers.Conv1D(
#     #             self.filters,
#     #             kernel_size=1,
#     #             padding='valid',
#     #             data_format='channels_last',
#     #             activation=None,
#     #             use_bias=self.use_bias,
#     #             kernel_initializer=self.kernel_initializer,
#     #             bias_initializer=self.bias_initializer,
#     #             kernel_regularizer=self.kernel_regularizer,
#     #             bias_regularizer=self.bias_regularizer,
#     #             kernel_constraint=self.kernel_constraint,
#     #             bias_constraint=self.bias_constraint,
#     #         )
#     #
#     #     self.add = keras.layers.Add()
#     #     self.act = keras.layers.Activation(activation=self.activation)
#     #
#     #     super(TemporalBlock, self).build(input_shape)
#     #
#     # def call(self, inputs, training=None):
#     #     out = self.conv1d1(inputs)
#     #     out = self.dropout1(out, training=training)
#     #
#     #     out = self.conv1d2(out)
#     #     out = self.dropout2(out, training=training)
#     #
#     #     res = inputs if self.downsample is None else self.downsample(inputs)
#     #
#     #     out = self.add([out, res])
#     #     out = self.act(out)
#     #
#     #     return out
#     #
#     # def compute_output_shape(self, input_shape):
#     #     if len(input_shape) != 3:
#     #         raise ValueError('Shape {} must have rank 3'.format(input_shape))
#     #
#     #     return input_shape[:-1].concatenate(self.filters)
#     #
#     # def get_config(self):
#     #     config = {
#     #         'filters': self.filters,
#     #         'kernel_size': self.kernel_size,
#     #         'dilation': self.dilation,
#     #         'dropout': self.dropout,
#     #         'padding': self.padding,
#     #         'activation': keras.activations.serialize(self.activation),
#     #         'use_bias': self.use_bias,
#     #         'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
#     #         'bias_initializer': keras.initializers.serialize(self.bias_initializer),
#     #         'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
#     #         'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
#     #         'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
#     #         'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
#     #         'bias_constraint': keras.constraints.serialize(self.bias_constraint),
#     #     }
#     #     base_config = super(TemporalBlock, self).get_config()
#     #
#     #     return dict(list(base_config.items()) + list(config.items()))
#
#
class TrellisNet(keras.layers.Layer):
    def __init__(self, nhid, nout,
                 nlevels=40,
                 kernel_size=2,
                 dropout=0.0,
                 wnorm=True,
                 aux_frequency=20,
                 dilation=None,
                 # padding='causal',
                 # activation='relu',
                 # use_bias=True,
                 # kernel_initializer='glorot_uniform',
                 # bias_initializer='zeros',
                 # kernel_regularizer=None,
                 # bias_regularizer=None,
                 # activity_regularizer=None,
                 # kernel_constraint=None,
                 # bias_constraint=None,
                 *args, **kwargs):

        if dilation is None:
            dilation = [1]

        if not isinstance(dilation, (list, tuple)) or not len(dilation):
            raise ValueError('Number of residual layers could not be zero.')

        # if padding not in {'causal', 'same'}:
        #     raise ValueError('Only "causal" and "same" padding are compatible with this layer.')

        self.nhid = nhid
        self.nout = nout
        self.h_size = nhid + nout
        self.nlevels = nlevels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.wnorm = wnorm
        self.aux_frequency = aux_frequency
        self.dilation = dilation

        # self.activation = keras.activations.get(activation)
        # self.use_bias = use_bias
        # self.kernel_initializer = keras.initializers.get(kernel_initializer)
        # self.bias_initializer = keras.initializers.get(bias_initializer)
        # self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        # self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        # self.kernel_constraint = keras.constraints.get(kernel_constraint)
        # self.bias_constraint = keras.constraints.get(bias_constraint)

        super(TrellisNet, self).__init__(
            # activity_regularizer=keras.regularizers.get(activity_regularizer),
            *args, **kwargs)
        self.input_spec = keras.layers.InputSpec(ndim=3)

    def build(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        self.full_conv = WeightShareConv1d(
            hidden_dim=self.h_size,
            n_out=4 * self.h_size,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        )
        if self.wnorm:
            self.full_conv = WeightNorm(
                self.full_conv,
                weight_names=['weight1', 'weight2']
            )  # dim=0 ?


    #     num_channels = input_shape[-1]
    #     if num_channels is None:
    #         raise ValueError('The last dimension of the inputs should be defined. Found `None`.')
    #
    #     self.input_spec = keras.layers.InputSpec(ndim=3, axes={-1: num_channels})
    #
    #     self.layers = []
    #     num_levels = len(self.kernels)
    #     for i in range(num_levels):
    #         temporal_block = TemporalBlock(
    #             filters=self.kernels[i],
    #             kernel_size=self.kernel_size,
    #             dilation=2 ** i,
    #             dropout=self.dropout,
    #             padding=self.padding,
    #             activation=self.activation,
    #             use_bias=self.use_bias,
    #             kernel_initializer=self.kernel_initializer,
    #             bias_initializer=self.bias_initializer,
    #             kernel_regularizer=self.kernel_regularizer,
    #             bias_regularizer=self.bias_regularizer,
    #             kernel_constraint=self.kernel_constraint,
    #             bias_constraint=self.bias_constraint,
    #             dtype=self.dtype,
    #         )
    #         self.layers.append(temporal_block)
    #         setattr(self, 'temporal_block_{}'.format(i), temporal_block)
    #
    #     super(TemporalConvNet, self).build(input_shape)
    #

    def call(self, inputs, hc, aux=True, **kwargs):
        ninp = self.ninp
        nout = self.nout
        Z = self.transform_input(inputs)
        aux_outs = []
        dilation_cycle = self.dilation

        if self.fn is not None:
            # Recompute weight normalization weights
            self.fn.reset(self.full_conv)
        for key in self.full_conv.dict:
            # Clear the pre-computed computations
            if key[1] == inputs.get_device():
                self.full_conv.dict[key] = None
        self.full_conv.drop.reset_mask(Z[:, ninp:])

        # Feed-forward layers
        for i in range(0, self.nlevels):
            d = dilation_cycle[i % len(dilation_cycle)]
            Z = self.step(Z, dilation=d, hc=hc)
            if aux and i % self.aux_frequency == (self.aux_frequency - 1):
                aux_outs.append(Z[:, -nout:].unsqueeze(0))

        out = Z[:, -nout:].transpose(1, 2)  # Dimension (N, L, nout)
        hc = (Z[:, ninp:, -1:], self.ct[:, :, -1:])  # Dimension (N, h_size, L)
        if aux:
            aux_outs = torch.cat(aux_outs, dim=0).transpose(0, 1).transpose(2, 3)
        else:
            aux_outs = None

        return out, hc, aux_outs
    #
    # def compute_output_shape(self, input_shape):
    #     if len(input_shape) != 3:
    #         raise ValueError('Shape {} must have rank 3'.format(input_shape))
    #
    #     return input_shape[:-1].concatenate(self.kernels[-1])
    #
    # def get_config(self):
    #     config = {
    #         'kernels': self.kernels,
    #         'kernel_size': self.kernel_size,
    #         'dropout': self.dropout,
    #         'padding': self.padding,
    #         'activation': keras.activations.serialize(self.activation),
    #         'use_bias': self.use_bias,
    #         'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
    #         'bias_initializer': keras.initializers.serialize(self.bias_initializer),
    #         'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
    #         'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
    #         'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
    #         'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
    #         'bias_constraint': keras.constraints.serialize(self.bias_constraint),
    #     }
    #     base_config = super(TemporalConvNet, self).get_config()
    #
    #     return dict(list(base_config.items()) + list(config.items()))
