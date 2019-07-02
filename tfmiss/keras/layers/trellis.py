from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.python.keras import backend as K
from tfmiss.keras.layers.wrappers import WeightNorm


class VariationalDropout(keras.layers.SpatialDropout1D):
    """ Given the relationship between TrellisNets and RNNs, the name `VariationalDropout`
    is just to remind the users that, once the dropout mask is created, this `SpatialDropout1D`
    should not only be applied on the sequence of a single layer, but on all layers of the deep network.

    In contrast, typically the `SpatialDropout1D` mask is different at each layer of a ConvNet.
    """

    def __init__(self, *args, **kwargs):
        super(VariationalDropout, self).__init__(*args, **kwargs)
        self.mask = None

    def reset_mask(self, inputs):
        self.mask = K.nn.dropout(
            K.ones_like(inputs),
            noise_shape=self._get_noise_shape(inputs),
            seed=self.seed,
            rate=self.rate
        )

    def call(self, inputs, reset_mask=True, training=None):
        if reset_mask:
            self.reset_mask(inputs)

        if self.mask is None:
            raise ValueError('You should reset mask before using `VariationalDropout`')

        def dropped_inputs():
            return inputs * self.mask

        return K.in_train_phase(dropped_inputs, inputs, training=training)


class WeightShareConv1D(keras.layers.Layer):
    # https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D
    # Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
    _STRIDES = 1

    def __init__(self,
                 n_out,
                 kernel_size,
                 dropout,
                 padding='causal',
                 data_format='channels_last',
                 activation='relu',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 *args,
                 **kwargs):
        padding = padding.lower()
        # Only causal padding support for now
        if padding not in {'causal'}:
            raise ValueError('Only "causal" padding is compatible with this layer.')
        # if padding not in {'causal', 'same'}:
        #     raise ValueError('Only "causal" and "same" padding are compatible with this layer.')

        self.n_out = n_out
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.padding = padding
        self.data_format = data_format

        self.activation = keras.activations.get(activation)
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)

        self.op_padding = 'valid' if self.padding == 'causal' else self.padding
        self.op_data_format = 'NWC' if data_format == 'channels_last' else 'NCW'
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1
        self.input_dim = None
        self.hidden_dim = None
        self.kernel1 = None
        self.kernel2 = None
        self.bias2 = None
        self.drop = None
        self.add = None
        self.cache = {}

        super(WeightShareConv1D, self).__init__(
            activity_regularizer=keras.regularizers.get(activity_regularizer), *args, **kwargs)
        self.input_spec = {
            'input': keras.layers.InputSpec(ndim=3),
            'hidden': keras.layers.InputSpec(ndim=3),
        }
        # self.supports_masking = True # TODO

    def build(self, input_shape):
        if not isinstance(input_shape, dict) or set(input_shape.keys()) != {'input', 'hidden'}:
            raise ValueError('A `WeightShareConv` layer should be called on a dict with `input` and `hidden` values.')
        inp_shape, hid_shape = input_shape['input'], input_shape['hidden']

        if len(inp_shape) != 3:
            raise ValueError('Shape of `input` {} must have rank 3'.format(inp_shape))
        self.input_dim = inp_shape[self.channel_axis]
        if self.input_dim is None:
            raise ValueError('Channel dimension of `input` should be defined. Found `None`.')

        if len(hid_shape) != 3:
            raise ValueError('Shape of `hidden` {} must have rank 3'.format(hid_shape))
        self.hidden_dim = hid_shape[self.channel_axis]
        if self.hidden_dim is None:
            raise ValueError('Channel dimension of `hidden` should be defined. Found `None`.')

        self.input_spec = {
            'input': keras.layers.InputSpec(ndim=3, axes={self.channel_axis: self.input_dim}),
            'hidden': keras.layers.InputSpec(ndim=3, axes={self.channel_axis: self.hidden_dim}),
        }

        self.kernel1 = self.add_weight(
            name='kernel1',
            shape=self.kernel_size + (self.input_dim, self.n_out),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)

        self.kernel2 = self.add_weight(
            name='kernel2',
            shape=self.kernel_size + (self.hidden_dim, self.n_out),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype)
        self.bias2 = self.add_weight(
            name='bias2',
            shape=(self.n_out,),
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            trainable=True,
            dtype=self.dtype)

        self.drop = VariationalDropout(rate=self.dropout)
        self.add = keras.layers.Add()

        super(WeightShareConv1D, self).build(input_shape)

    def compute_causal_padding(self, dilation_rate):
        left_pad = dilation_rate * (self.kernel_size - 1)

        if self.data_format == 'channels_last':
            causal_padding = [[0, 0], [left_pad, 0], [0, 0]]
        else:
            causal_padding = [[0, 0], [0, 0], [left_pad, 0]]

        return causal_padding, left_pad

    def clear_cache(self):
        for key in self.cache.keys():
            self.cache[key] = None

    def call(self, inputs, training=None, dilation=1, hid=None, **kwargs):
        clear_cache = kwargs.pop('clear_cache', True)
        if clear_cache:
            self.clear_cache()

        reset_mask = kwargs.pop('reset_mask', True)
        # if not isinstance(inputs, dict):
        #     raise ValueError('A merge layer should be called on a list of inputs.')

        left_pad = 0
        if self.padding == 'causal':
            causal_padding, left_pad = self.compute_causal_padding(dilation)
            inputs = K.array_ops.pad(inputs, causal_padding)

        if self.data_format == 'channel_last':
            x_1 = inputs[:, :, :self.input_dim]  # Input part

            z_1 = inputs[:, :, self.input_dim:]  # Hidden part
            z_1 = K.concatenate([  # Note: we only pad the hidden part :-)
                z_1[:, :left_pad, :],
                K.tile(hid, [1, left_pad, 1])
            ], axis=self.channel_axis)
        else:
            x_1 = inputs[:, :self.input_dim]  # Input part

            z_1 = inputs[:, self.input_dim:]  # Hidden part
            z_1 = K.concatenate([  # Note: we only pad the hidden part :-)
                z_1[:, :, :left_pad],
                K.tile(hid, [1, 1, left_pad])
            ], axis=self.channel_axis)

        if dilation not in self.cache or self.cache[dilation] is None:
            self.cache[dilation] = K.nn.conv1d_v2(
                input=x_1,
                filters=self.weight1,
                stride=self._STRIDES,
                padding=self.op_padding,
                data_format=self.op_data_format,
                dilations=dilation
            )

        out = K.nn.conv1d_v2(
            input=self.drop(z_1, reset_mask=reset_mask),
            filters=self.weight2,
            stride=self._STRIDES,
            padding=self.op_padding,
            data_format=self.op_data_format,
            dilations=dilation
        )
        out = K.nn.bias_add(out, self.bias2, data_format=self.op_data_format)

        return self.add(self.cache[dilation], out)

    def compute_output_shape(self, input_shape):
        if len(input_shape) != 3:
            raise ValueError('Shape {} must have rank 3'.format(input_shape))

        if self.data_format == 'channels_last':
            return input_shape[:-1].concatenate(self.n_out)
        else:
            return input_shape[:-2].concatenate(self.n_out).concatenate(input_shape[-1:])

    # def get_config(self):
    #     config = {
    #         'filters': self.filters,
    #         'kernel_size': self.kernel_size,
    #         'dilation': self.dilation,
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
    #     base_config = super(TemporalBlock, self).get_config()
    #
    #     return dict(list(base_config.items()) + list(config.items()))


class TrellisNet(keras.layers.Layer):
    def __init__(self, ninp, nhid, nout, nlevels=40, kernel_size=2, dropouth=0.0,
                 wnorm=True, aux_frequency=20, dilation=1):
        """
        Build a trellis network with LSTM-style gated activations
        :param ninp: The input (e.g., embedding) dimension
        :param nhid: The hidden unit dimension (excluding the output dimension). In other words, if you want to build
                     a TrellisNet with hidden size 1000 and output size 400, you should set nhid = 1000-400 = 600.
                     (The reason we want to separate this is from Theorem 1.)
        :param nout: The output dimension
        :param nlevels: Number of layers
        :param kernel_size: Kernel size of the TrellisNet
        :param dropouth: Hidden-to-hidden (VD-based) dropout rate
        :param wnorm: A boolean indicating whether to use weight normalization
        :param aux_frequency: Frequency of taking the auxiliary loss; (-1 means no auxiliary loss)
        :param dilation: The dilation of the convolution operation in TrellisNet
        """
        super(TrellisNet, self).__init__()
        if not isinstance(dilation, (list, tuple)):
            dilation = [dilation]
        self.ninp = ninp
        self.nhid = nhid
        self.nout = nout
        self.h_size = h_size = nhid + nout
        self.dilation = dilation
        self.nlevels = nlevels
        self.fn = None
        self.last_output = None
        self.aux_frequency = aux_frequency

        self.kernel_size = ker = kernel_size
        self.wnorm = wnorm

    def build(self, input_shape):
        # if not isinstance(input_shape, dict) or set(input_shape.keys()) != {'input', 'hidden'}:
        #     raise ValueError('A `WeightShareConv` layer should be called on a dict with `input` and `hidden` values.')
        # inp_shape, hid_shape = input_shape['input'], input_shape['hidden']

        full_conv = WeightShareConv1D(ninp, h_size, 4 * h_size, kernel_size=kernel_size, dropouth=dropouth)
        if self.wnorm:
            full_conv = WeightNorm(full_conv, weight_names=['weight1', 'weight2'])

        self.full_conv = full_conv

        super(TrellisNet, self).build(input_shape)

    def transform_input(self, X):
        # X has dimension (N, ninp, L)
        batch_size = X.size(0)
        seq_len = X.size(2)
        h_size = self.h_size

        self.ht = torch.zeros(batch_size, h_size, seq_len).cuda()
        self.ct = torch.zeros(batch_size, h_size, seq_len).cuda()
        return torch.cat([X] + [self.ht], dim=1)  # "Injecting" input sequence at layer 1

    def step(self, first, Z, dilation=1, hc=None):
        ninp = self.ninp
        h_size = self.h_size
        (hid, cell) = hc

        # Apply convolution
        conv_kwargs = self.full_conv_kwargs(first)
        out = self.full_conv(Z, dilation=dilation, hid=hid, **conv_kwargs)

        # Gated activations among channel groups
        ct_1 = F.pad(self.ct, (dilation, 0))[:, :, :-dilation]  # Dimension (N, h_size, L)
        ct_1[:, :, :dilation] = cell.repeat(1, 1, dilation)

        it = torch.sigmoid(out[:, :h_size])
        ot = torch.sigmoid(out[:, h_size: 2 * h_size])
        gt = torch.tanh(out[:, 2 * h_size: 3 * h_size])
        ft = torch.sigmoid(out[:, 3 * h_size: 4 * h_size])
        ct = ft * ct_1 + it * gt
        ht = ot * torch.tanh(ct)

        # Put everything back to form Z (i.e., injecting input to hidden unit)
        Z = torch.cat([Z[:, :ninp], ht], dim=1)
        self.ct = ct
        return Z

    def full_conv_kwargs(self, first_step=True):
        conv_kwargs = {
            'clear_cache': first_step,  # Clear the pre-computed computations
            'reset_mask': first_step,  # Recompute dropout mask
        }
        if self.wnorm:
            conv_kwargs['compute_weights'] = first_step  # Recompute weight normalization weights

        return conv_kwargs

    def forward(self, X, hc, aux=True):
        ninp = self.ninp
        nout = self.nout
        Z = self.transform_input(X)
        aux_outs = []
        dilation_cycle = self.dilation

        # Feed-forward layers
        for i in range(0, self.nlevels):
            d = dilation_cycle[i % len(dilation_cycle)]
            Z = self.step(
                first=i==0,
                Z=Z, dilation=d, hc=hc
            )
            if aux and i % self.aux_frequency == (self.aux_frequency - 1):
                aux_outs.append(Z[:, -nout:].unsqueeze(0))

        out = Z[:, -nout:].transpose(1, 2)  # Dimension (N, L, nout)
        hc = (Z[:, ninp:, -1:], self.ct[:, :, -1:])  # Dimension (N, h_size, L)
        aux_outs = torch.cat(aux_outs, dim=0).transpose(0, 1).transpose(2, 3) if aux else None

        return out, hc, aux_outs

# class WeightDrop(torch.nn.Module):
#     def __init__(self, module, weights, dropout=0, temporal=False):
#         """
#         Weight DropConnect, adapted from a recurrent setting by Merity et al. 2017
#         :param module: The module whose weights are to be applied dropout on
#         :param weights: A 2D list identifying the weights to be regularized. Each element of weights should be a
#                         list containing the "path" to the weight kernel. For instance, if we want to regularize
#                         module.layer2.weight3, then this should be ["layer2", "weight3"].
#         :param dropout: The dropout rate (0 means no dropout)
#         :param temporal: Whether we apply DropConnect only to the temporal parts of the weight (empirically we found
#                          this not very important)
#         """
#         super(WeightDrop, self).__init__()
#         self.module = module
#         self.weights = weights
#         self.dropout = dropout
#         self.temporal = temporal
#         self._setup()
#
#     def _setup(self):
#         for path in self.weights:
#             full_name_w = '.'.join(path)
#
#             module = self.module
#             name_w = path[-1]
#             for i in range(len(path) - 1):
#                 module = getattr(module, path[i])
#             w = getattr(module, name_w)
#             del module._parameters[name_w]
#             module.register_parameter(name_w + '_raw', Parameter(w.data))
#
#     def _setweights(self):
#         for path in self.weights:
#             module = self.module
#             name_w = path[-1]
#             for i in range(len(path) - 1):
#                 module = getattr(module, path[i])
#             raw_w = getattr(module, name_w + '_raw')
#
#             if len(raw_w.size()) > 2 and raw_w.size(2) > 1 and self.temporal:
#                 # Drop the temporal parts of the weight; if 1x1 convolution then drop the whole kernel
#                 w = torch.cat([F.dropout(raw_w[:, :, :-1], p=self.dropout, training=self.training),
#                                raw_w[:, :, -1:]], dim=2)
#             else:
#                 w = F.dropout(raw_w, p=self.dropout, training=self.training)
#
#             setattr(module, name_w, w)
#
#     def forward(self, *args, **kwargs):
#         self._setweights()
#         return self.module.forward(*args, **kwargs)
#
#
# ##############################################################################################################
# #
# # Embedding dropout
# #
# ##############################################################################################################
#
# def embedded_dropout(embed, words, dropout=0.1, scale=None):
#     """
#     Apply embedding encoder (whose weight we apply a dropout)
#     :param embed: The embedding layer
#     :param words: The input sequence
#     :param dropout: The embedding weight dropout rate
#     :param scale: Scaling factor for the dropped embedding weight
#     :return: The embedding output
#     """
#     if dropout:
#         mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(
#             embed.weight) / (1 - dropout)
#         masked_embed_weight = mask * embed.weight
#     else:
#         masked_embed_weight = embed.weight
#
#     if scale:
#         masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight
#
#     padding_idx = embed.padding_idx
#     if padding_idx is None:
#         padding_idx = -1
#
#     # Handle PyTorch issue
#     if '0.3' not in torch.__version__:
#         X = F.embedding(
#             words, masked_embed_weight,
#             padding_idx,
#             embed.max_norm, embed.norm_type,
#             embed.scale_grad_by_freq, embed.sparse
#         )
#     else:
#         X = embed._backend.Embedding.apply(words, masked_embed_weight,
#                                            padding_idx, embed.max_norm, embed.norm_type,
#                                            embed.scale_grad_by_freq, embed.sparse
#                                            )
#     return X
#
#
# ##############################################################################################################
# #
# # Variational dropout (for input/output layers, and for hidden layers)
# #
# ##############################################################################################################
#
#
# ##############################################################################################################
# #
# # Weight normalization. Modified from the original PyTorch's implementation of weight normalization.
# #
# ##############################################################################################################
#
# def _norm(p, dim):
#     """Computes the norm over all dimensions except dim"""
#     if dim is None:
#         return p.norm()
#     elif dim == 0:
#         output_size = (p.size(0),) + (1,) * (p.dim() - 1)
#         return p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
#     elif dim == p.dim() - 1:
#         output_size = (1,) * (p.dim() - 1) + (p.size(-1),)
#         return p.contiguous().view(-1, p.size(-1)).norm(dim=0).view(*output_size)
#     else:
#         return _norm(p.transpose(0, dim), 0).transpose(0, dim)
#
#
# class WeightNorm(object):
#     def __init__(self, names, dim):
#         """
#         Weight normalization module
#         :param names: The list of weight names to apply weightnorm on
#         :param dim: The dimension of the weights to be normalized
#         """
#         self.names = names
#         self.dim = dim
#
#     def compute_weight(self, module, name):
#         g = getattr(module, name + '_g')
#         v = getattr(module, name + '_v')
#         return v * (g / _norm(v, self.dim))
#
#     @staticmethod
#     def apply(module, names, dim):
#         fn = WeightNorm(names, dim)
#
#         for name in names:
#             weight = getattr(module, name)
#
#             # remove w from parameter list
#             del module._parameters[name]
#
#             # add g and v as new parameters and express w as g/||v|| * v
#             module.register_parameter(name + '_g', Parameter(_norm(weight, dim).data))
#             module.register_parameter(name + '_v', Parameter(weight.data))
#             setattr(module, name, fn.compute_weight(module, name))
#
#         # recompute weight before every forward()
#         module.register_forward_pre_hook(fn)
#         return fn
#
#     def remove(self, module):
#         for name in self.names:
#             weight = self.compute_weight(module, name)
#             delattr(module, name)
#             del module._parameters[name + '_g']
#             del module._parameters[name + '_v']
#             module.register_parameter(name, Parameter(weight.data))
#
#     def reset(self, module):
#         for name in self.names:
#             setattr(module, name, self.compute_weight(module, name))
#
#     def __call__(self, module, inputs):
#         # Typically, every time the module is called we need to recompute the weight. However,
#         # in the case of TrellisNet, the same weight is shared across layers, and we can save
#         # a lot of intermediate memory by just recomputing once (at the beginning of first call).
#         pass
#
#
# def weight_norm(module, names, dim=0):
#     fn = WeightNorm.apply(module, names, dim)
#     return module, fn
#
#

# class MixSoftmax(nn.Module):
#     def __init__(self, n_components, n_classes, nlasthid, ninp, decoder, dropoutl):
#         """
#         Apply mixture of softmax on the last layer of the network
#         :param n_components: The number of softmaxes to use
#         :param n_classes: The number of classes to predict
#         :param nlasthid: The dimension of the last hidden layer from the deep network
#         :param ninp: The embedding size
#         :param decoder: The decoder layer
#         :param dropoutl: The dropout to be applied on the pre-softmax output
#         """
#         super(MixSoftmax, self).__init__()
#         self.n_components = n_components
#         self.n_classes = n_classes
#         self.prior = nn.Linear(nlasthid, n_components)  # C ---> m
#         self.latent = nn.Linear(nlasthid, n_components * ninp)  # C ---> m*C
#         self.decoder = decoder
#         self.var_drop = VariationalDropout()
#         self.ninp = ninp
#         self.nlasthid = nlasthid
#         self.dropoutl = dropoutl
#
#     def init_weights(self):
#         initrange = 0.1
#         self.prior.weight.data.uniform_(-initrange, initrange)
#         self.latent.weight.data.uniform_(-initrange, initrange)
#
#     def forward(self, context):
#         n_components = self.n_components
#         n_classes = self.n_classes
#         decoder = self.decoder
#         ninp = self.ninp
#         dim = len(context.size())
#
#         if dim == 4:
#             # context: (N, M, L, C)  (used for the auxiliary outputs)
#             batch_size = context.size(0)
#             aux_size = context.size(1)
#             seq_len = context.size(2)
#             priors = F.softmax(self.prior(context), dim=3).view(-1, n_components)  # (M*N*L, m)
#             latent = self.var_drop(self.latent(context), self.dropoutl, dim=4)
#             latent = F.softmax(decoder(F.tanh(latent.view(-1, n_components, ninp))), dim=2)
#             return (priors.unsqueeze(2).expand_as(latent) * latent).sum(1).view(batch_size, aux_size, seq_len,
#                                                                                 n_classes)
#         else:
#             batch_size = context.size(0)
#             seq_len = context.size(1)
#             priors = F.softmax(self.prior(context), dim=2).view(-1, n_components)  # (N*L, m)
#             latent = self.var_drop(self.latent(context), self.dropoutl)
#             latent = F.softmax(decoder(F.tanh(latent.view(-1, n_components, ninp))), dim=2)  # (N*L, m, n_classes)
#             return (priors.unsqueeze(2).expand_as(latent) * latent).sum(1).view(batch_size, seq_len, n_classes)
#
#
# class TrellisNetModel(nn.Module):
#     def __init__(self, ntoken, ninp, nhid, nout, nlevels, kernel_size=2, dilation=[1],
#                  dropout=0.0, dropouti=0.0, dropouth=0.0, dropoutl=0.0, emb_dropout=0.0, wdrop=0.0,
#                  temporalwdrop=True, tie_weights=True, repack=False, wnorm=True, aux=True, aux_frequency=20,
#                  n_experts=0,
#                  load=""):
#         """
#         A deep sequence model based on TrellisNet
#         :param ntoken: The number of unique tokens
#         :param ninp: The input dimension
#         :param nhid: The hidden unit dimension (excluding the output dimension). In other words, if you want to build
#                      a TrellisNet with hidden size 1000 and output size 400, you should set nhid = 1000-400 = 600.
#                      (The reason we want to separate this is from Theorem 1.)
#         :param nout: The output dimension
#         :param nlevels: The number of TrellisNet layers
#         :param kernel_size: Kernel size of the TrellisNet
#         :param dilation: Dilation size of the TrellisNet
#         :param dropout: Output (variational) dropout
#         :param dropouti: Input (variational) dropout
#         :param dropouth: Hidden-to-hidden (VD-based) dropout
#         :param dropoutl: Mixture-of-Softmax dropout (only valid if MoS is used)
#         :param emb_dropout: Embedding dropout
#         :param wdrop: Weight dropout
#         :param temporalwdrop: Whether we drop only the temporal parts of the weight (only valid if wdrop > 0)
#         :param tie_weights: Whether to tie the encoder and decoder weights
#         :param repack: Whether to use history repackaging for TrellisNet
#         :param wnorm: Whether to apply weight normalization
#         :param aux: Whether to use auxiliary loss (deep supervision)
#         :param aux_frequency: The frequency of the auxiliary loss (only valid if aux == True)
#         :param n_experts: The number of softmax "experts" (i.e., whether MoS is used)
#         :param load: The path to the pickled weight file (the weights/biases should be in numpy format)
#         """
#         super(TrellisNetModel, self).__init__()
#         self.emb_dropout = emb_dropout
#         self.dropout = dropout  # Rate for dropping eventual output
#         self.dropouti = dropouti  # Rate for dropping embedding output
#         self.dropoutl = dropoutl
#         self.var_drop = VariationalDropout()
#
#         self.repack = repack
#         self.nout = nout
#         self.nhid = nhid
#         self.ninp = ninp
#         self.aux = aux
#         self.n_experts = n_experts
#         self.tie_weights = tie_weights
#         self.wnorm = wnorm
#
#         # 1) Set up encoder and decoder (embeddings)
#         self.encoder = nn.Embedding(ntoken, ninp)
#         self.decoder = nn.Linear(nhid, ntoken)
#         self.init_weights()
#         if tie_weights:
#             if nout != ninp and self.n_experts == 0:
#                 raise ValueError('When using the tied flag, nhid must be equal to emsize')
#             self.decoder.weight = self.encoder.weight
#
#         # 2) Set up TrellisNet
#         tnet = TrellisNet
#         self.tnet = tnet(ninp, nhid, nout=nout, nlevels=nlevels, kernel_size=kernel_size,
#                          dropouth=dropouth, wnorm=wnorm, aux_frequency=aux_frequency, dilation=dilation)
#
#         # 3) Set up MoS, if needed
#         if n_experts > 0:
#             print("Applied Mixture of Softmax")
#             self.mixsoft = MixSoftmax(n_experts, ntoken, nlasthid=nout, ninp=ninp, decoder=self.decoder,
#                                       dropoutl=dropoutl)
#
#         # 4) Apply weight drop connect. If weightnorm is used, we apply the dropout to its "direction", instead of "scale"
#         reg_term = '_v' if wnorm else ''
#         self.tnet = WeightDrop(self.tnet,
#                                [['full_conv', 'weight1' + reg_term],
#                                 ['full_conv', 'weight2' + reg_term]],
#                                dropout=wdrop,
#                                temporal=temporalwdrop)
#         self.network = nn.ModuleList([self.tnet])
#         if n_experts > 0: self.network.append(self.mixsoft)
#
#         # 5) Load model, if path specified
#         if len(load) > 0:
#             params_dict = torch.load(open(load, 'rb'))
#             self.load_weights(params_dict)
#             print("Model loaded successfully from {0}".format(load))
#
#     def init_weights(self):
#         initrange = 0.1
#         self.encoder.weight.data.uniform_(-initrange, initrange)
#         self.decoder.bias.data.fill_(0)
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#
#     def load_weights(self, params_dict):
#         self.load_state_dict(params_dict)
#
#     def save_weights(self, name):
#         with open(name, 'wb') as f:
#             d = self.state_dict()
#             torch.save(d, f)
#
#     def forward(self, input, hidden, decode=True):
#         """
#         Execute the forward pass of the deep network
#         :param input: The input sequence, with dimesion (N, L)
#         :param hidden: The initial hidden state (h, c)
#         :param decode: Whether to use decoder
#         :return: The predicted sequence
#         """
#         emb = embedded_dropout(self.encoder, input, self.emb_dropout if self.training else 0)
#         emb = self.var_drop(emb, self.dropouti)
#         emb = emb.transpose(1, 2)
#
#         trellisnet = self.network[0]
#         raw_output, hidden, all_raw_outputs = trellisnet(emb, hidden, aux=self.aux)
#         output = self.var_drop(raw_output, self.dropout)
#         all_outputs = self.var_drop(all_raw_outputs, self.dropout, dim=4) if self.aux else None  # N x M x L x C
#         decoded, all_decoded = None, None
#
#         if self.n_experts > 0 and not decode:
#             raise ValueError("Mixture of softmax involves decoding phase. Must set decode=True")
#
#         if self.n_experts > 0:
#             decoded = torch.log(self.mixsoft(output).add_(1e-8))
#             all_decoded = torch.log(self.mixsoft(all_outputs).add_(1e-8)) if self.aux else None
#
#         if decode:
#             decoded = decoded if self.n_experts > 0 else self.decoder(output)
#             if self.aux: all_decoded = all_decoded if self.n_experts > 0 else self.decoder(all_outputs)  # N x M x L x C
#             return (raw_output, output, decoded), hidden, all_decoded
#
#         return (raw_output, output, output), hidden, all_outputs
#
#     def init_hidden(self, bsz):
#         h_size = self.nhid + self.nout
#         weight = next(self.parameters()).data
#         return (weight.new(bsz, h_size, 1).zero_(),
#                 weight.new(bsz, h_size, 1).zero_())
