from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from keras import backend, constraints, initializers, layers, models, regularizers
from keras.backend import convert_inputs_if_ragged, maybe_convert_to_ragged
from keras.utils.control_flow_util import smart_cond
from keras.utils.generic_utils import register_keras_serializable
from keras.utils.losses_utils import compute_weighted_loss as _compute_weighted_loss, ReductionV2 as Reduction
from keras.utils.tf_utils import shape_type_conversion
from tensorflow.python.distribute import distribution_strategy_context


def compute_weighted_loss(losses, sample_weight=None, reduction=Reduction.SUM_OVER_BATCH_SIZE):
    if distribution_strategy_context.has_strategy() and \
            reduction in {Reduction.AUTO, Reduction.SUM_OVER_BATCH_SIZE}:
        raise ValueError(
            'Please use `Reduction.SUM` or  `Reduction.NONE` for loss reduction when '
            'losses are used with `tf.distribute.Strategy` outside of the built-in training loops. You can implement '
            '`Reduction.SUM_OVER_BATCH_SIZE` using global batch size like:\n'
            '```\n'
            'with strategy.scope():\n'
            '    loss_obj = losses.CategoricalCrossentropy(reduction=Reduction.NONE)\n'
            '....\n'
            '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * (1. / global_batch_size)\n'
            '```\n'
            'Please see https://www.tensorflow.org/tutorials/distribute/custom_training for more details.')

    return _compute_weighted_loss(losses, sample_weight=sample_weight, reduction=reduction)


@register_keras_serializable(package='Miss')
class AdaptiveSoftmax(layers.Layer):
    """Adaptive softmax layer.
    Reference https://arxiv.org/pdf/1609.04309.pdf
    Efficient softmax approximation for GPUs
    Edouard Grave, Armand Joulin, Moustapha Cisse, David Grangier, Herve Jegou (2017)

    Args:
        units: Positive integer, dimensionality of the output space (number of classes).
        cutoff: Ordered list of positive integers, numbers for next class-cluster start id's.
        factor: Reduction factor for second level projection matrices.
        dropout: Dropout for second level projections.
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
    Returns:
        N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D input logits and 1D input targets
        with shapes `(batch_size, input_dim)` and `(batch_size,)`, the output would have shape `(batch_size, units)`.
    """

    def __init__(
            self, units, cutoff, factor=4, dropout=0., use_bias=True, kernel_initializer='glorot_uniform',
            bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None,
            bias_constraint=None, loss_reduction=Reduction.AUTO, **kwargs):
        kwargs['autocast'] = False
        super(AdaptiveSoftmax, self).__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(min_ndim=2),  # predictions
            layers.InputSpec(min_ndim=1, dtype='int32'),  # targets
        ]
        self.supports_masking = True
        self._supports_ragged_inputs = True

        if cutoff[-1] > units - 1:
            raise ValueError('Can\'t specify `cutoff` larger than `units` size')
        units = int(units)
        Reduction.validate(loss_reduction)

        self.cutoff = cutoff
        self._cutoff = cutoff + [units] if units > cutoff[-1] else cutoff
        self.units = units
        self.factor = factor
        self.dropout = dropout
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.loss_reduction = loss_reduction

    @shape_type_conversion
    def build(self, input_shape):
        dtype = tf.dtypes.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `AdaptiveSoftmax` layer with non-floating point dtype {}'.format(dtype))

        predictions_shape, targets_shape = input_shape
        predictions_rank = len(predictions_shape)
        if len(targets_shape) + 1 != predictions_rank:
            raise ValueError('Targets shape {} rank must be one less than predictions '
                             'shape rank {}'.format(targets_shape, predictions_shape))

        self.input_channels = predictions_shape[-1]
        if self.input_channels is None:
            raise ValueError('Channel dimension of predictions should be defined. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=predictions_rank, axes={-1: self.input_channels}),
            layers.InputSpec(ndim=predictions_rank - 1, dtype=tf.int32)
        ]

        self.head = layers.Dense(
            units=self._cutoff[0] + len(self._cutoff) - 1,
            activation=None,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            name='head'
        )

        self.tails = []
        self.tail_channels = []
        prev_dim = None
        for i in range(len(self._cutoff) - 1):
            dim = self.input_channels / (self.factor ** (i + 1))
            dim = max(1, round(dim / 8)) * 8

            if dim == prev_dim:
                raise ValueError('Some cutoffs have same internal size. '
                                 'Try to shorten `cutoffs` or decrease `factor`')
            prev_dim = dim

            tail = models.Sequential([
                layers.Dense(
                    units=dim,
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    name='tail_proj_{}'.format(i),
                    input_shape=(self.input_channels,)
                ),
                layers.Dropout(self.dropout, name='tail_drop_{}'.format(i)),
                layers.Dense(
                    units=self._cutoff[i + 1] - self._cutoff[i],
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_regularizer=self.kernel_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    name='tail_scale_{}'.format(i)
                ),
            ])
            self.tails.append(tail)
            self.tail_channels.append(self._cutoff[i + 1] - self._cutoff[i])

        super(AdaptiveSoftmax, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = backend.learning_phase()

        input_logits, input_targets = inputs
        input_logits = tf.cast(input_logits, self.compute_dtype)

        input_logits, row_lengths = convert_inputs_if_ragged(input_logits)
        input_targets, _ = convert_inputs_if_ragged(input_targets)
        is_ragged_input = (row_lengths is not None)

        loss_weights = tf.ones_like(input_targets, dtype=tf.bool)
        loss_weights = maybe_convert_to_ragged(is_ragged_input, loss_weights, row_lengths)
        if is_ragged_input:
            loss_weights = loss_weights.to_tensor(False)
        if mask is not None:
            loss_weights = tf.logical_and(loss_weights, mask)
        loss_weights = tf.cast(loss_weights, self.compute_dtype)

        probs, loss = smart_cond(
            training,
            lambda: self._train_probs_loss(input_logits, input_targets, loss_weights),
            lambda: self._eval_probs_loss(input_logits, input_targets, loss_weights)
        )
        self.add_loss(loss, inputs=True)

        probs = maybe_convert_to_ragged(is_ragged_input, probs, row_lengths)

        return probs

    def _train_probs_loss(self, inputs, targets, weights):
        root_logits = self.head(inputs)
        root_logits = tf.cast(root_logits, 'float32')
        root_logprobs = tf.nn.log_softmax(root_logits)
        head_logprobs = root_logprobs[..., :self._cutoff[0]]

        tail_masks = []
        root_targets = targets
        for i in range(len(self._cutoff) - 1):
            tail_masks.append(tf.logical_and(
                tf.greater_equal(targets, self._cutoff[i]),
                tf.less(targets, self._cutoff[i + 1])
            ))
            clust_targets = tf.fill(tf.shape(root_targets), tf.cast(self._cutoff[0] + i, root_targets.dtype))
            root_targets = tf.where(tail_masks[i], clust_targets, root_targets)
        root_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=root_logits, labels=root_targets)
        root_loss = compute_weighted_loss(root_loss, sample_weight=weights, reduction=self.loss_reduction)

        full_loss = [root_loss]
        full_logprobs = [head_logprobs]
        targets_shape = tf.shape(targets)
        for i in range(len(self._cutoff) - 1):
            clust_start = self._cutoff[0] + i
            clust_logprob = root_logprobs[..., clust_start:clust_start + 1]
            tail_targets = targets - self._cutoff[i]

            true_mask = tail_masks[i]
            true_inputs = tf.boolean_mask(inputs, true_mask)
            true_logits = self.tails[i](true_inputs, training=True)
            true_logits = tf.cast(true_logits, 'float32')
            true_clust_logprob = tf.boolean_mask(clust_logprob, true_mask)
            true_logprobs = tf.nn.log_softmax(true_logits)
            true_logprobs = tf.math.add(true_logprobs, true_clust_logprob)

            false_mask = tf.logical_not(true_mask)
            false_clust_logprob = tf.boolean_mask(clust_logprob, false_mask)
            clust_size = tf.cast(self._cutoff[i + 1] - self._cutoff[i], false_clust_logprob.dtype)
            false_logprobs = false_clust_logprob - tf.math.log(clust_size)
            false_logprobs = tf.tile(false_logprobs, [1, clust_size])

            target_indices = tf.range(tf.size(targets))
            target_indices = tf.reshape(target_indices, targets_shape)
            true_indices = tf.boolean_mask(target_indices, true_mask)
            false_indices = tf.boolean_mask(target_indices, false_mask)
            target_logprobs = tf.dynamic_stitch(  # TODO: data_flow_ops.parallel_dynamic_stitch ?
                [true_indices, false_indices],
                [true_logprobs, false_logprobs]
            )

            probs_shape = tf.concat([targets_shape, tf.shape(target_logprobs)[-1:]], axis=-1)
            tail_probs = tf.reshape(target_logprobs, probs_shape)
            full_logprobs.append(tail_probs)

            true_targets = tf.boolean_mask(tail_targets, tail_masks[i])
            true_weights = tf.boolean_mask(weights, tail_masks[i])
            true_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=true_logits, labels=true_targets)
            true_loss = compute_weighted_loss(true_loss, sample_weight=true_weights, reduction=self.loss_reduction)
            full_loss.append(true_loss)

        loss = tf.reduce_mean(full_loss)

        full_logprobs = tf.concat(full_logprobs, axis=-1)
        probs = tf.math.exp(full_logprobs)

        return probs, loss

    def _eval_probs_loss(self, inputs, targets, weights):
        root_logits = self.head(inputs)
        root_logits = tf.cast(root_logits, 'float32')
        root_logprobs = tf.nn.log_softmax(root_logits)
        head_logprobs = root_logprobs[..., :self._cutoff[0]]

        # required to match tails input shape in train branch
        flat_inputs = tf.reshape(inputs, [-1, self.input_channels])
        full_logprobs = [head_logprobs]
        targets_shape = tf.shape(targets)
        for i in range(len(self._cutoff) - 1):
            flat_logits = self.tails[i](flat_inputs, training=False)
            tail_shape = tf.concat([targets_shape, [self.tail_channels[i]]], axis=-1)
            tail_logits = tf.reshape(flat_logits, tail_shape)
            tail_logits = tf.cast(tail_logits, 'float32')
            tail_logprobs = tf.nn.log_softmax(tail_logits)

            clust_start = self._cutoff[0] + i
            clust_logprob = root_logprobs[..., clust_start:clust_start + 1]
            tail_logprobs = tf.math.add(tail_logprobs, clust_logprob)
            full_logprobs.append(tail_logprobs)
        full_logprobs = tf.concat(full_logprobs, axis=-1)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=full_logprobs, labels=targets)
        loss = compute_weighted_loss(loss, sample_weight=weights, reduction=self.loss_reduction)

        probs = tf.math.exp(full_logprobs)

        return probs, loss

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        predictions_shape, _ = input_shape

        return predictions_shape[:-1] + (self.units,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)

        return tf.TensorSpec(dtype='float32', shape=outptut_signature.shape)

    def get_config(self):
        config = super(AdaptiveSoftmax, self).get_config()
        config.update({
            'cutoff': self.cutoff,
            'units': self.units,
            'factor': self.factor,
            'dropout': self.dropout,
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'loss_reduction': self.loss_reduction,
        })

        return config


@register_keras_serializable(package='Miss')
class SampledSofmax(layers.Layer):
    """Sampled softmax layer.
    Reference http://arxiv.org/abs/1412.2007.pdf
    On Using Very Large Target Vocabulary for Neural Machine Translation
    Jean et al. (2014)

    Note: The full softmax cross entropy loss calculated for evaluation.
    Note: By default this uses a log-uniform (Zipfian) distribution for sampling, so your labels must be sorted in
    order of decreasing frequency to achieve good results.  For more details, see
    `tf.random.log_uniform_candidate_sampler`.

    Args:
        units: An `int`. The number of possible classes.
        negatives: An `int`.  The number of negative classes to randomly sample per batch. This single sample of
            negative classes is evaluated for each element in the batch.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
    Returns:
        N-D tensor with shape: `(batch_size, ..., units)`. For instance, for a 2D input with shape
        `(batch_size, input_dim)`, the output would have shape `(batch_size, units)`.
    """

    def __init__(
            self, units, negatives, kernel_initializer='zeros', bias_initializer='zeros', kernel_regularizer=None,
            bias_regularizer=None, kernel_constraint=None, bias_constraint=None,
            loss_reduction=Reduction.AUTO, **kwargs):
        kwargs['dtype'] = 'float32'
        kwargs['autocast'] = False
        super(SampledSofmax, self).__init__(**kwargs)
        self.input_spec = [
            layers.InputSpec(min_ndim=2),  # predictions
            layers.InputSpec(min_ndim=1, dtype=tf.int32),  # targets
        ]
        self.supports_masking = True
        self._supports_ragged_inputs = True

        Reduction.validate(loss_reduction)

        self.units = units
        self.negatives = negatives
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.loss_reduction = loss_reduction

    @shape_type_conversion
    def build(self, input_shape):
        dtype = tf.dtypes.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                'Unable to build `{}` layer with non-floating point dtype {}'.format(self.__class__.__name__, dtype))

        predictions_shape, targets_shape = input_shape
        predictions_rank = len(predictions_shape)
        if len(targets_shape) + 1 != predictions_rank:
            raise ValueError('Targets shape {} rank must be one less than predictions '
                             'shape rank {}'.format(targets_shape, predictions_shape))

        self.num_channels = predictions_shape[-1]
        if self.num_channels is None:
            raise ValueError('Channel dimension of predictions should be defined. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=predictions_rank, axes={-1: self.num_channels}),
            layers.InputSpec(ndim=predictions_rank - 1, dtype=tf.int32)
        ]

        with tf.device('cpu:0'):
            self.kernel = self.add_weight(
                shape=(self.units, self.num_channels),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='kernel',
                dtype=self.dtype,
                trainable=True
            )
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias',
                dtype=self.dtype,
                trainable=True
            )

        super(SampledSofmax, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        with tf.device('cpu:0'):
            if training is None:
                training = backend.learning_phase()

            input_logits, input_targets = inputs
            input_logits = tf.cast(input_logits, self.compute_dtype)
            input_logits, row_lengths = convert_inputs_if_ragged(input_logits)
            input_targets, _ = convert_inputs_if_ragged(input_targets)
            is_ragged_input = (row_lengths is not None)

            loss_weights = tf.ones_like(input_targets, dtype=tf.bool)
            loss_weights = maybe_convert_to_ragged(is_ragged_input, loss_weights, row_lengths)
            if is_ragged_input:
                loss_weights = loss_weights.to_tensor(False)
            if mask is not None:
                loss_weights = tf.logical_and(loss_weights, mask)
            loss_weights = tf.cast(loss_weights, self.compute_dtype)

            input_shape = tf.shape(input_logits)
            output_shape = tf.stack(tf.unstack(input_shape)[:-1] + [self.units])
            input_logits = tf.reshape(input_logits, [-1, self.num_channels])
            input_targets = tf.reshape(input_targets, [-1])
            loss_weights = tf.reshape(loss_weights, [-1])

            output_logits = tf.matmul(input_logits, self.kernel, transpose_b=True)
            output_logits = tf.nn.bias_add(output_logits, self.bias)

            loss = smart_cond(
                training,
                lambda: self._train_loss(input_logits, input_targets),
                lambda: self._eval_loss(output_logits, input_targets)
            )
            loss = compute_weighted_loss(loss, sample_weight=loss_weights, reduction=self.loss_reduction)
            self.add_loss(loss, inputs=True)

            output_probs = tf.nn.softmax(output_logits)
            output_probs = tf.reshape(output_probs, output_shape)
            output_probs = maybe_convert_to_ragged(is_ragged_input, output_probs, row_lengths)

            return output_probs

    def _train_loss(self, logits, targets):
        labels_exp_dim = tf.expand_dims(targets, axis=-1)

        return tf.nn.sampled_softmax_loss(
            weights=self.kernel,
            biases=self.bias,
            labels=labels_exp_dim,
            inputs=logits,
            num_sampled=self.negatives,
            num_classes=self.units,
        )

    def _eval_loss(self, logits, targets):
        labels_one_hot = tf.one_hot(targets, self.units)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels_one_hot
        )

        return loss

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        predictions_shape, _ = input_shape

        return predictions_shape[:-1] + (self.units,)

    def get_config(self):
        config = super(SampledSofmax, self).get_config()
        config.update({
            'units': self.units,
            'negatives': self.negatives,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'loss_reduction': self.loss_reduction,
        })

        return config


@register_keras_serializable(package='Miss')
class NoiseContrastiveEstimation(SampledSofmax):
    """Noise-contrastive estimation layer.
    Reference: http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf
    Noise-contrastive estimation: A new estimation principle for unnormalized statistical models
    Gutmann, Hyvarinen (2010)

    Note: The full sigmoid loss calculated for evaluation.
    Note: By default this uses a log-uniform (Zipfian) distribution for sampling, so your labels must be sorted in
    order of decreasing frequency to achieve good results.  For more details, see
    `tf.random.log_uniform_candidate_sampler`.
    """

    def _train_loss(self, logits, targets):
        labels_exp_dim = tf.expand_dims(targets, axis=-1)
        loss = tf.nn.nce_loss(
            weights=self.kernel,
            biases=self.bias,
            labels=labels_exp_dim,
            inputs=logits,
            num_sampled=self.negatives,
            num_classes=self.units,
        )

        return loss / tf.cast(1 + self.negatives, 'float32')

    def _eval_loss(self, logits, targets):
        labels_one_hot = tf.one_hot(targets, self.units)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=logits,
            labels=labels_one_hot
        )

        return tf.reduce_sum(loss, axis=-1)
