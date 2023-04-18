from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend, constraints, initializers, layers, models, regularizers
from keras.saving import register_keras_serializable
from keras.src.utils.control_flow_util import smart_cond
from keras.src.utils.losses_utils import compute_weighted_loss as _compute_weighted_loss, ReductionV2 as Reduction
from keras.src.utils.tf_utils import shape_type_conversion
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.ops import data_flow_ops


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
        return_probs: Boolean, whether to estimate and return full probability matrix. Disabled by default
            to reduce computations.
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
            self, units, cutoff, return_probs=False, factor=4, dropout=0., kernel_initializer='glorot_uniform',
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

        self.units = units
        self.cutoff = cutoff
        self.return_probs = return_probs
        self.factor = factor
        self.dropout = dropout
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.loss_reduction = loss_reduction

        self._cutoff = cutoff + [units] if units > cutoff[-1] else cutoff
        self._splits = self._cutoff[:1] + [1] * (len(self._cutoff) - 1)

    @shape_type_conversion
    def build(self, input_shape):
        predictions_shape, targets_shape = input_shape
        predictions_rank = len(predictions_shape)
        if len(targets_shape) + 1 != predictions_rank:
            raise ValueError('Targets shape {} rank must be one less than predictions '
                             'shape rank {}'.format(targets_shape, predictions_shape))

        self.channels = predictions_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of predictions should be defined. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=predictions_rank, axes={-1: self.channels}),
            layers.InputSpec(ndim=predictions_rank - 1, dtype=tf.int32)
        ]

        self.root = layers.Dense(
            units=self._cutoff[0] + len(self._cutoff) - 1,
            activation=None,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            kernel_constraint=self.kernel_constraint,
            name='head')

        self.tails = []
        prev_dim = None
        for i in range(len(self._cutoff) - 1):
            dim = self.channels / (self.factor ** (i + 1))
            dim = max(1, round(dim / 8)) * 8

            if dim == prev_dim:
                raise ValueError('Some cutoffs have same internal size. '
                                 'Try to shorten `cutoffs` or decrease `factor`')
            prev_dim = dim

            tail = models.Sequential([
                layers.Dense(
                    units=dim,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    name='tail_proj_{}'.format(i),
                    input_shape=(self.channels,)
                ),
                layers.Dropout(self.dropout, name='tail_drop_{}'.format(i)),
                layers.Dense(
                    units=self._cutoff[i + 1] - self._cutoff[i],
                    activation=None,
                    use_bias=False,
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

        super(AdaptiveSoftmax, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        if training is None:
            training = backend.learning_phase()

        input_logits, input_targets = inputs
        input_logits = tf.cast(input_logits, self.compute_dtype)

        inputs_flat = tf.reshape(input_logits, [-1, self.channels])
        targets_flat = tf.reshape(input_targets, [-1])
        if mask is not None:
            mask = tf.reshape(mask, [-1])

        if not self.return_probs:
            loss = self._train_loss(inputs_flat, targets_flat, mask)
            loss = compute_weighted_loss(loss, reduction=self.loss_reduction)
            self.add_loss(loss)

            return input_logits

        logprobs_flat, loss = smart_cond(
            training,
            lambda: self._train_probs_loss(inputs_flat, targets_flat, mask),
            lambda: self._eval_probs_loss(inputs_flat, targets_flat, mask))
        self.add_loss(loss)

        probs_flat = tf.math.exp(logprobs_flat)

        if isinstance(input_logits, tf.RaggedTensor):
            output_probs = input_targets.with_flat_values(probs_flat)
        else:
            probs_shape = tf.concat([tf.shape(input_logits)[:-1], [self.units]], axis=-1)
            output_probs = tf.reshape(probs_flat, probs_shape)

        return output_probs

    def _train_loss(self, inputs, targets, mask):
        inputs = inputs if mask is None else inputs[mask]
        targets = targets if mask is None else targets[mask]

        root_logits = self.root(inputs)
        root_logits = tf.cast(root_logits, 'float32')

        full_loss = []
        root_targets = targets
        for i in range(len(self._cutoff) - 1):
            tail_mask = (targets >= self._cutoff[i]) & (targets < self._cutoff[i + 1])
            root_targets = tf.where(tail_mask, self._cutoff[0] + i, root_targets)

            tail_inputs = inputs[tail_mask]
            tail_logits = self.tails[i](tail_inputs)
            tail_logits = tf.cast(tail_logits, 'float32')

            tail_targets = targets[tail_mask] - self._cutoff[i]

            tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tail_logits, labels=tail_targets)
            tail_loss = compute_weighted_loss(tail_loss, reduction=self.loss_reduction)
            full_loss.append(tail_loss)

        root_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=root_logits, labels=root_targets)
        root_loss = compute_weighted_loss(root_loss, reduction=self.loss_reduction)
        full_loss.insert(0, root_loss)

        loss = tf.reduce_mean(full_loss)

        return loss

    def _train_probs_loss(self, inputs, targets, mask):
        root_logits = self.root(inputs)
        root_logits = tf.cast(root_logits, 'float32')
        root_logprobs = tf.nn.log_softmax(root_logits)
        root_logprobs = tf.split(root_logprobs, self._splits, axis=-1)

        target_indices = tf.range(tf.size(targets))

        full_loss = []
        full_logprobs = root_logprobs[:1]
        root_targets = targets
        for i in range(len(self._cutoff) - 1):
            tail_mask = (targets >= self._cutoff[i]) & (targets < self._cutoff[i + 1])
            root_targets = tf.where(tail_mask, self._cutoff[0] + i, root_targets)

            tail_inputs = inputs[tail_mask]
            tail_logits = self.tails[i](tail_inputs)
            tail_logits = tf.cast(tail_logits, 'float32')
            tail_logprobs = tf.nn.log_softmax(tail_logits)
            tail_logprobs += root_logprobs[i + 1][tail_mask]

            tail_targets = targets[tail_mask] - self._cutoff[i]

            tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tail_logits, labels=tail_targets)
            tail_loss = compute_weighted_loss(tail_loss, reduction=self.loss_reduction)
            full_loss.append(tail_loss)

            false_mask = ~tail_mask
            cutoff_size = self._cutoff[i + 1] - self._cutoff[i]
            false_logprobs = root_logprobs[i + 1][false_mask] - np.log(cutoff_size)
            false_logprobs = tf.tile(false_logprobs, [1, cutoff_size])

            full_logprobs.append(data_flow_ops.parallel_dynamic_stitch(
                [target_indices[tail_mask], target_indices[false_mask]],
                [tail_logprobs, false_logprobs]
            ))

        root_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=root_logits, labels=root_targets)
        root_loss = compute_weighted_loss(root_loss, reduction=self.loss_reduction)
        full_loss.insert(0, root_loss)

        full_loss = full_loss if mask is None else full_loss[mask]
        full_loss = tf.reduce_mean(full_loss)

        full_logprobs = tf.concat(full_logprobs, axis=-1)

        return full_logprobs, full_loss

    def _eval_probs_loss(self, inputs, targets, mask):
        root_logits = self.root(inputs)
        root_logits = tf.cast(root_logits, 'float32')
        root_logprobs = tf.nn.log_softmax(root_logits)
        root_logprobs = tf.split(root_logprobs, self._splits, axis=-1)

        full_logprobs = root_logprobs[:1]
        for i in range(len(self._cutoff) - 1):
            tail_logits = self.tails[i](inputs, training=False)
            tail_logits = tf.cast(tail_logits, 'float32')
            tail_logprobs = tf.nn.log_softmax(tail_logits)
            full_logprobs.append(tail_logprobs + root_logprobs[i + 1])
        full_logprobs = tf.concat(full_logprobs, axis=-1)

        full_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=full_logprobs, labels=targets)
        full_loss = full_loss if mask is None else full_loss[mask]
        full_loss = compute_weighted_loss(full_loss, reduction=self.loss_reduction)

        return full_logprobs, full_loss

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not self.return_probs:
            return input_shape[0]

        predictions_shape, _ = input_shape

        return predictions_shape[:-1] + (self.units,)

    def compute_output_signature(self, input_signature):
        outptut_signature = super().compute_output_signature(input_signature)
        if not self.return_probs:
            return outptut_signature

        return tf.TensorSpec(dtype='float32', shape=outptut_signature.shape)

    def get_config(self):
        config = super(AdaptiveSoftmax, self).get_config()
        config.update({
            'units': self.units,
            'cutoff': self.cutoff,
            'return_probs': self.return_probs,
            'factor': self.factor,
            'dropout': self.dropout,
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
        return_probs: Boolean, whether to estimate and return full probability matrix. Disabled by default
            to reduce computations.
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
            self, units, negatives, return_probs=False, kernel_initializer='zeros', bias_initializer='zeros',
            kernel_regularizer=None, bias_regularizer=None, kernel_constraint=None, bias_constraint=None,
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
        self.return_probs = return_probs
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.loss_reduction = loss_reduction

    @shape_type_conversion
    def build(self, input_shape):
        predictions_shape, targets_shape = input_shape
        predictions_rank = len(predictions_shape)
        if len(targets_shape) + 1 != predictions_rank:
            raise ValueError('Targets shape {} rank must be one less than predictions '
                             'shape rank {}'.format(targets_shape, predictions_shape))

        self.channels = predictions_shape[-1]
        if self.channels is None:
            raise ValueError('Channel dimension of predictions should be defined. Found `None`.')
        self.input_spec = [
            layers.InputSpec(ndim=predictions_rank, axes={-1: self.channels}),
            layers.InputSpec(ndim=predictions_rank - 1, dtype=tf.int32)
        ]

        with tf.device('cpu:0'):
            self.kernel = self.add_weight(
                shape=(self.units, self.channels),
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint,
                name='kernel',
                dtype=self.dtype,
                trainable=True)
            self.bias = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias',
                dtype=self.dtype,
                trainable=True)

        super(SampledSofmax, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        with tf.device('cpu:0'):
            if training is None:
                training = backend.learning_phase()

            input_logits, input_targets = inputs
            input_logits = tf.cast(input_logits, self.compute_dtype)

            inputs_flat = tf.reshape(input_logits, [-1, self.channels])
            targets_flat = tf.reshape(input_targets, [-1])
            if mask is not None:
                mask = tf.reshape(mask, [-1])

            if not self.return_probs:
                loss = self._train_loss(inputs_flat, targets_flat)
                loss = loss if mask is None else loss[mask]
                loss = compute_weighted_loss(loss, reduction=self.loss_reduction)
                self.add_loss(loss)

                return input_logits

            logits_flat = tf.matmul(inputs_flat, self.kernel, transpose_b=True)
            logits_flat = tf.nn.bias_add(logits_flat, self.bias)

            loss = smart_cond(
                training,
                lambda: self._train_loss(inputs_flat, targets_flat),
                lambda: self._eval_loss(logits_flat, targets_flat))
            loss = loss if mask is None else loss[mask]
            loss = compute_weighted_loss(loss, reduction=self.loss_reduction)
            self.add_loss(loss)

            probs_flat = tf.nn.softmax(logits_flat)

            if isinstance(input_logits, tf.RaggedTensor):
                output_probs = input_targets.with_flat_values(probs_flat)
            else:
                probs_shape = tf.concat([tf.shape(input_logits)[:-1], [self.units]], axis=-1)
                output_probs = tf.reshape(probs_flat, probs_shape)

            return output_probs

    def _train_loss(self, logits, targets):
        return tf.nn.sampled_softmax_loss(
            weights=self.kernel,
            biases=self.bias,
            labels=targets[..., None],
            inputs=logits,
            num_sampled=self.negatives,
            num_classes=self.units)

    def _eval_loss(self, logits, targets):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

    @shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not self.return_probs:
            return input_shape[0]

        predictions_shape, _ = input_shape

        return predictions_shape[:-1] + (self.units,)

    def get_config(self):
        config = super(SampledSofmax, self).get_config()
        config.update({
            'units': self.units,
            'negatives': self.negatives,
            'return_probs': self.return_probs,
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

    Note: This layer uses a log-uniform (Zipfian) distribution for sampling, so your labels must be sorted in order of
    decreasing frequency to achieve good results.  For more details, see `tf.random.log_uniform_candidate_sampler`.
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
