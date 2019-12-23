from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import losses_utils


def compute_weighted_loss(losses, sample_weight=None, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
    if distribution_strategy_context.has_strategy() and \
            reduction in {tf.keras.losses.Reduction.AUTO, tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE}:
        raise ValueError(
            'Please use `tf.keras.losses.Reduction.SUM` or `tf.keras.losses.Reduction.NONE` for loss reduction '
            'when losses are used with `tf.distribute.Strategy` outside of the built-in training loops. You can '
            'implement `tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch size like:\n'
            '```\n'
            'with strategy.scope():\n'
            '    loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.reduction.NONE)\n'
            '....\n'
            '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * (1. / global_batch_size)\n'
            '```\n'
            'Please see https://www.tensorflow.org/alpha/tutorials/distribute/training_loops for more details.')

    return losses_utils.compute_weighted_loss(losses, sample_weight=sample_weight, reduction=reduction)


@tf.keras.utils.register_keras_serializable(package='Miss')
class AdaptiveSoftmax(tf.keras.layers.Layer):
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

    def __init__(self,
                 units, cutoff,
                 factor=4,
                 dropout=0.,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 loss_reduction=tf.keras.losses.Reduction.AUTO,
                 **kwargs):
        super(AdaptiveSoftmax, self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)

        if cutoff[-1] > units - 1:
            raise ValueError('Can\'t specify `cutoff` larger than `units` size')
        units = int(units)
        tf.keras.losses.Reduction.validate(loss_reduction)

        self.cutoff = cutoff + [units] if units > cutoff[-1] else cutoff
        self._cutoff = cutoff
        self.units = units
        self.factor = factor
        self.dropout = dropout
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.loss_reduction = loss_reduction

        self.supports_masking = True
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2),  # predictions
            tf.keras.layers.InputSpec(min_ndim=1),  # targets
        ]

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `AdaptiveSoftmax` layer with non-floating point dtype {}'.format(dtype))

        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError('An `AdaptiveSoftmax` layer should be called on exactly 2 inputs: '
                             '`predictions` and `targets`'.format(self.__class__.__name__))

        predictions_shape, targets_shape = input_shape
        predictions_rank = len(predictions_shape)
        if predictions_rank < 2:
            raise ValueError('Predictions shape {} must have rank >= 2'.format(predictions_shape))
        if len(targets_shape) + 1 != predictions_rank:
            raise ValueError('Targets shape {} rank must be one less than predictions '
                             'shape rank {}'.format(targets_shape, predictions_shape))

        num_channels = predictions_shape[-1]
        if num_channels is None:
            raise ValueError('Channel dimension of predictions should be defined. Found `None`.')
        self.input_spec = [
            tf.keras.layers.InputSpec(ndim=predictions_rank, axes={-1: num_channels}),
            tf.keras.layers.InputSpec(ndim=predictions_rank - 1)
        ]

        self.head = tf.keras.layers.Dense(
            units=self.cutoff[0] + len(self.cutoff) - 1,
            activation=None,
            use_bias=False,
            kernel_initializer=self.kernel_initializer,
            kernel_regularizer=self.kernel_regularizer,
            activity_regularizer=self.activity_regularizer,
            kernel_constraint=self.kernel_constraint,
            dtype=self.dtype,
            name='head'
        )

        self.tails = []
        prev_dim = None
        for i in range(len(self.cutoff) - 1):
            dim = num_channels / (self.factor ** (i + 1))
            dim = max(1, round(dim / 8)) * 8

            if dim == prev_dim:
                raise ValueError('Some cutoffs have same internal size. '
                                 'Try to shorten `cutoffs` or decrease `factor`')
            prev_dim = dim

            tail = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    units=dim,
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    bias_regularizer=self.bias_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    dtype=self.dtype,
                    name='tail_proj_{}'.format(i)
                ),
                tf.keras.layers.Dropout(self.dropout, 'tail_drop_{}'.format(i)),
                tf.keras.layers.Dense(
                    units=self.cutoff[i + 1] - self.cutoff[i],
                    activation=None,
                    use_bias=self.use_bias,
                    kernel_initializer=self.kernel_initializer,
                    bias_initializer=self.bias_initializer,
                    bias_regularizer=self.bias_regularizer,
                    kernel_regularizer=self.kernel_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.kernel_constraint,
                    bias_constraint=self.bias_constraint,
                    dtype=self.dtype,
                    name='tail_scale_{}'.format(i)
                ),
            ])
            self.tails.append(tail)
            setattr(self, 'tail_{}'.format(i), tail)

        super(AdaptiveSoftmax, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        input_logits, input_targets = inputs
        input_logits = tf.cast(input_logits, self.dtype)
        if tf.executing_eagerly() and len(input_logits.shape) == len(input_targets.shape):
            # https://github.com/tensorflow/tensorflow/issues/34687
            input_targets = tf.squeeze(input_targets, axis=-1)

        root_logits = self.head(input_logits)
        root_logprobs = tf.nn.log_softmax(root_logits)
        head_probs = tf.math.exp(root_logprobs[..., :self.cutoff[0]])

        tail_masks = []
        root_targets = input_targets
        for t in range(len(self.cutoff) - 1):
            tail_masks.append(tf.logical_and(
                tf.greater_equal(input_targets, self.cutoff[t]),
                tf.less(input_targets, self.cutoff[t + 1])
            ))
            clust_targets = tf.fill(tf.shape(root_targets), tf.cast(self.cutoff[0] + t, root_targets.dtype))
            root_targets = tf.where(tail_masks[t], clust_targets, root_targets)
        root_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=root_logits, labels=root_targets)
        root_loss = compute_weighted_loss(root_loss, sample_weight=None, reduction=self.loss_reduction)

        full_loss = [root_loss]
        full_probs = [head_probs]

        def _train_tail_loss_probs():
            train_losses, train_probs = [], []

            targets_shape = tf.shape(input_targets)
            for i in range(len(self.cutoff) - 1):
                tail_clust_start = self.cutoff[0] + i
                tail_clust_logprobs = root_logprobs[..., tail_clust_start:tail_clust_start + 1]
                tail_input_targets = input_targets - self.cutoff[i]

                true_tail_inputs = tf.boolean_mask(input_logits, tail_masks[i])
                true_tail_logits = self.tails[i](true_tail_inputs)
                true_clust_logprobs = tf.boolean_mask(tail_clust_logprobs, tail_masks[i])
                true_tail_logprobs = tf.nn.log_softmax(true_tail_logits)
                true_tail_logprobs = tf.math.add(true_tail_logprobs, true_clust_logprobs)
                true_tail_probs = tf.math.exp(true_tail_logprobs)

                false_tail_mask = tf.logical_not(tail_masks[i])
                false_clust_logprobs = tf.boolean_mask(tail_clust_logprobs, false_tail_mask)
                tail_clust_size = self.cutoff[i + 1] - self.cutoff[i]
                false_tail_probs = tf.math.exp(false_clust_logprobs) / tail_clust_size
                false_tail_probs = tf.tile(false_tail_probs, [1, tail_clust_size])

                target_tail_indices = tf.range(tf.size(input_targets))
                target_tail_indices = tf.reshape(target_tail_indices, targets_shape)
                true_tail_indices = tf.boolean_mask(target_tail_indices, tail_masks[i])
                false_tail_indices = tf.boolean_mask(target_tail_indices, false_tail_mask)

                target_tail_probs = tf.dynamic_stitch(
                    [true_tail_indices, false_tail_indices],
                    [true_tail_probs, false_tail_probs]
                )
                target_probs_shape = tf.concat([targets_shape, tf.shape(target_tail_probs)[-1:]], axis=-1)
                target_tail_probs = tf.reshape(target_tail_probs, target_probs_shape)
                train_probs.append(target_tail_probs)

                true_tail_targets = tf.boolean_mask(tail_input_targets, tail_masks[i])
                true_tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=true_tail_logits, labels=true_tail_targets)
                false_tail_targets = tf.boolean_mask(tail_input_targets, false_tail_mask)
                false_tail_loss = tf.fill(tf.shape(false_tail_targets), tf.cast(0., true_tail_loss.dtype))
                full_tail_loss = tf.dynamic_stitch(
                    [true_tail_indices, false_tail_indices],
                    [true_tail_loss, false_tail_loss]
                )
                full_tail_loss = tf.reshape(full_tail_loss, targets_shape)
                full_tail_loss = compute_weighted_loss(full_tail_loss, sample_weight=None,
                                                       reduction=self.loss_reduction)
                train_losses.append(full_tail_loss)

            return train_losses, train_probs

        def _eval_tail_loss_probs():
            eval_losses, eval_probs = [], []
            for i in range(len(self.cutoff) - 1):
                tail_logits = self.tails[i](input_logits)

                tail_targets = tf.where(
                    tail_masks[i],
                    input_targets - self.cutoff[i],
                    tf.zeros_like(input_targets, dtype=input_targets.dtype)
                )
                tail_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=tail_logits, labels=tail_targets)
                tail_loss = compute_weighted_loss(tail_loss, sample_weight=None, reduction=self.loss_reduction)
                eval_losses.append(tail_loss)

                tail_logprobs = tf.nn.log_softmax(tail_logits)
                clust_start = self.cutoff[0] + i
                tail_logprobs = tf.math.add(tail_logprobs, root_logprobs[..., clust_start:clust_start + 1])
                eval_probs.append(tf.math.exp(tail_logprobs))

            return eval_losses, eval_probs

        tail_losses, tail_probs = tf_utils.smart_cond(training, _train_tail_loss_probs, _eval_tail_loss_probs)
        full_loss.extend(tail_losses)
        full_probs.extend(tail_probs)

        self.add_loss(tf.add_n(full_loss), inputs=True)

        return tf.concat(full_probs, axis=-1)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError('An `AdaptiveSoftmax` layer should be called on exactly 2 inputs: '
                             '`predictions` and `targets`'.format(self.__class__.__name__))
        predictions_shape, _ = input_shape

        if len(predictions_shape) < 2:
            raise ValueError('Predictions shape {} must have rank => 2'.format(predictions_shape))

        return predictions_shape[:-1] + (self.units,)

    def get_config(self):
        config = {
            'cutoff': self._cutoff,
            'units': self.units,
            'factor': self.factor,
            'dropout': self.dropout,
            'use_bias': self.use_bias,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'loss_reduction': self.loss_reduction,
        }
        base_config = super(AdaptiveSoftmax, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


class _SoftmaxSamplingWrapper(tf.keras.layers.Layer):
    """Wrapper for softmax sampling layers.

    Note: The full sigmoid loss calculated for evaluation.

    Args:
        sample_loss: Softmax sampling loss function.
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

    def __init__(self,
                 sample_loss,
                 units,
                 negatives,
                 kernel_initializer='zeros',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 loss_reduction=tf.keras.losses.Reduction.AUTO,
                 **kwargs):
        super(_SoftmaxSamplingWrapper, self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)

        tf.keras.losses.Reduction.validate(loss_reduction)

        self.sample_loss = sample_loss
        self.units = units
        self.negatives = negatives
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.loss_reduction = loss_reduction

        self.supports_masking = True
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2),  # predictions
            tf.keras.layers.InputSpec(min_ndim=1),  # targets
        ]

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `{}` layer with non-floating '
                            'point dtype {}'.format(self.__class__.__name__, dtype))

        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError('A `{}` layer should be called on exactly 2 inputs: '
                             '`predictions` and `targets`'.format(self.__class__.__name__))

        predictions_shape, targets_shape = input_shape
        predictions_rank = len(predictions_shape)
        if predictions_rank < 2:
            raise ValueError('Predictions shape {} must have rank >= 2'.format(predictions_shape))
        if len(targets_shape) + 1 != predictions_rank:
            raise ValueError('Targets shape {} rank must be one less than predictions '
                             'shape rank {}'.format(targets_shape, predictions_shape))

        self.num_channels = predictions_shape[-1]
        if self.num_channels is None:
            raise ValueError('Channel dimension of predictions should be defined. Found `None`.')
        self.input_spec = [
            tf.keras.layers.InputSpec(ndim=predictions_rank, axes={-1: self.num_channels}),
            tf.keras.layers.InputSpec(ndim=predictions_rank - 1)
        ]

        self.kernel = self.add_weight(
            shape=(self.num_channels, self.units),
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

        super(_SoftmaxSamplingWrapper, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        input_logits, input_targets = inputs
        input_logits = tf.cast(input_logits, self._compute_dtype)

        input_shape = tf.shape(input_logits)
        input_logits = tf.reshape(input_logits, [-1, self.num_channels])
        input_targets = tf.reshape(input_targets, [-1])

        output_logits = tf.matmul(input_logits, self.kernel)
        output_logits = tf.nn.bias_add(output_logits, self.bias)

        def _train_loss():
            labels_exp_dim = tf.expand_dims(input_targets, axis=-1)
            return self.sample_loss(
                weights=tf.transpose(self.kernel),
                biases=self.bias,
                labels=labels_exp_dim,
                inputs=input_logits,
                num_sampled=self.negatives,
                num_classes=self.units,
            )

        def _eval_loss():
            labels_one_hot = tf.one_hot(input_targets, self.units)
            per_logit_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_one_hot,
                logits=output_logits
            )
            return tf.reduce_sum(per_logit_loss, axis=-1)

        loss = tf_utils.smart_cond(training, _train_loss, _eval_loss)
        loss = compute_weighted_loss(loss, sample_weight=None, reduction=self.loss_reduction)
        self.add_loss(loss, inputs=True)

        output_probs = tf.nn.softmax(output_logits)

        output_shape = tf.stack(tf.unstack(input_shape)[:-1] + [self.units])
        output_probs = tf.reshape(output_probs, output_shape)

        return output_probs

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError('A `{}` layer should be called on exactly 2 inputs: '
                             '`predictions` and `targets`'.format(self.__class__.__name__))
        predictions_shape, _ = input_shape

        if len(predictions_shape) < 2:
            raise ValueError('Predictions shape {} must have rank => 2'.format(predictions_shape))

        return predictions_shape[:-1] + (self.units,)

    def get_config(self):
        config = {
            'units': self.units,
            'negatives': self.negatives,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': tf.keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': tf.keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': tf.keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': tf.keras.constraints.serialize(self.bias_constraint),
            'loss_reduction': self.loss_reduction,
        }
        base_config = super(_SoftmaxSamplingWrapper, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='Miss')
class NoiseContrastiveEstimation(_SoftmaxSamplingWrapper):
    """Noise-contrastive estimation layer.
    Reference: http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf
    Noise-contrastive estimation: A new estimation principle for unnormalized statistical models
    Gutmann, Hyvarinen (2010)

    Note: The full sigmoid loss calculated for evaluation.
    Note: By default this uses a log-uniform (Zipfian) distribution for sampling, so your labels must be sorted in
    order of decreasing frequency to achieve good results.  For more details, see
    `tf.random.log_uniform_candidate_sampler`.

    Args:
        units: An `int`. The number of possible classes.
        negatives: An `int`.  The number of negative classes to randomly sample per batch. This single sample of
            negative classes is evaluated for each element in the batch.
    Returns:
        A `batch_size` 1-D tensor of per-example NCE losses.
    """

    def __init__(self, units, negatives, **kwargs):
        super(NoiseContrastiveEstimation, self).__init__(
            sample_loss=tf.nn.nce_loss, units=units, negatives=negatives, **kwargs)


@tf.keras.utils.register_keras_serializable(package='Miss')
class SampledSofmax(_SoftmaxSamplingWrapper):
    """Sampled softmax layer.
    Reference http://arxiv.org/abs/1412.2007.pdf
    On Using Very Large Target Vocabulary for Neural Machine Translation
    Jean et al. (2014)

    Note: The full sigmoid loss calculated for evaluation.
    Note: By default this uses a log-uniform (Zipfian) distribution for sampling, so your labels must be sorted in
    order of decreasing frequency to achieve good results.  For more details, see
    `tf.random.log_uniform_candidate_sampler`.

    Args:
        units: An `int`. The number of possible classes.
        negatives: An `int`.  The number of negative classes to randomly sample per batch. This single sample of
            negative classes is evaluated for each element in the batch.
    Returns:
        A `batch_size` 1-D tensor of per-example NCE losses.
    """

    def __init__(self, units, negatives, **kwargs):
        super(SampledSofmax, self).__init__(
            sample_loss=tf.nn.sampled_softmax_loss, units=units, negatives=negatives, **kwargs)
