from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.keras.backend import convert_inputs_if_ragged, maybe_convert_to_ragged
from tensorflow.python.keras.utils import losses_utils, tf_utils


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
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2),  # predictions
            tf.keras.layers.InputSpec(min_ndim=1),  # targets
        ]
        self.supports_masking = True
        self._supports_ragged_inputs = True

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
        if tf.executing_eagerly() \
                and not isinstance(input_logits, tf.RaggedTensor) \
                and not isinstance(input_targets, tf.RaggedTensor) \
                and len(input_logits.shape) == len(input_targets.shape):
            # https://github.com/tensorflow/tensorflow/issues/34687
            input_targets = tf.squeeze(input_targets, axis=-1)

        input_logits, row_lengths = convert_inputs_if_ragged(input_logits)
        input_targets, _ = convert_inputs_if_ragged(input_targets)
        is_ragged_input = (row_lengths is not None)

        probs, loss = tf_utils.smart_cond(
            training,
            lambda: self._train_probs_loss(input_logits, input_targets),
            lambda: self._eval_probs_loss(input_logits, input_targets)
        )
        self.add_loss(loss, inputs=True)

        probs = maybe_convert_to_ragged(is_ragged_input, probs, row_lengths)

        return probs

    def _train_probs_loss(self, inputs, targets):
        root_logits = self.head(inputs)
        root_logits = tf.cast(root_logits, tf.float32)
        root_logprobs = tf.nn.log_softmax(root_logits)
        head_logprobs = root_logprobs[..., :self.cutoff[0]]

        tail_masks = []
        root_targets = targets
        for i in range(len(self.cutoff) - 1):
            tail_masks.append(tf.logical_and(
                tf.greater_equal(targets, self.cutoff[i]),
                tf.less(targets, self.cutoff[i + 1])
            ))
            clust_targets = tf.fill(tf.shape(root_targets), tf.cast(self.cutoff[0] + i, root_targets.dtype))
            root_targets = tf.where(tail_masks[i], clust_targets, root_targets)
        root_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=root_logits, labels=root_targets)
        root_loss = compute_weighted_loss(root_loss, sample_weight=None, reduction=self.loss_reduction)

        full_loss = [root_loss]
        full_logprobs = [head_logprobs]
        targets_shape = tf.shape(targets)
        for i in range(len(self.cutoff) - 1):
            clust_start = self.cutoff[0] + i
            clust_logprob = root_logprobs[..., clust_start:clust_start + 1]
            tail_targets = targets - self.cutoff[i]

            true_mask = tail_masks[i]
            true_inputs = tf.boolean_mask(inputs, true_mask)
            true_logits = self.tails[i](true_inputs, training=True)
            true_logits = tf.cast(true_logits, tf.float32)
            true_clust_logprob = tf.boolean_mask(clust_logprob, true_mask)
            true_logprobs = tf.nn.log_softmax(true_logits)
            true_logprobs = tf.math.add(true_logprobs, true_clust_logprob)

            false_mask = tf.logical_not(true_mask)
            false_clust_logprob = tf.boolean_mask(clust_logprob, false_mask)
            clust_size = tf.cast(self.cutoff[i + 1] - self.cutoff[i], false_clust_logprob.dtype)
            false_logprobs = false_clust_logprob - tf.math.log(clust_size)
            false_logprobs = tf.tile(false_logprobs, [1, clust_size])

            target_indices = tf.range(tf.size(targets))
            target_indices = tf.reshape(target_indices, targets_shape)
            true_indices = tf.boolean_mask(target_indices, true_mask)
            false_indices = tf.boolean_mask(target_indices, false_mask)

            target_logprobs = tf.dynamic_stitch(
                [true_indices, false_indices],
                [true_logprobs, false_logprobs]
            )
            probs_shape = tf.concat([targets_shape, tf.shape(target_logprobs)[-1:]], axis=-1)
            tail_probs = tf.reshape(target_logprobs, probs_shape)
            full_logprobs.append(tail_probs)

            true_targets = tf.boolean_mask(tail_targets, tail_masks[i])
            true_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=true_logits, labels=true_targets)
            true_loss = compute_weighted_loss(true_loss, sample_weight=None, reduction=self.loss_reduction)
            full_loss.append(true_loss)

        loss = tf.reduce_mean(full_loss)

        full_logprobs = tf.concat(full_logprobs, axis=-1)
        probs = tf.math.exp(full_logprobs)

        return probs, loss

    def _eval_probs_loss(self, inputs, targets):
        root_logits = self.head(inputs)
        root_logits = tf.cast(root_logits, tf.float32)
        root_logprobs = tf.nn.log_softmax(root_logits)
        head_logprobs = root_logprobs[..., :self.cutoff[0]]

        full_logprobs = [head_logprobs]
        for i in range(len(self.cutoff) - 1):
            tail_logits = self.tails[i](inputs, training=False)
            tail_logits = tf.cast(tail_logits, tf.float32)
            tail_logprobs = tf.nn.log_softmax(tail_logits)

            clust_start = self.cutoff[0] + i
            clust_logprob = root_logprobs[..., clust_start:clust_start + 1]
            tail_logprobs = tf.math.add(tail_logprobs, clust_logprob)
            full_logprobs.append(tail_logprobs)
        full_logprobs = tf.concat(full_logprobs, axis=-1)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=full_logprobs, labels=targets)
        loss = compute_weighted_loss(loss, sample_weight=None, reduction=self.loss_reduction)

        probs = tf.math.exp(full_logprobs)

        return probs, loss

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


@tf.keras.utils.register_keras_serializable(package='Miss')
class SampledSofmax(tf.keras.layers.Layer):
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

    def __init__(self,
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
        kwargs['dtype'] = 'float32'
        super(SampledSofmax, self).__init__(
            activity_regularizer=tf.keras.regularizers.get(activity_regularizer), **kwargs)
        self.input_spec = [
            tf.keras.layers.InputSpec(min_ndim=2),  # predictions
            tf.keras.layers.InputSpec(min_ndim=1),  # targets
        ]
        self.supports_masking = True
        self._supports_ragged_inputs = True

        tf.keras.losses.Reduction.validate(loss_reduction)

        self.units = units
        self.negatives = negatives
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)
        self.loss_reduction = loss_reduction

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

        with tf.device('cpu:0'):
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

        super(SampledSofmax, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        input_logits, input_targets = inputs
        input_logits = tf.cast(input_logits, self._compute_dtype)
        input_logits, row_lengths = convert_inputs_if_ragged(input_logits)
        input_targets, _ = convert_inputs_if_ragged(input_targets)
        is_ragged_input = (row_lengths is not None)

        input_shape = tf.shape(input_logits)
        output_shape = tf.stack(tf.unstack(input_shape)[:-1] + [self.units])
        input_logits = tf.reshape(input_logits, [-1, self.num_channels])
        input_targets = tf.reshape(input_targets, [-1])

        output_logits = tf.matmul(input_logits, self.kernel)
        output_logits = tf.nn.bias_add(output_logits, self.bias)
        output_logits = tf.cast(output_logits, tf.float32)

        loss = tf_utils.smart_cond(
            training,
            lambda: self._train_loss(input_logits, input_targets),
            lambda: self._eval_loss(output_logits, input_targets)
        )
        loss = compute_weighted_loss(loss, sample_weight=None, reduction=self.loss_reduction)
        self.add_loss(loss, inputs=True)

        output_probs = tf.nn.softmax(output_logits)
        output_probs = tf.reshape(output_probs, output_shape)
        output_probs = maybe_convert_to_ragged(is_ragged_input, output_probs, row_lengths)

        return output_probs

    def _train_loss(self, inputs, targets):
        labels_exp_dim = tf.expand_dims(targets, axis=-1)

        return tf.nn.sampled_softmax_loss(
            weights=tf.transpose(self.kernel),
            biases=self.bias,
            labels=labels_exp_dim,
            inputs=inputs,
            num_sampled=self.negatives,
            num_classes=self.units,
        )

    def _eval_loss(self, logits, targets):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)

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
        base_config = super(SampledSofmax, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package='Miss')
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

    def _train_loss(self, input_logits, input_targets):
        labels_exp_dim = tf.expand_dims(input_targets, axis=-1)
        loss = tf.nn.nce_loss(
            weights=tf.transpose(self.kernel),
            biases=self.bias,
            labels=labels_exp_dim,
            inputs=input_logits,
            num_sampled=self.negatives,
            num_classes=self.units,
        )

        return loss / tf.cast(1 + self.negatives, tf.float32)

    def _eval_loss(self, output_logits, input_targets):
        labels_one_hot = tf.one_hot(input_targets, self.units)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=output_logits,
            labels=labels_one_hot
        )

        return tf.reduce_sum(loss, axis=-1)
