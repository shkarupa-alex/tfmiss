from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils


class NoiseContrastiveEstimation(tf.keras.layers.Layer):
    """Computes the noise-contrastive estimation training loss.
    See [Noise-contrastive estimation: A new estimation principle for unnormalized statistical models]
    (http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
    Also see our [Candidate Sampling Algorithms Reference](https://www.tensorflow.org/extras/candidate_sampling.pdf)
    The full sigmoid loss calculated for evaluation.
    Note: By default this uses a log-uniform (Zipfian) distribution for sampling, so your labels must be sorted in
    order of decreasing frequency to achieve good results.  For more details, see
    `tf.random.log_uniform_candidate_sampler`.
    Args:
        num_classes: An `int`. The number of possible classes.
        num_negative: An `int`.  The number of negative classes to randomly sample per batch. This single sample of
            negative classes is evaluated for each element in the batch.
        dtype: A layer's weights dtype (optional).
        name: A name for the operation (optional).
    Returns:
        A `batch_size` 1-D tensor of per-example NCE losses.
    """

    def __init__(self, num_classes, num_negative, dtype=None, name='noise_contrastive_estimation'):
        self.num_classes = num_classes
        self.num_negative = num_negative

        dtype = tf.dtypes.as_dtype(dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError('Unable to build `NoiseContrastiveEstimation` layer '
                            'with non-floating point dtype {}'.format(dtype))

        super(NoiseContrastiveEstimation, self).__init__(dtype=dtype, name=name)
        self.input_spec = [
            tf.keras.layers.InputSpec(ndim=2),  # predictions
            tf.keras.layers.InputSpec(ndim=2, axes={-1: 1}),  # targets
        ]

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError('A `NoiseContrastiveEstimation` layer should be called '
                             'on exactly 2 inputs: `predictions` and `targets`')
        predictions_shape, targets_shape = input_shape

        if len(predictions_shape) != 2:
            raise ValueError('Predictions shape {} must have rank 2'.format(predictions_shape))

        if len(targets_shape) != 2:
            raise ValueError('Targets shape {} must have rank 1'.format(targets_shape))

        num_channels = predictions_shape[-1]
        if num_channels is None:
            raise ValueError('Channel dimension of predictions should be defined. Found `None`.')

        self.input_spec = [
            tf.keras.layers.InputSpec(ndim=2, axes={-1: num_channels}),
            tf.keras.layers.InputSpec(ndim=2, axes={-1: 1})
        ]

        # Note: most sparse optimizers do not have GPU kernels defined. When
        # building graphs, the placement algorithm is able to place variables on CPU
        # since it knows all kernels using the variable only exist on CPU.
        # When eager execution is enabled, the placement decision has to be made
        # right now. Checking for the presence of GPUs to avoid complicating the
        # TPU codepaths which can handle sparse optimizers.
        if context.executing_eagerly() and context.context().num_gpus():
            with tf.device('cpu:0'):
                self.kernel = self.add_weight(
                    shape=(num_channels, self.num_classes),
                    initializer='zeros',
                    name='nce_kernel',
                )
                self.bias = self.add_weight(
                    shape=(self.num_classes,),
                    initializer='zeros',
                    name='nce_bias',
                )
        else:
            self.kernel = self.add_weight(
                shape=(num_channels, self.num_classes),
                initializer='zeros',
                name='nce_kernel',
            )
            self.bias = self.add_weight(
                shape=(self.num_classes,),
                initializer='zeros',
                name='nce_bias',
            )

        super(NoiseContrastiveEstimation, self).build(input_shape)

    def call(self, inputs, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        predictions, targets = inputs

        logits = tf.matmul(predictions, self.kernel)
        logits = tf.nn.bias_add(logits, self.bias)

        def _train_loss():
            # TODO: tf.compat.v2.nn.sampled_softmax_loss
            return tf.compat.v1.nn.nce_loss(
                weights=tf.transpose(self.kernel),
                biases=self.bias,
                labels=targets,
                inputs=predictions,
                num_sampled=self.num_negative,
                num_classes=self.num_classes,
                # partition_strategy='div'
            )

        def _test_loss():
            labels_one_hot = tf.one_hot(tf.squeeze(targets, axis=-1), self.num_classes)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=labels_one_hot,
                logits=logits
            )

            return tf.reduce_sum(loss, axis=-1)

        loss = tf_utils.smart_cond(training,
                                   _train_loss,
                                   _test_loss)
        # if training:
        #     loss = tf.compat.v1.nn.nce_loss(
        #         weights=self.kernel,
        #         biases=self.bias,
        #         labels=targets,
        #         inputs=predictions,
        #         num_sampled=self.num_negative,
        #         num_classes=self.num_classes,
        #         partition_strategy='div'
        #     )
        # else:
        #     labels_one_hot = tf.one_hot(targets, self.num_classes)
        #     loss = tf.nn.sigmoid_cross_entropy_with_logits(
        #         labels=labels_one_hot,
        #         logits=logits
        #     )
        #     loss = tf.reduce_sum(loss, axis=-1)
        self.add_loss(loss, inputs=True)

        return logits

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError('A `NoiseContrastiveEstimation` layer should be called '
                             'on exactly 2 inputs: `predictions` and `targets`')
        predictions_shape, _ = input_shape

        if len(predictions_shape) != 2:
            raise ValueError('Predictions shape {} must have rank 2'.format(predictions_shape))

        return predictions_shape[:1] + (self.num_classes,)

    def get_config(self):
        config = {
            'num_classes': self.num_classes,
            'num_negative': self.num_negative,
        }
        base_config = super(NoiseContrastiveEstimation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
