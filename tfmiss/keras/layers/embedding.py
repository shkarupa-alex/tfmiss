from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.utils import tf_utils
from tfmiss.nn.embedding import adaptive_embedding_lookup


@tf.keras.utils.register_keras_serializable(package='Miss')
class AdaptiveEmbedding(Embedding):
    """Adaptive input embedding layer.
    Reference: https://arxiv.org/pdf/1809.10853v3.pdf
    Adaptive Input Representations for Neural Language Modeling
    Baevski and Auli (2018)
    """

    def __init__(self,
                 cutoff, input_dim, output_dim,
                 factor=4,
                 proj0=False,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        super(AdaptiveEmbedding, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            embeddings_initializer=embeddings_initializer,
            embeddings_regularizer=embeddings_regularizer,
            activity_regularizer=activity_regularizer,
            embeddings_constraint=embeddings_constraint,
            mask_zero=mask_zero,
            input_length=input_length,
            **kwargs)

        if cutoff[-1] > input_dim:
            raise ValueError('Can\'t specify cutoff larger than vocab size')

        self.cutoff = cutoff + [self.input_dim] if self.input_dim > cutoff[-1] else cutoff
        self._cutoff = cutoff
        self.factor = factor
        self.proj0 = proj0
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.embeddings = []
        self.projections = []

        prev_dim = None
        for i in range(len(self.cutoff)):
            prev = self.cutoff[i - 1] if i > 0 else 0
            size = self.cutoff[i] - prev
            dim = self.output_dim / (self.factor ** i)
            dim = max(1, round(dim / 8)) * 8

            if dim == prev_dim:
                raise ValueError('Some cutoffs have same embedding size.  Try to shorten `cutoffs`, '
                                 'decrease `factor` or increase `output_dim`')
            prev_dim = dim

            with tf.device('CPU:0'):
                embed = self.add_weight(
                    shape=(size, dim),
                    initializer=self.embeddings_initializer,
                    name='embeddings_{}'.format(i),
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint
                )
            setattr(self, 'embed_{}'.format(i), embed)
            self.embeddings.append(embed)

            if dim != self.output_dim or self.proj0:
                project = Dense(
                    units=self.output_dim,
                    activation=None,
                    use_bias=False,
                    kernel_initializer=self.kernel_initializer,
                    kernel_regularizer=self.kernel_regularizer,
                    activity_regularizer=self.activity_regularizer,
                    kernel_constraint=self.kernel_constraint
                )
                setattr(self, 'project_{}'.format(i), project)
            else:
                project = AdaptiveEmbedding._no_projection
            self.projections.append(project)

        self.built = True

    @staticmethod
    def _no_projection(embedding):
        return embedding

    def call(self, inputs):
        dtype = tf.keras.backend.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = tf.cast(inputs, 'int32')

        out = adaptive_embedding_lookup(self.embeddings, inputs, self.projections)

        return out

    def get_config(self):
        config = {
            'cutoff': self._cutoff,
            'factor': self.factor,
            'proj0': self.proj0,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
        }
        base_config = super(AdaptiveEmbedding, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
