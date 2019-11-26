from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.eager import context
from tensorflow.python.keras.utils import tf_utils
from tfmiss.nn.embedding import adaptive_embedding_lookup


class AdaptiveEmbedding(Embedding):
    """Adaptive input embedding layer.
    Reference: https://arxiv.org/pdf/1809.10853v3.pdf
    Adaptive Input Representations for Neural Language Modeling
    Baevski and Auli (2018)
    """

    def __init__(self,
                 cutoff, input_dim, output_dim,
                 factor=4,
                 mod8=True,
                 proj0=False,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None,
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

        self.cutoff = cutoff
        self.factor = factor
        self.mod8 = mod8
        self.proj0 = proj0
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        self.embeddings = []
        self.projections = []

        if self.input_dim > self.cutoff[-1]:
            cutoff = self.cutoff + [self.input_dim]
        else:
            cutoff = self.cutoff

        prev_dim = None
        for i in range(len(cutoff)):
            prev = cutoff[i - 1] if i > 0 else 0
            size = cutoff[i] - prev
            denom = 8 if self.mod8 else 1
            out = int(self.output_dim // (self.factor ** i))
            out = int(np.ceil(out / denom)) * denom
            dim = max(denom, out)

            if dim != prev_dim:
                prev_dim = dim
            else:
                raise ValueError('Some cutoffs have same embedding size. '
                                 'Try to shorten `cutoffs`, decrease `factor` or increase `output_dim`')

            # Note: most sparse optimizers do not have GPU kernels defined. When
            # building graphs, the placement algorithm is able to place variables on CPU
            # since it knows all kernels using the variable only exist on CPU.
            # When eager execution is enabled, the placement decision has to be made
            # right now. Checking for the presence of GPUs to avoid complicating the
            # TPU codepaths which can handle sparse optimizers.
            if context.executing_eagerly() and context.context().num_gpus():
                with tf.device('cpu:0'):
                    embed = self.add_weight(
                        shape=(size, dim),
                        initializer=self.embeddings_initializer,
                        name='embeddings_{}'.format(i),
                        regularizer=self.embeddings_regularizer,
                        constraint=self.embeddings_constraint
                    )
            else:
                embed = self.add_weight(
                    shape=(size, dim),
                    initializer=self.embeddings_initializer,
                    name='embeddings_{}'.format(i),
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint
                )
            setattr(self, 'embed_{}'.format(i), embed)
            self.embeddings.append(embed)

            if 0 == i and not self.proj0:
                project = AdaptiveEmbedding._no_projection
            else:
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
            'cutoff': self.cutoff,
            'factor': self.factor,
            'mod8': self.mod8,
            'proj0': self.proj0,
            'kernel_initializer': tf.keras.initializers.serialize(self.kernel_initializer),
            'kernel_regularizer': tf.keras.regularizers.serialize(self.kernel_regularizer),
            'kernel_constraint': tf.keras.constraints.serialize(self.kernel_constraint),
        }
        base_config = super(AdaptiveEmbedding, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
