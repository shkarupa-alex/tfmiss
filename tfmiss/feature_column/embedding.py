from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import tensorflow as tf
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.framework import ops
from tensorflow.python.keras import initializers
from tfmiss.nn.embedding import safe_adaptive_embedding_lookup_sparse


def adaptive_embedding_column(
        categorical_column, cutoff, dimension, factor=4, proj0=False,
        combiner='mean', initializer=None, projection_initializer=None, max_norm=None, trainable=True):
    """`DenseColumn` that converts from sparse, categorical input.
    Use this when your inputs are sparse, but you want to convert them to a dense representation (e.g., to feed to a
    DNN).
    Inputs must be a `CategoricalColumn` created by any of the `categorical_column_*` function.
    Args:
        categorical_column: A `CategoricalColumn` created by a `categorical_column_with_*` function. This column
            produces the sparse IDs that are inputs to the embedding lookup.
        cutoff: A list of split sizes for embeddings.
        dimension: An integer specifying dimension of the embedding, must be > 0.
        factor: A split factor used to calculate reduced embedding size di = d0 / (factor**i).
        proj0: If false, no projection will be applied to the head embedding
        combiner: A string specifying how to reduce if there are multiple entries in a single row. Currently 'mean',
            'sqrtn' and 'sum' are supported, with 'mean' the default. 'sqrtn' often achieves good accuracy, in
            particular with bag-of-words columns. Each of this can be thought as example level normalizations on the
            column. For more information, see `tf.embedding_lookup_sparse`.
        initializer: A variable initializer function to be used in embedding variable initialization. If not specified,
            defaults to `truncated_normal_initializer` with mean `0.0` and standard deviation `1/sqrt(dimension)`.
        projection_initializer: A variable initializer function to be used in projection variable initialization. If
            not specified, defaults to `glorot_uniform_initializer`.
        max_norm: If not `None`, embedding values are l2-normalized to this value.
        trainable: Whether or not the embedding is trainable. Default is True.
    Returns:
        `DenseColumn` that converts from sparse input.
    """
    if not isinstance(categorical_column, (
            fc.VocabularyListCategoricalColumn, fc.VocabularyFileCategoricalColumn,
            fc_old._VocabularyListCategoricalColumn, fc_old._VocabularyFileCategoricalColumn,
    )):
        raise ValueError('Adaptive embedding column supports only categorical columns with vocabulary')
    if 0 != categorical_column.num_oov_buckets:
        raise ValueError('Adaptive embedding column does not support OOV buckets')
    if 0 != categorical_column.default_value:
        raise ValueError('Adaptive embedding column supports only 0 as default value for OOV labels')

    if (dimension is None) or (dimension < 1):
        raise ValueError('Invalid dimension {}.'.format(dimension))
    if (initializer is not None) and (not callable(initializer)):
        raise ValueError('initializer must be callable if specified.')
    if initializer is None:
        initializer = tf.compat.v1.truncated_normal_initializer(
            mean=0.0, stddev=1 / tf.math.sqrt(float(dimension)))

    if (projection_initializer is not None) and (not callable(projection_initializer)):
        raise ValueError('projection_initializer must be callable if specified.')
    if projection_initializer is None:
        projection_initializer = tf.compat.v1.glorot_uniform_initializer()

    return AdaptiveEmbeddingColumn(
        categorical_column=categorical_column,
        cutoff=tuple(cutoff),
        dimension=dimension,
        factor=factor,
        proj0=proj0,
        combiner=combiner,
        initializer=initializer,
        projection_initializer=projection_initializer,
        max_norm=max_norm,
        trainable=trainable
    )


class AdaptiveEmbeddingColumn(
    collections.namedtuple('AdaptiveEmbeddingColumn', (
            'categorical_column', 'cutoff', 'dimension', 'factor', 'proj0',
            'combiner', 'initializer', 'projection_initializer', 'max_norm', 'trainable')),
    fc.EmbeddingColumn):
    """See `adaptive_embedding_column`."""

    @property
    def name(self):
        """See `FeatureColumn` base class."""
        return '{}_adaptive_embedding'.format(self.categorical_column.name)

    def _get_cutoff(self):
        default_num_buckets = (
            self.categorical_column.num_buckets if self._is_v2_column
            else self.categorical_column._num_buckets
        )  # pylint: disable=protected-access
        num_buckets = getattr(self.categorical_column, 'num_buckets', default_num_buckets)

        if num_buckets > self.cutoff[-1]:
            cutoff = self.cutoff + (num_buckets,)
        else:
            cutoff = self.cutoff

        return cutoff

    def create_state(self, state_manager):
        """Creates the embedding lookup variable."""
        cutoff = self._get_cutoff()

        prev_dim = None
        for i in range(len(cutoff)):
            prev = cutoff[i - 1] if i > 0 else 0
            size = cutoff[i] - prev
            dim = self.dimension // (self.factor ** i)
            dim = max(1, round(dim / 8)) * 8

            if dim == prev_dim:
                raise ValueError('Some cutoffs have same embedding size. '
                                 'Try to shorten `cutoffs`, decrease `factor` or increase `dimension`')
            prev_dim = dim

            state_manager.create_variable(
                self,
                name='embedding_weights_{}'.format(i),
                shape=(size, dim),
                dtype=tf.float32,
                trainable=self.trainable,
                use_resource=True,
                initializer=self.initializer
            )

            if dim != self.dimension or self.proj0:
                state_manager.create_variable(
                    self,
                    name='embedding_projections_{}'.format(i),
                    shape=(dim, self.dimension),
                    dtype=tf.float32,
                    trainable=self.trainable,
                    use_resource=True,
                    initializer=self.projection_initializer
                )

    def _get_dense_tensor_internal_helper(self, sparse_tensors, embedding_weights, embedding_projections):
        sparse_ids = sparse_tensors.id_tensor
        sparse_weights = sparse_tensors.weight_tensor

        # Prepare transforms
        transforms = [TransformProxy(p) for p in embedding_projections]

        # Return embedding lookup result.
        return safe_adaptive_embedding_lookup_sparse(
            embedding_weights=embedding_weights,
            sparse_ids=sparse_ids,
            sparse_weights=sparse_weights,
            transforms=transforms,
            combiner=self.combiner,
            name='{}_weights'.format(self.name),
            max_norm=self.max_norm
        )

    def _get_dense_tensor_internal(self, sparse_tensors, state_manager):
        """Private method that follows the signature of get_dense_tensor."""
        cutoff = self._get_cutoff()

        embedding_weights, embedding_projections = [], []
        for i in range(len(cutoff)):
            embedding_weights.append(state_manager.get_variable(self, name='embedding_weights_{}'.format(i)))

            dim = self.dimension // (self.factor ** i)
            dim = max(1, round(dim / 8)) * 8
            if dim != self.dimension or self.proj0:
                embedding_projections.append(state_manager.get_variable(
                    self, name='embedding_projections_{}'.format(i)))
            else:
                embedding_projections.append(None)

        return self._get_dense_tensor_internal_helper(sparse_tensors, embedding_weights, embedding_projections)

    def _old_get_dense_tensor_internal(self, sparse_tensors, weight_collections, trainable):
        """Private method that follows the signature of _get_dense_tensor."""
        if weight_collections and ops.GraphKeys.GLOBAL_VARIABLES not in weight_collections:
            weight_collections.append(ops.GraphKeys.GLOBAL_VARIABLES)

        cutoff = self._get_cutoff()

        prev_dim = None
        embedding_weights, embedding_projections = [], []
        for i in range(len(cutoff)):
            prev = cutoff[i - 1] if i > 0 else 0
            size = cutoff[i] - prev
            dim = self.dimension // (self.factor ** i)
            dim = max(1, round(dim / 8)) * 8

            if dim == prev_dim:
                raise ValueError('Some cutoffs have same embedding size. '
                                 'Try to shorten `cutoffs`, decrease `factor` or increase `dimension`.')
            prev_dim = dim

            embedding_weights.append(tf.compat.v1.get_variable(
                name='embedding_weights_{}'.format(i),
                shape=(size, dim),
                dtype=tf.float32,
                initializer=self.initializer,
                trainable=self.trainable and trainable,
                collections=weight_collections
            ))
            if dim != self.dimension or self.proj0:
                embedding_projections.append(tf.compat.v1.get_variable(
                    name='embedding_projections_{}'.format(i),
                    shape=(dim, self.dimension),
                    dtype=tf.float32,
                    initializer=self.projection_initializer,
                    trainable=self.trainable and trainable,
                    collections=weight_collections
                ))
            else:
                embedding_projections.append(None)

        return self._get_dense_tensor_internal_helper(sparse_tensors, embedding_weights, embedding_projections)

    def _get_config(self):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import \
            serialize_feature_column  # pylint: disable=g-import-not-at-top

        config = dict(zip(self._fields, self))
        config['categorical_column'] = serialize_feature_column(self.categorical_column)
        config['initializer'] = initializers.serialize(self.initializer)
        config['projection_initializer'] = initializers.serialize(self.projection_initializer)

        return config

    @classmethod
    def _from_config(cls, config, custom_objects=None, columns_by_name=None):
        """See 'FeatureColumn` base class."""
        from tensorflow.python.feature_column.serialization import \
            deserialize_feature_column  # pylint: disable=g-import-not-at-top
        from tensorflow.python.feature_column.feature_column_v2 import \
            _check_config_keys, _standardize_and_copy_config  # pylint: disable=g-import-not-at-top

        _check_config_keys(config, cls._fields)
        kwargs = _standardize_and_copy_config(config)
        kwargs['categorical_column'] = deserialize_feature_column(
            config['categorical_column'], custom_objects, columns_by_name)
        kwargs['initializer'] = initializers.deserialize(
            config['initializer'], custom_objects=custom_objects)
        kwargs['projection_initializer'] = initializers.deserialize(
            config['projection_initializer'], custom_objects=custom_objects)

        return cls(**kwargs)


class TransformProxy:
    def __init__(self, projection):
        self.projection = projection

    def __call__(self, inputs):
        if self.projection is None:
            return inputs

        return tf.matmul(inputs, self.projection)
