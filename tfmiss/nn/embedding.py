from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops, sparse_tensor, tensor_shape
from tensorflow.python.ops import embedding_ops, data_flow_ops, resource_variable_ops, sparse_ops, variables
from tensorflow.python.platform import tf_logging as logging


def adaptive_embedding_lookup(params, ids, transforms, max_norm=None, name=None):
    """Looks up `ids` in a list of embedding tensors.
    This function is used to perform parallel lookups on the list of tensors in `params`.  It is a generalization of
    `tf.gather`, where `params` is interpreted as a partitioning of a large embedding tensor with different number
    of rows and columns.  `params` may be a `PartitionedVariable`s as returned by using `tf.compat.v1.get_variable()`
    with a partitioner.
    Each element `id` of `ids` is partitioned between the elements of `params` according to the `div` partition
    strategy.
    Id space should be the same with total number of `params` rows.
    The results of the lookup are transformed with corresponding functions from `transforms` and concatenated into a
    dense tensor. The returned tensor shape selected by `transforms`.
    Args:
        params: A list of P tensors of different shape, representing sharded embedding tensors.  Alternatively, a
            `PartitionedVariable`, created by partitioning along dimension 0.
        ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked up in `params`.
        transforms: Functions applied to each retrieved embedding before concatenation. Required due to possible
            different embedding sizes of `params`.
        max_norm: If not `None`, each embedding is clipped if its l2-norm is larger than this value.
        name: A name for the operation (optional).
    Returns:
        A `Tensor` with the same type as the tensors in `params`.
    """

    if isinstance(ids, tf.RaggedTensor):
        return tf.ragged.map_flat_values(adaptive_embedding_lookup, params, ids, transforms, max_norm, name)

    if not isinstance(params, (list, tuple)) or len(params) < 2:
        raise ValueError('At least 2 variables required in params')

    if not isinstance(transforms, (list, tuple)) or len(transforms) != len(params):
        raise ValueError('Each param should have corresponding transform')

    if not all([callable(t) for t in transforms]):
        raise ValueError('Each transform should be callable')

    with tf.name_scope(name or 'adaptive_embedding_lookup') as name:
        np = len(params)  # Number of partitions

        # Preserve the resource variable status to avoid accidental dense reads.
        if not any(isinstance(p, resource_variable_ops.ResourceVariable) for p in params):
            params = ops.convert_n_to_tensor_or_indexed_slices(params, name='params')
        ids = tf.convert_to_tensor(ids, name='ids')

        # Flatten the ids. There is more than one params tensor.
        flat_ids = tf.reshape(ids, [-1])
        original_indices = tf.range(tf.size(flat_ids))

        # Create p_assignments and set new_ids for adaptive strategy.
        # Compute total_ids_capacity as the sum of dim-0 of params, then assign to
        # partitions based on a variable number of ids per partition. Optimize
        # if we already know the full shape statically.
        dim_0_sizes = []
        for p in range(np):
            param_p_dim = tensor_shape.Dimension(tensor_shape.dimension_value(params[p].get_shape()[0]))
            dim_0_sizes.append(param_p_dim)

        dim_0_size_value = sum(dim_0_sizes).value
        if dim_0_size_value:
            dim_0_sizes = tf.TensorShape(dim_0_sizes).as_list()
            total_ids_capacity = tf.constant(dim_0_size_value, dtype=flat_ids.dtype)
        else:
            dim_0_sizes = []
            for p in range(np):
                param_p_dim = tensor_shape.dimension_value(params[p].get_shape()[0])
                if param_p_dim is not None:
                    dim_0_sizes.append(param_p_dim)
                else:
                    with ops.colocate_with(params[p]):
                        dim_0_sizes.append(tf.shape(params[p])[0])
            dim_0_sizes = tf.stack(dim_0_sizes)
            total_ids_capacity = tf.reduce_sum(dim_0_sizes)

        p_cumsum = tf.cumsum(tf.cast(dim_0_sizes, dtype=flat_ids.dtype))
        assert_max_id = tf.debugging.assert_less(
            tf.math.reduce_max(flat_ids),
            total_ids_capacity,
            'Invalid id. Maximum id should be less then total number of params rows'
        )
        with tf.control_dependencies([assert_max_id]):
            p_assignments = tf.searchsorted(
                p_cumsum,
                flat_ids,
                side='right',
            )

        # Cast partition assignments to int32 for use in dynamic_partition.
        # There really should not be more than 2^32 partitions.
        p_assignments = tf.cast(p_assignments, tf.int32)

        # Partition list of ids based on assignments into np separate lists
        p_intervals = tf.concat(([0], p_cumsum), 0)
        new_ids = flat_ids - tf.gather(p_intervals, p_assignments)
        gather_ids = tf.dynamic_partition(new_ids, p_assignments, np)

        # Similarly, partition the original indices.
        p_indices = tf.dynamic_partition(original_indices, p_assignments, np)

        # Do np separate lookups, finding embeddings for plist[p] in params[p]
        partitioned_result = []
        for p in range(np):
            pids = gather_ids[p]
            transform_fn = transforms[p]
            with ops.colocate_with(params[p]):
                result = tf.gather(params[p], pids)
                result = embedding_ops._clip(transform_fn(result), pids, max_norm)  # TODO check speed
            partitioned_result.append(result)

        # Stitch these back together
        ret = data_flow_ops.parallel_dynamic_stitch(p_indices, partitioned_result, name=name)  # TODO check speed

        # Determine the static element shape.
        element_shape_s = ret.get_shape()[1:]

        # Compute the dynamic element shape.
        if element_shape_s.is_fully_defined():
            element_shape_d = element_shape_s
        else:
            element_shape_d = tf.shape(ret)[1:]

        # Reshape to reverse the flattening of ids.
        ret = tf.reshape(ret, tf.concat([tf.shape(ids), element_shape_d], 0))

        # Normally the reshape is sufficient, but setting shape explicitly
        # teaches shape inference that params[1:].get_shape() matters.
        ret.set_shape(ids.get_shape().concatenate(element_shape_s))

        return ret


def adaptive_embedding_lookup_sparse(
        params, sp_ids, sp_weights, transforms, name=None, combiner=None, max_norm=None):
    """Computes embeddings for the given ids and weights.
    This op assumes that there is at least one id for each row in the dense tensor represented by sp_ids
    (i.e. there are no rows with empty features), and that all the indices of sp_ids are in canonical row-major order.
    It also assumes that all id values lie in the range [0, p0), where p0 is the sum of the size of params along
    dimension 0.
    Args:
        params: A list of P tensors all of same shape except for the first dimension, representing sharded embedding
            tensors.  Alternatively, a `PartitionedVariable`, created by partitioning along dimension 0. Each
            element must be appropriately sized for `div` partition strategy.
        sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size and M is arbitrary.
        sp_weights: either a `SparseTensor` of float / double weights, or `None` to indicate all weights should be
            taken to be 1. If specified, `sp_weights` must have exactly the same shape and indices as `sp_ids`.
        transforms: Functions applied to each retrieved embedding before concatenation. Required due to possible
            different embedding sizes of `params`.
        name: Optional name for the op.
        combiner: A string specifying the reduction op. Currently "mean", "sqrtn" and "sum" are supported. "sum"
            computes the weighted sum of the embedding results for each row. "mean" is the weighted sum divided by the
            total weight. "sqrtn" is the weighted sum divided by the square root of the sum of the squares of the
            weights.
        max_norm: If not `None`, each embedding is clipped if its l2-norm is larger than this value, before combining.
    Returns:
        A dense tensor representing the combined embeddings for the sparse ids. For each row in the dense tensor
        represented by `sp_ids`, the op looks up the embeddings for all ids in that row, multiplies them by the
        corresponding weight, and combines these embeddings as specified.
        In other words, if
            `shape(combined params) = [p0, p1, ..., pm]`
        and
            `shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]`
        then
            `shape(output) = [d0, d1, ..., dn-1, p1, ..., pm]`.
        For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are
            ```python
            [0, 0]: id 1, weight 2.0
            [0, 1]: id 3, weight 0.5
            [1, 0]: id 0, weight 1.0
            [2, 3]: id 1, weight 3.0
            ```
        with `combiner`="mean", then the output will be a 3x20 matrix where
            ```python
            output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
            output[1, :] = (params[0, :] * 1.0) / 1.0
            output[2, :] = (params[1, :] * 3.0) / 3.0
    """
    if combiner is None:
        logging.warn('The default value of combiner will change from "mean" to "sqrtn" after 2016/11/01.')
        combiner = 'mean'
    if combiner not in ('mean', 'sqrtn', 'sum'):
        raise ValueError('combiner must be one of "mean", "sqrtn" or "sum"')
    if isinstance(params, variables.PartitionedVariable):
        params = list(params)  # Iterate to get the underlying Variables.
    if not isinstance(params, list):
        params = [params]
    if not isinstance(sp_ids, sparse_tensor.SparseTensor):
        raise TypeError('sp_ids must be SparseTensor')
    ignore_weights = sp_weights is None
    if not ignore_weights:
        if not isinstance(sp_weights, sparse_tensor.SparseTensor):
            raise TypeError('sp_weights must be either None or SparseTensor')
        sp_ids.values.get_shape().assert_is_compatible_with(sp_weights.values.get_shape())
        sp_ids.indices.get_shape().assert_is_compatible_with(sp_weights.indices.get_shape())
        sp_ids.dense_shape.get_shape().assert_is_compatible_with(sp_weights.dense_shape.get_shape())

    with ops.name_scope(name or 'adaptive_embedding_lookup_sparse') as name:
        segment_ids = sp_ids.indices[:, 0]
        if segment_ids.dtype != tf.int32:
            segment_ids = tf.cast(segment_ids, tf.int32)

        ids = sp_ids.values
        ids, idx = tf.unique(ids)

        embeddings = adaptive_embedding_lookup(params, ids, transforms, max_norm=max_norm)
        if embeddings.dtype in (tf.float16, tf.bfloat16):
            embeddings = tf.cast(embeddings, tf.float32)
        if not ignore_weights:
            weights = sp_weights.values
            if weights.dtype != embeddings.dtype:
                weights = tf.cast(weights, embeddings.dtype)

            embeddings = tf.gather(embeddings, idx)

            # Reshape weights to allow broadcast
            ones = tf.fill(tf.expand_dims(tf.rank(embeddings) - 1, 0), 1)
            bcast_weights_shape = tf.concat([tf.shape(weights), ones], 0)

            orig_weights_shape = weights.get_shape()
            weights = tf.reshape(weights, bcast_weights_shape)

            # Set the weight shape, since after reshaping to bcast_weights_shape,
            # the shape becomes None.
            if embeddings.get_shape().ndims is not None:
                weights.set_shape(orig_weights_shape.concatenate(
                    [1 for _ in range(embeddings.get_shape().ndims - 1)]
                ))

            embeddings *= weights

            if combiner == 'sum':
                embeddings = tf.math.segment_sum(embeddings, segment_ids, name=name)
            elif combiner == 'mean':
                embeddings = tf.math.segment_sum(embeddings, segment_ids)
                weight_sum = tf.math.segment_sum(weights, segment_ids)
                embeddings = tf.math.divide(embeddings, weight_sum, name=name)
            elif combiner == 'sqrtn':
                embeddings = tf.math.segment_sum(embeddings, segment_ids)
                weights_squared = tf.math.pow(weights, 2)
                weight_sum = tf.math.segment_sum(weights_squared, segment_ids)
                weight_sum_sqrt = tf.math.sqrt(weight_sum)
                embeddings = tf.math.divide(embeddings, weight_sum_sqrt, name=name)
            else:
                assert False, 'Unrecognized combiner'
        else:
            assert idx is not None
            if combiner == 'sum':
                embeddings = tf.compat.v1.sparse_segment_sum(embeddings, idx, segment_ids, name=name)
            elif combiner == 'mean':
                embeddings = tf.compat.v1.sparse_segment_mean(embeddings, idx, segment_ids, name=name)
            elif combiner == 'sqrtn':
                embeddings = tf.compat.v1.sparse_segment_sqrt_n(embeddings, idx, segment_ids, name=name)
            else:
                assert False, 'Unrecognized combiner'

        return embeddings


def safe_adaptive_embedding_lookup_sparse(
        embedding_weights, sparse_ids, sparse_weights, transforms,
        combiner='mean', default_id=None, name=None, max_norm=None):
    """Lookup embedding results, accounting for invalid IDs and empty features.
    The partitioned embedding in `embedding_weights` may have different shape including the first dimension.
    `embedding_weights` may be a `PartitionedVariable` as returned by using `tf.compat.v1.get_variable()` with a
    partitioner.
    Invalid IDs (< 0) are pruned from input IDs and weights, as well as any IDs with non-positive weight. For an entry
    with no features, the embedding vector for `default_id` is returned, or the 0-vector if `default_id` is not
    supplied.
    The ids and weights may be multi-dimensional. Embeddings are always aggregated along the last dimension.
    Args:
        embedding_weights:  A list of `P` float `Tensor`s or values representing partitioned embedding `Tensor`s.
            Alternatively, a `PartitionedVariable` created by partitioning along dimension 0.  The total unpartitioned
            shape should be `[e_0, e_1, ..., e_m]`, where `e_0` represents the vocab size and `e_1, ..., e_m` are the
            embedding dimensions.
        sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the ids. `d_0` is typically batch size.
        sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing float weights corresponding to
            `sparse_ids`, or `None` if all weights are be assumed to be 1.0.
        combiner: A string specifying how to combine embedding results for each entry. Currently "mean", "sqrtn" and
            "sum" are supported, with "mean" the default.
        default_id: The id to use for an entry with no features.
        name: A name for this operation (optional).
        transforms: Functions applied to each retrieved embedding before concatenation. Required due to possible
            different embedding sizes of `embedding_weights`.
        max_norm: If not `None`, all embeddings are l2-normalized to max_norm before combining.
    Returns:
        Dense `Tensor` of shape `[d_0, d_1, ..., d_{n-1}, e_1, ..., e_m]`.
    """
    if embedding_weights is None:
        raise ValueError('Missing embedding_weights {}.'.format(embedding_weights))
    if isinstance(embedding_weights, variables.PartitionedVariable):
        embedding_weights = list(embedding_weights)  # get underlying Variables.
    if not isinstance(embedding_weights, list):
        embedding_weights = [embedding_weights]
    if len(embedding_weights) < 1:
        raise ValueError('Missing embedding_weights {}.'.format(embedding_weights))

    dtype = sparse_weights.dtype if sparse_weights is not None else None
    embedding_weights = [
        w if (isinstance(w, resource_variable_ops.ResourceVariable) and dtype in (None, w.dtype))
        else ops.convert_to_tensor(w, dtype=dtype)
        for w in embedding_weights
    ]

    with ops.name_scope(name or 'safe_adaptive_embedding_lookup_sparse') as scope:
        # Reshape higher-rank sparse ids and weights to linear segment ids.
        original_shape = sparse_ids.dense_shape
        original_rank_dim = tensor_shape.dimension_value(sparse_ids.dense_shape.get_shape()[0])
        original_rank = (tf.size(original_shape) if original_rank_dim is None else original_rank_dim)
        sparse_ids = sparse_ops.sparse_reshape(sparse_ids, [
            tf.reduce_prod(tf.slice(original_shape, [0], [original_rank - 1])),
            tf.gather(original_shape, original_rank - 1)
        ])
        if sparse_weights is not None:
            sparse_weights = sparse_tensor.SparseTensor(
                sparse_ids.indices, sparse_weights.values, sparse_ids.dense_shape)

        # Prune invalid ids and weights.
        sparse_ids, sparse_weights = embedding_ops._prune_invalid_ids(sparse_ids, sparse_weights)
        if combiner != 'sum':
            sparse_ids, sparse_weights = embedding_ops._prune_invalid_weights(sparse_ids, sparse_weights)

        # Fill in dummy values for empty features, if necessary.
        sparse_ids, is_row_empty = sparse_ops.sparse_fill_empty_rows(sparse_ids, default_id or 0)
        if sparse_weights is not None:
            sparse_weights, _ = sparse_ops.sparse_fill_empty_rows(sparse_weights, 1.0)

        result = adaptive_embedding_lookup_sparse(
            embedding_weights,
            sparse_ids,
            sparse_weights,
            transforms,
            combiner=combiner,
            name=None if default_id is None else scope,
            max_norm=max_norm
        )

        if default_id is None:
            # Broadcast is_row_empty to the same shape as embedding_lookup_result,
            # for use in Select.
            is_row_empty = tf.tile(
                tf.reshape(is_row_empty, [-1, 1]),
                tf.stack([1, tf.shape(result)[1]])
            )

            result = tf.where(is_row_empty, tf.zeros_like(result), result, name=scope)

        # Reshape back from linear ids back into higher-dimensional dense result.
        final_result = tf.reshape(
            result,
            tf.concat([
                tf.slice(tf.cast(original_shape, tf.int32), [0], [original_rank - 1]),
                tf.slice(tf.shape(result), [1], [-1])
            ], 0))
        final_result.set_shape(tensor_shape.unknown_shape(
            (tensor_shape.Dimension(original_rank_dim) - 1).value
        ).concatenate(result.get_shape()[1:]))

        return final_result
