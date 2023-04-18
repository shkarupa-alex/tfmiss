from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import indexed_slices, ops, tensor_shape
from tensorflow.python.ops import embedding_ops, data_flow_ops, resource_variable_ops


def adaptive_embedding_lookup(params, ids, transforms, max_norm=None, name=None):
    """Looks up `ids` in a list of embedding tensors.
    This function is used to perform parallel lookups on the list of tensors in `params`.  It is a generalization of
    `tf.gather`, where `params` is interpreted as a partitioning of a large embedding tensor with different number
    of rows and columns. `params` may be a `PartitionedVariable`s as returned by using `tf.compat.v1.get_variable()`
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
            params = indexed_slices.convert_n_to_tensor_or_indexed_slices(params, name='params')
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
                result = embedding_ops._clip(transform_fn(result), pids, max_norm)
            partitioned_result.append(result)

        # Stitch these back together
        ret = data_flow_ops.parallel_dynamic_stitch(p_indices, partitioned_result, name=name)

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
