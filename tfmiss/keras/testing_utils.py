from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.testing_utils import _thread_local_data, should_run_eagerly
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.util import tf_inspect


@test_util.use_deterministic_cudnn
def layer_multi_io_test(layer_cls, kwargs=None, input_shapes=None, input_dtypes=None,
                        input_datas=None, expected_outputs=None, expected_output_dtypes=None):
    """Test routine for a layer with a single or multiple inputs and single or multiple outputs.

    Arguments:
      layer_cls: Layer class object.
      kwargs: Optional dictionary of keyword arguments for instantiating the
        layer.
      input_shapes: Input shape tuples.
      input_dtypes: Data types of the input data.
      input_datas: Numpy arrays of input data.
      expected_outputs: Shape tuples for the expected shape of the output.
      expected_output_dtypes: Data types expected for the output.

    Returns:
      The output data (Numpy arrays) returned by the layer,
      for additional checks to be done by the calling code.

    Raises:
      ValueError: if one of `input_shape` subshapes is None.
    """

    if input_shapes is None:
        input_shapes = [input_shapes]
    elif not isinstance(input_shapes, (list, tuple)):
        raise ValueError('input_shapes should be a list or None')
    if input_dtypes is None:
        input_dtypes = [input_dtypes]
    elif not isinstance(input_dtypes, (list, tuple)):
        raise ValueError('input_dtypes should be a list or None')
    if input_datas is None:
        input_datas = [input_datas]
    elif not isinstance(input_datas, (list, tuple)):
        raise ValueError('input_datas should be a list or None')
    if expected_outputs is None:
        expected_outputs = [expected_outputs]
    elif not isinstance(expected_outputs, (list, tuple)):
        raise ValueError('expected_outputs should be a list or None')
    if expected_output_dtypes is None:
        expected_output_dtypes = [input_dtypes[0]]
    elif not isinstance(expected_output_dtypes, (list, tuple)):
        raise ValueError('expected_output_dtypes should be a list or None')

    input_size = max([len(input_shapes), len(input_dtypes), len(input_datas)])
    if len({len(input_shapes), len(input_dtypes), len(input_datas)} - {1, input_size}):
        raise ValueError('input_shapes, input_dtypes, input_datas, expected_outputs '
                         'and expected_output_dtypes should have same size if provided')

    input_shapes = input_shapes * (input_size // len(input_shapes))
    input_dtypes = input_dtypes * (input_size // len(input_dtypes))
    input_datas = input_datas * (input_size // len(input_datas))

    output_size = max([len(expected_outputs), len(expected_output_dtypes)])
    expected_outputs = expected_outputs * (output_size // len(expected_outputs))
    expected_output_dtypes = expected_output_dtypes * (output_size // len(expected_output_dtypes))

    for i in range(input_size):
        input_shape = input_shapes[i]
        input_dtype = input_dtypes[i]
        input_data = input_datas[i]

        if input_data is None:
            if input_shape is None:
                raise ValueError('input_shape is None')
            if not input_dtype:
                input_dtype = 'float32'
            input_data_shape = list(input_shape)
            for j, e in enumerate(input_data_shape):
                if e is None:
                    input_data_shape[j] = np.random.randint(1, 4)
            input_data = 10 * np.random.random(input_data_shape)
            if input_dtype[:5] == 'float':
                input_data -= 0.5
            input_data = input_data.astype(input_dtype)
        elif input_shape is None:
            input_shape = input_data.shape
        if input_dtype is None:
            input_dtype = input_data.dtype

        input_shapes[i] = input_shape
        input_dtypes[i] = input_dtype
        input_datas[i] = input_data

    for i in range(output_size):
        output_dtype = expected_output_dtypes[i]
        if output_dtype is None:
            output_dtype = input_dtypes[0]
        expected_output_dtypes[i] = output_dtype

    # instantiation
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    # test get_weights , set_weights at layer level
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
        kwargs['weights'] = weights
        layer = layer_cls(**kwargs)

    # test in functional API
    x = [keras.layers.Input(shape=input_shapes[i][1:], dtype=input_dtypes[i]) for i in range(input_size)]
    x = x if input_size > 1 else x[0]
    y = layer(x)
    if isinstance(y, (list, tuple)) and output_size == 1:
        raise ValueError('expected_outputs or expected_output_dtypes should be '
                         'provided for multi-output models')
    y = y if output_size > 1 else [y]
    for i in range(output_size):
        if keras.backend.dtype(y[i]) != expected_output_dtypes[i]:
            raise AssertionError('When testing layer %s, for input %s, found output '
                                 'dtype=%s but expected to find %s.\nFull kwargs: %s' %
                                 (layer_cls.__name__,
                                  x,
                                  keras.backend.dtype(y[i]),
                                  expected_output_dtypes[i],
                                  kwargs))
    y = y if output_size > 1 else y[0]

    # check shape inference
    model = keras.models.Model(x, y)
    actual_outputs = model.predict(input_datas if input_size > 1 else input_datas[0])
    actual_outputs = actual_outputs if output_size > 1 else [actual_outputs]
    input_tensor_shapes = [tensor_shape.TensorShape(shape) for shape in input_shapes]
    input_tensor_shapes = input_tensor_shapes if input_size > 1 else input_tensor_shapes[0]
    expected_output_shapes = layer.compute_output_shape(input_tensor_shapes)
    expected_output_shapes = expected_output_shapes if output_size > 1 else [expected_output_shapes]
    expected_output_shapes = [tuple(shape.as_list()) for shape in expected_output_shapes]
    actual_output_shapes = [output.shape for output in actual_outputs]

    for expected_output_shape, actual_output_shape in zip(expected_output_shapes, actual_output_shapes):
        for expected_dim, actual_dim in zip(expected_output_shape, actual_output_shape):
            if expected_dim is not None:
                if expected_dim != actual_dim:
                    raise AssertionError(
                        'When testing layer %s, for input %s, found output_shape='
                        '%s but expected to find %s.\nFull kwargs: %s' %
                        (layer_cls.__name__,
                         x,
                         actual_output_shape,
                         expected_output_shape,
                         kwargs))
    for expected_output, actual_output in zip(expected_outputs, actual_outputs):
        if expected_output is not None:
            np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization, weight setting at model level
    model_config = model.get_config()
    recovered_model = keras.models.Model.from_config(model_config)
    if model.weights:
        weights = model.get_weights()
        recovered_model.set_weights(weights)
        outputs = recovered_model.predict(input_datas if input_size > 1 else input_datas[0])
        outputs = outputs if output_size > 1 else [outputs]
        for output, actual_output in zip(outputs, actual_outputs):
            np.testing.assert_allclose(output, actual_output, rtol=2e-3)

    # test training mode (e.g. useful for dropout tests)
    # Rebuild the model to avoid the graph being reused between predict() and
    # train(). This was causing some error for layer with Defun as it body.
    # See b/120160788 for more details. This should be mitigated after 2.0.
    model = keras.models.Model(x, layer(x))
    if _thread_local_data.run_eagerly is not None:
        model.compile(
            'rmsprop',
            'mse',
            weighted_metrics=['acc'],
            run_eagerly=should_run_eagerly())
    else:
        model.compile('rmsprop', 'mse', weighted_metrics=['acc'])
    model.train_on_batch(
        input_datas if input_size > 1 else input_datas[0],
        actual_outputs if output_size > 1 else actual_outputs[0]
    )

    # Sequential models do not support multiple inputs and outputs
    # # test as first layer in Sequential API
    # layer_config = layer.get_config()
    # layer_config['batch_input_shape'] = input_shapes if input_size > 1 else input_shapes[0]
    # layer = layer.__class__.from_config(layer_config)
    #
    # model = keras.models.Sequential()
    # model.add(layer)
    # actual_outputs = model.predict(input_datas if input_size > 1 else input_datas[0])
    # actual_outputs = actual_outputs if output_size > 1 else [actual_outputs]
    # actual_output_shapes = [output.shape for output in actual_outputs]
    #
    # for expected_output_shape, actual_output_shape in zip(expected_output_shapes, actual_output_shapes):
    #     for expected_dim, actual_dim in zip(expected_output_shape, actual_output_shape):
    #         if expected_dim is not None:
    #             if expected_dim != actual_dim:
    #                 raise AssertionError(
    #                     'When testing layer %s **after deserialization**, '
    #                     'for input %s, found output_shape='
    #                     '%s but expected to find inferred shape %s.\nFull kwargs: %s' %
    #                     (layer_cls.__name__,
    #                      x,
    #                      actual_output_shape,
    #                      expected_output_shape,
    #                      kwargs))
    # for expected_output, actual_output in zip(expected_outputs, actual_outputs):
    #     if expected_output is not None:
    #         np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3)
    #
    # # test serialization, weight setting at model level
    # model_config = model.get_config()
    # recovered_model = keras.models.Sequential.from_config(model_config)
    # if model.weights:
    #     weights = model.get_weights()
    #     recovered_model.set_weights(weights)
    #     outputs = recovered_model.predict(input_datas if input_size > 1 else input_datas[0])
    #     outputs = outputs if output_size > 1 else [outputs]
    #     for output, actual_output in zip(outputs, actual_outputs):
    #         np.testing.assert_allclose(output, actual_output, rtol=2e-3)

    # for further checks in the caller function
    return actual_outputs if output_size > 1 else actual_outputs[0]
