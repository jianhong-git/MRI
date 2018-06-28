def conv3d_transpose(value,
                     filter,
                     output_shape,
                     strides,
                     padding="SAME",
                     data_format="NDHWC",
                     name=None):
  """The transpose of `conv3d`.
  This operation is sometimes called "deconvolution" after [Deconvolutional
  Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf), but is
  actually the transpose (gradient) of `conv3d` rather than an actual
  deconvolution.
  Args:
    value: A 5-D `Tensor` of type `float` and shape
      `[batch, depth, height, width, in_channels]`.
    filter: A 5-D `Tensor` with the same type as `value` and shape
      `[depth, height, width, output_channels, in_channels]`.  `filter`'s
      `in_channels` dimension must match that of `value`.
    output_shape: A 1-D `Tensor` representing the output shape of the
      deconvolution op.
    strides: A list of ints. The stride of the sliding window for each
      dimension of the input tensor.
    padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
      See the @{tf.nn.convolution$comment here}
    data_format: A string, either `'NDHWC'` or `'NCDHW`' specifying the layout
      of the input and output tensors. Defaults to `'NDHWC'`.
    name: Optional name for the returned tensor.
  Returns:
    A `Tensor` with the same type as `value`.
  Raises:
    ValueError: If input/output depth does not match `filter`'s shape, or if
      padding is other than `'VALID'` or `'SAME'`.
  """
  with ops.name_scope(name, "conv3d_transpose",
                      [value, filter, output_shape]) as name:
    value = ops.convert_to_tensor(value, name="value")
    filter = ops.convert_to_tensor(filter, name="filter")
    axis = 1 if data_format == "NCDHW" else 4
    if not value.get_shape()[axis].is_compatible_with(filter.get_shape()[4]):
      raise ValueError("input channels does not match filter's input channels, "
                       "{} != {}".format(value.get_shape()[axis],
                                         filter.get_shape()[4]))

    output_shape_ = ops.convert_to_tensor(output_shape, name="output_shape")
    if not output_shape_.get_shape().is_compatible_with(tensor_shape.vector(5)):
      raise ValueError("output_shape must have shape (5,), got {}"
                       .format(output_shape_.get_shape()))

    if isinstance(output_shape, (list, np.ndarray)):
      # output_shape's shape should be == [5] if reached this point.
      if not filter.get_shape()[3].is_compatible_with(output_shape[4]):
        raise ValueError(
            "output_shape does not match filter's output channels, "
            "{} != {}".format(output_shape[4], filter.get_shape()[3]))

    if padding != "VALID" and padding != "SAME":
      raise ValueError("padding must be either VALID or SAME:"
                       " {}".format(padding))

    return gen_nn_ops.conv3d_backprop_input_v2(input_sizes=output_shape_,
                                               filter=filter,
                                               out_backprop=value,
                                               strides=strides,
                                               padding=padding,
                                               data_format=data_format,
                                               name=name)


def conv3d_backprop_input_v2(input_sizes, filter, out_backprop, strides,
                             padding, name=None):
  r"""Computes the gradients of 3-D convolution with respect to the input.
 
  Args:
    input_sizes: A `Tensor` of type `int32`.
      An integer vector representing the tensor shape of `input`,
      where `input` is a 5-D
      `[batch, depth, rows, cols, in_channels]` tensor.
    filter: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
      Shape `[depth, rows, cols, in_channels, out_channels]`.
      `in_channels` must match between `input` and `filter`.
    out_backprop: A `Tensor`. Must have the same type as `filter`.
      Backprop signal of shape `[batch, out_depth, out_rows, out_cols,
      out_channels]`.
    strides: A list of `ints` that has length `>= 5`.
      1-D tensor of length 5. The stride of the sliding window for each
      dimension of `input`. Must have `strides[0] = strides[4] = 1`.
    padding: A `string` from: `"SAME", "VALID"`.
      The type of padding algorithm to use.
    name: A name for the operation (optional).
 
  Returns:
    A `Tensor`. Has the same type as `filter`.
  """
  result = _op_def_lib.apply_op("Conv3DBackpropInputV2",
                                input_sizes=input_sizes, filter=filter,
                                out_backprop=out_backprop, strides=strides,
                                padding=padding, name=name)
  return result


  op {
  name: "Conv3DBackpropInputV2"
  input_arg {
    name: "input_sizes"
    type: DT_INT32
  }
  input_arg {
    name: "filter"
    type_attr: "T"
  }
  input_arg {
    name: "out_backprop"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
        type: DT_INT64
        type: DT_INT32
        type: DT_UINT8
        type: DT_UINT16
        type: DT_INT16
        type: DT_INT8
        type: DT_COMPLEX64
        type: DT_COMPLEX128
        type: DT_QINT8
        type: DT_QUINT8
        type: DT_QINT32
        type: DT_HALF
      }
    }
  }
  attr {
    name: "strides"
    type: "list(int)"
    has_minimum: true
    minimum: 5
  }
  attr {
    name: "padding"
    type: "string"
    allowed_values {
      list {
        s: "SAME"
        s: "VALID"
      }
    }
  }
}                                               