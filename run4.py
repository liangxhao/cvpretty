def nansum(x, axis, keepdims=False, *, dtype=None):
    """
    Computes sum of all elements, treating Not a Numbers (NaNs) as zero.

    Args:
        x (Tensor) - The input tensor.
        axis (Union[int, tuple(int)]) - The dimensions to reduce. Must be in the range [-rank(`x`), rank(`x`)).
        keepdims (bool, optional) - Whether the output tensor has dim retained or not. Default: False.
        dtype (mindspore type, optional) - The desired data type of returned tensor. Default: None.

    Returns:
        Tensor, the sum of each row of the input tensor in the given dimension dim,
        treating Not a Numbers (NaNs) as zero.

        - If axis is (), keepdims is False,
          the output is a 0-D tensor representing the sum of all elements in the input tensor.
        - If axis is int, set as 2, and keepdims is False,
          the shape of output is :math:`(x_1, x_3, ..., x_R)`.
        - If axis is tuple(int) or list(int), set as (2, 3), and keepdims is False,
          the shape of output is :math:`(x_1, x_4, ..., x_R)`.
        - If x_dtype or dtype is complex type, nansum does not supported.
        - If x_dtype is floating-point type, and dtype is integer type, nansum does not supported.

    Raises:
        TypeError: If `x` is not tensor.
        TypeError: If `keepdims` is not a bool.
        TypeError: If x_dtype or dtype is complex type.
        TypeError: If x_dtype is floating-point type, and dtype is integer type.
        valueError: If 'axis' not in [-rank(`x`), rank(`x`)).

    Supported Platforms:
        ``CPU``

    Examples:
        >>> x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]), mindspore.float32)
        >>> axis = [0]
        >>> output = ops.nansum(x, axis, dtype=mindspore.float32)
        >>> print(output)
        [2. 4. 6.]
    """

    if not isinstance(x, (Tensor, Tensor_)):
        raise TypeError("For nansum, input must be Tensor.")
    res_dtype = dtype
    dtype_op = P.DType()
    x_dtype = dtype_op(x)

    if (x_dtype is not None and x_dtype in (mstype.complex64, mstype.complex128)) or \
            (dtype is not None and dtype in (mstype.complex64, mstype.complex128)):
        raise TypeError('nansum not supported complex type.')
    if x_dtype == mstype.bool_:
        x = x.astype(mstype.int64)

    if dtype is None:
        if x_dtype not in (mstype.float32, mstype.float16, mstype.float64):
            dtype = mstype.int64
        else:
            dtype = x_dtype
    if x_dtype in (mstype.float32, mstype.float16, mstype.float64):
        if dtype not in (mstype.float32, mstype.float16, mstype.float64):
            raise TypeError(f'nansum not supported for this dtype {dtype} when x_dtype is floa16, float32 or float64')
        get_nan = P.IsNan()(x)
        x = P.MaskedFill()(x, get_nan, Tensor(0.0, dtype=x_dtype))

    if x_dtype != dtype:
        x = x.astype(dtype)

    rank = len(x.shape)
    validator.check_value_type("axis", axis, [int, tuple, list], 'nansum')
    real_axis = axis if not isinstance(axis, int) else [axis]
    for i in real_axis:
        validator.check_int_range(i, -rank, rank, Rel.INC_LEFT, 'axis', 'nansum')
    real_axis = [i % rank for i in real_axis]
    repeated_real_axis = [i for i in real_axis if real_axis.count(i) > 1]
    if repeated_real_axis:
        raise ValueError(f"For nansum, axis {repeated_real_axis[0]} appears multiple times in the list of axis.")

    res = P.ReduceSum(keepdims)(x, axis)
    if (res_dtype is not None) and (res_dtype == mstype.bool_):
        res = res.astype(res_dtype)
    return res