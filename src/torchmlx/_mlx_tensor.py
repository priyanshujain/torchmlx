import mlx.core as mx


_transpose = mx.array.transpose
_reshape = mx.array.reshape
_squeeze = mx.array.squeeze
_mean = mx.array.mean
_var = mx.array.var
_any = mx.array.any
_getitem = mx.array.__getitem__
_setitem = mx.array.__setitem__
_item = mx.array.item


def _torch_transpose(self, dim0=None, dim1=None):
    if dim1 is None:
        if dim0 is None:
            axes = list(reversed(range(self.ndim)))
        else:
            axes = dim0
    else:
        axes = list(range(self.ndim))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
    result = _transpose(self, axes)
    from ._autograd import propagate

    return propagate(
        self,
        result,
        lambda value: _transpose(value, axes),
        ("transpose", tuple(axes)),
    )


def _view(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    return _torch_reshape(self, shape)


def _torch_reshape(self, *shape):
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    result = _reshape(self, shape)
    from ._autograd import propagate

    return propagate(
        self,
        result,
        lambda value: _reshape(value, shape),
        ("reshape", tuple(shape)),
    )


def _unsqueeze(self, dim):
    result = mx.expand_dims(self, axis=dim)
    from ._autograd import propagate

    return propagate(
        self,
        result,
        lambda value: mx.expand_dims(value, axis=dim),
        ("unsqueeze", dim),
    )


def _torch_squeeze(self, dim=None):
    result = _squeeze(self, axis=dim)
    from ._autograd import propagate

    return propagate(
        self,
        result,
        lambda value: _squeeze(value, axis=dim),
        ("squeeze", dim),
    )


def _flatten(self, start_dim=0, end_dim=-1):
    if end_dim < 0:
        end_dim += self.ndim
    flattened = 1
    for dimension in self.shape[start_dim : end_dim + 1]:
        flattened *= dimension
    shape = self.shape[:start_dim] + (flattened,) + self.shape[end_dim + 1 :]
    return _torch_reshape(self, shape)


def _float(self):
    return self.astype(mx.float32)


def _bool(self):
    return self.astype(mx.bool_)


def _pow(self, exponent):
    return mx.power(self, exponent)


def _torch_mean(self, dim=None, keepdim=False, dtype=None):
    value = self.astype(dtype) if dtype is not None else self
    return _mean(value, axis=dim, keepdims=keepdim)


def _torch_var(
    self,
    dim=None,
    unbiased=True,
    keepdim=False,
    *,
    correction=None,
):
    ddof = int(unbiased) if correction is None else correction
    return _var(self, axis=dim, keepdims=keepdim, ddof=ddof)


def _size(self, dim=None):
    return self.shape if dim is None else self.shape[dim]


def _chunk(self, chunks, dim=0):
    if chunks <= 0:
        raise ValueError("chunks must be greater than 0")
    length = self.shape[dim]
    if length == 0:
        return tuple(mx.split(self, chunks, axis=dim))
    chunk_size = (length + chunks - 1) // chunks
    indices = list(range(chunk_size, length, chunk_size))
    return tuple(mx.split(self, indices, axis=dim))


def _to(self, *args, dtype=None, device=None, **kwargs):
    if kwargs:
        name = next(iter(kwargs))
        raise TypeError(f"to() got an unexpected keyword argument {name!r}")
    for value in args:
        if isinstance(value, mx.Dtype):
            dtype = value
        elif isinstance(value, mx.array):
            dtype = value.dtype
        else:
            device = value
    if device is not None and str(device) != "mps":
        raise ValueError("the MLX backend only accepts device='mps'")
    return self.astype(dtype) if dtype is not None and dtype != self.dtype else self


def _repeat_interleave(self, repeats, dim=None):
    return mx.repeat(self, repeats, axis=dim)


def _masked_fill(self, mask, value):
    return mx.where(mask, mx.array(value, dtype=self.dtype), self)


def _expand(self, *sizes):
    if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
        sizes = tuple(sizes[0])
    if len(sizes) < self.ndim:
        raise ValueError("expanded size must have at least as many dimensions as the tensor")
    source = (1,) * (len(sizes) - self.ndim) + self.shape
    target = tuple(current if requested == -1 else requested for requested, current in zip(sizes, source))
    return mx.broadcast_to(self.reshape(source), target)


def _torch_any(self, dim=None, keepdim=False):
    return _any(self, axis=dim, keepdims=keepdim)


def _contiguous(self, memory_format=None):
    return self


def _requires_grad(self, requires_grad=True):
    if requires_grad:
        raise RuntimeError(
            "requires_grad_ is not supported by the MLX backend; use torchmlx.Trainer"
        )
    return self


def _backward(self, *args, **kwargs):
    if args or kwargs:
        raise TypeError("MLX backward compatibility does not accept arguments")
    from ._autograd import backward

    backward(self)


def _torch_item(self):
    from ._autograd import note_eager_evaluation, replayed_value

    note_eager_evaluation()
    return _item(replayed_value(self))


def _mask_indices(mask):
    flat = mask.reshape(-1)
    count = int(mx.sum(flat).item())
    order = mx.argsort(flat.astype(mx.int32))
    if count == 0:
        return order[:0].astype(mx.int64)
    return order[-count:].astype(mx.int64)


def _torch_getitem(self, key):
    if isinstance(key, mx.array) and key.dtype == mx.bool_:
        indices = _mask_indices(key)
        if key.shape == self.shape:
            result = _getitem(self.reshape(-1), indices)
            operation = lambda value, current_indices: _getitem(
                _reshape(value, (-1,)), current_indices
            )
            signature = ("getitem_bool", "flat")
        else:
            result = _getitem(self, indices)
            operation = lambda value, current_indices: _getitem(
                value, current_indices
            )
            signature = ("getitem_bool", "first_axis")
        operands = (indices,)
    elif isinstance(key, mx.array):
        result = _getitem(self, key)
        operation = lambda value, current_key: _getitem(value, current_key)
        signature = ("getitem_array",)
        operands = (key,)
    else:
        result = _getitem(self, key)
        operation = lambda value: _getitem(value, key)
        signature = ("getitem", type(key).__qualname__, repr(key))
        operands = ()
    from ._autograd import propagate

    return propagate(self, result, operation, signature, operands)


def _torch_setitem(self, key, value):
    if isinstance(key, mx.array) and key.dtype == mx.bool_:
        indices = _mask_indices(key)
        if key.shape == self.shape:
            flat = self.reshape(-1)
            _setitem(flat, indices, value)
            return
        _setitem(self, indices, value)
        return
    _setitem(self, key, value)


def install(device_type):
    mx.array.transpose = _torch_transpose
    mx.array.reshape = _torch_reshape
    mx.array.view = _view
    mx.array.unsqueeze = _unsqueeze
    mx.array.squeeze = _torch_squeeze
    mx.array.flatten = _flatten
    mx.array.float = _float
    mx.array.bool = _bool
    mx.array.pow = _pow
    mx.array.mean = _torch_mean
    mx.array.var = _torch_var
    mx.array.size = _size
    mx.array.chunk = _chunk
    mx.array.to = _to
    mx.array.repeat_interleave = _repeat_interleave
    mx.array.masked_fill = _masked_fill
    mx.array.expand = _expand
    mx.array.any = _torch_any
    mx.array.contiguous = _contiguous
    mx.array.requires_grad_ = _requires_grad
    mx.array.backward = _backward
    mx.array.item = _torch_item
    mx.array.device = property(lambda self: device_type("mps"))
    mx.array.__getitem__ = _torch_getitem
    mx.array.__setitem__ = _torch_setitem
