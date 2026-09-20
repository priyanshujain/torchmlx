from __future__ import annotations

from typing import Any, Callable, cast

import mlx.core as mx


_native_getitem = mx.array.__getitem__
_native_setitem = mx.array.__setitem__
_native_item = mx.array.item
_native_reshape = mx.array.reshape
_native_squeeze = mx.array.squeeze
_native_transpose = mx.array.transpose


def wrap(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, mx.array):
        return Tensor(value)
    if isinstance(value, tuple):
        return tuple(wrap(item) for item in value)
    if isinstance(value, list):
        return [wrap(item) for item in value]
    if isinstance(value, dict):
        return {key: wrap(item) for key, item in value.items()}
    return value


class Tensor(mx.array):
    _lineage: Any = None
    _loss_plan: Any = None
    _replayed_value: mx.array | None = None

    def _result(
        self,
        value: Any,
        operation: Callable[..., mx.array] | None = None,
        signature: tuple[Any, ...] = (),
        operands: tuple[mx.array, ...] = (),
    ) -> Any:
        result = wrap(value)
        if operation is not None and isinstance(result, Tensor):
            from ._autograd import propagate

            propagate(self, result, operation, signature, operands)
        return result

    @property
    def device(self) -> Any:
        from ._mlx_backend import device

        return device("mps")

    @property
    def grad(self) -> Any:
        raise RuntimeError("parameter gradients are not exposed by the MLX backend")

    def register_hook(self, hook: Any) -> Any:
        raise RuntimeError("gradient hooks are not supported by the MLX backend")

    def transpose(self, dim0: Any = None, dim1: int | None = None) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        if dim1 is None:
            axes = list(reversed(range(self.ndim))) if dim0 is None else dim0
        else:
            axes = list(range(self.ndim))
            axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return self._result(
            _native_transpose(self, axes),
            lambda value: _native_transpose(value, axes),
            ("transpose", tuple(axes)),
        )

    def view(self, *shape: Any) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.reshape(*shape)

    def reshape(self, *shape: Any) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._result(
            _native_reshape(self, shape),
            lambda value: _native_reshape(value, shape),
            ("reshape", tuple(shape)),
        )

    def unsqueeze(self, dim: int) -> Tensor:
        return self._result(
            mx.expand_dims(self, axis=dim),
            lambda value: mx.expand_dims(value, axis=dim),
            ("unsqueeze", dim),
        )

    def squeeze(self, dim: int | None = None) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._result(
            _native_squeeze(self, axis=dim),
            lambda value: _native_squeeze(value, axis=dim),
            ("squeeze", dim),
        )

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        if end_dim < 0:
            end_dim += self.ndim
        flattened = 1
        for dimension in self.shape[start_dim : end_dim + 1]:
            flattened *= dimension
        shape = self.shape[:start_dim] + (flattened,) + self.shape[end_dim + 1 :]
        return self.reshape(shape)

    def float(self) -> Tensor:
        return self.astype(mx.float32)

    def bool(self) -> Tensor:
        return self.astype(mx.bool_)

    def pow(self, exponent: Any) -> Tensor:
        return self._binary(mx.power, exponent, "pow")

    def mean(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, dim: Any = None, keepdim: bool = False, dtype: Any = None
    ) -> Tensor:
        value = mx.astype(self, dtype) if dtype is not None else self
        return wrap(mx.mean(value, axis=dim, keepdims=keepdim))

    def var(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        dim: Any = None,
        unbiased: bool = True,
        keepdim: bool = False,
        *,
        correction: int | None = None,
    ) -> Tensor:
        ddof = int(unbiased) if correction is None else correction
        return wrap(mx.var(self, axis=dim, keepdims=keepdim, ddof=ddof))

    def size(self, dim: int | None = None) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.shape if dim is None else self.shape[dim]

    def chunk(self, chunks: int, dim: int = 0) -> tuple[Tensor, ...]:
        if chunks <= 0:
            raise ValueError("chunks must be greater than 0")
        length = self.shape[dim]
        if length == 0:
            return wrap(tuple(mx.split(self, chunks, axis=dim)))
        chunk_size = (length + chunks - 1) // chunks
        indices = list(range(chunk_size, length, chunk_size))
        return wrap(tuple(mx.split(self, indices, axis=dim)))

    def to(
        self,
        *args: Any,
        dtype: Any = None,
        device: Any = None,
        **kwargs: Any,
    ) -> Tensor:
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

    def repeat_interleave(self, repeats: int, dim: int | None = None) -> Tensor:
        return wrap(mx.repeat(self, repeats, axis=dim))

    def masked_fill(self, mask: mx.array, value: Any) -> Tensor:
        return wrap(mx.where(mask, mx.array(value, dtype=self.dtype), self))

    def expand(self, *sizes: Any) -> Tensor:
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        if len(sizes) < self.ndim:
            raise ValueError("expanded size must have at least as many dimensions as the tensor")
        source = (1,) * (len(sizes) - self.ndim) + self.shape
        target = tuple(
            current if requested == -1 else requested
            for requested, current in zip(sizes, source)
        )
        return wrap(mx.broadcast_to(_native_reshape(self, source), target))

    def any(self, dim: Any = None, keepdim: bool = False) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return wrap(mx.any(self, axis=dim, keepdims=keepdim))

    def contiguous(self, memory_format: Any = None) -> Tensor:
        return self

    def requires_grad_(self, requires_grad: bool = True) -> Tensor:
        if requires_grad:
            raise RuntimeError("requires_grad_ is not supported by the MLX backend")
        return self

    def backward(self, *args: Any, **kwargs: Any) -> None:
        if args or kwargs:
            if kwargs.get("create_graph") or kwargs.get("retain_graph"):
                raise RuntimeError("higher-order gradients are not supported by the MLX backend")
            raise TypeError("MLX backward compatibility does not accept arguments")
        from ._autograd import backward

        backward(self)

    def item(self) -> Any:
        from ._autograd import note_eager_evaluation, replayed_value

        note_eager_evaluation()
        return _native_item(replayed_value(self))

    def astype(self, dtype: Any, *, stream: Any = None) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        kwargs = {} if stream is None else {"stream": stream}
        return wrap(mx.astype(self, dtype, **kwargs))

    def __getitem__(self, key: Any) -> Tensor:
        if isinstance(key, mx.array) and key.dtype == mx.bool_:
            indices = _mask_indices(key)
            if key.shape == self.shape:
                result = _native_getitem(_native_reshape(self, (-1,)), indices)
                operation = lambda value, current: _native_getitem(
                    _native_reshape(value, (-1,)), current
                )
                signature = ("getitem_bool", "flat")
            else:
                result = _native_getitem(self, indices)
                operation = lambda value, current: _native_getitem(value, current)
                signature = ("getitem_bool", "first_axis")
            return self._result(result, operation, signature, (indices,))
        if isinstance(key, mx.array):
            return self._result(
                _native_getitem(self, key),
                lambda value, current: _native_getitem(value, current),
                ("getitem_array",),
                (key,),
            )
        return self._result(
            _native_getitem(self, key),
            lambda value: _native_getitem(value, key),
            ("getitem", type(key).__qualname__, repr(key)),
        )

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, mx.array) and key.dtype == mx.bool_:
            indices = _mask_indices(key)
            if key.shape == self.shape:
                _native_setitem(_native_reshape(self, (-1,)), indices, value)
                return
            _native_setitem(self, indices, value)
            return
        _native_setitem(self, key, value)

    def _binary(self, function: Callable[..., mx.array], other: Any, name: str) -> Tensor:
        if isinstance(other, Tensor) and other._lineage is not None:
            return wrap(function(self, other))
        operands = (other,) if isinstance(other, mx.array) else ()
        operation = (
            (lambda value, current: function(value, current))
            if operands
            else (lambda value: function(value, other))
        )
        signature = (name, "array" if operands else repr(other))
        return self._result(function(self, other), operation, signature, operands)

    def _reverse(self, function: Callable[..., mx.array], other: Any) -> Tensor:
        return wrap(function(other, self))

    def __add__(self, other: Any) -> Tensor:
        return self._binary(mx.add, other, "add")

    def __radd__(self, other: Any) -> Tensor:
        return self._reverse(mx.add, other)

    def __sub__(self, other: Any) -> Tensor:
        return self._binary(mx.subtract, other, "sub")

    def __rsub__(self, other: Any) -> Tensor:
        return self._reverse(mx.subtract, other)

    def __mul__(self, other: Any) -> Tensor:
        return self._binary(mx.multiply, other, "mul")

    def __rmul__(self, other: Any) -> Tensor:
        return self._reverse(mx.multiply, other)

    def __truediv__(self, other: Any) -> Tensor:
        return self._binary(mx.divide, other, "div")

    def __rtruediv__(self, other: Any) -> Tensor:
        return self._reverse(mx.divide, other)

    def __pow__(self, other: Any) -> Tensor:
        return self.pow(other)

    def __rpow__(self, other: Any) -> Tensor:
        return self._reverse(mx.power, other)

    def __matmul__(self, other: Any) -> Tensor:
        return self._binary(mx.matmul, other, "matmul")

    def __rmatmul__(self, other: Any) -> Tensor:
        return self._reverse(mx.matmul, other)

    def __neg__(self) -> Tensor:
        return wrap(mx.negative(self))

    def __eq__(self, other: Any) -> Tensor:
        return wrap(mx.equal(self, other))

    def __ne__(self, other: Any) -> Tensor:
        return wrap(mx.not_equal(self, other))

    def __lt__(self, other: Any) -> Tensor:
        return wrap(mx.less(self, other))

    def __le__(self, other: Any) -> Tensor:
        return wrap(mx.less_equal(self, other))

    def __gt__(self, other: Any) -> Tensor:
        return wrap(mx.greater(self, other))

    def __ge__(self, other: Any) -> Tensor:
        return wrap(mx.greater_equal(self, other))


def _mask_indices(mask: mx.array) -> mx.array:
    flat = _native_reshape(mask, (-1,))
    count = int(cast(Any, _native_item(mx.sum(flat))))
    order = mx.argsort(mx.astype(flat, mx.int32))
    if count == 0:
        return mx.astype(_native_getitem(order, slice(0, 0)), mx.int64)
    return mx.astype(_native_getitem(order, slice(-count, None)), mx.int64)
