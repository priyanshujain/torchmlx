import builtins as _builtins
import math as _math
import platform as _platform
from types import SimpleNamespace as _SimpleNamespace

from ._backend import BACKEND, unsupported


def current_backend():
    return BACKEND


if BACKEND == "torch":
    import torch as _native

    Tensor = _native.Tensor
    tensor = _native.tensor
    from_numpy = _native.from_numpy
    arange = _native.arange
    randint = _native.randint
    chunk = _native.chunk
    transpose = _native.transpose
    float16 = _native.float16
    float32 = _native.float32
    float64 = _native.float64
    bfloat16 = _native.bfloat16
    int8 = _native.int8
    int16 = _native.int16
    int32 = _native.int32
    int64 = _native.int64
    uint8 = _native.uint8
    bool = _native.bool
    half = float16
    float = float32
    double = float64
    int = int32
    long = int64

    def categorical(logits, dim=-1, num_samples=1):
        probabilities = _native.softmax(logits, dim=dim)
        return _native.multinomial(probabilities, num_samples=num_samples)

    def __getattr__(name):
        return getattr(_native, name)

else:
    import mlx.core as _native
    import numpy as _np

    Tensor = _native.array
    float16 = _native.float16
    float32 = _native.float32
    float64 = _native.float64
    bfloat16 = _native.bfloat16
    int8 = _native.int8
    int16 = _native.int16
    int32 = _native.int32
    int64 = _native.int64
    uint8 = _native.uint8
    bool = _native.bool_
    half = float16
    float = float32
    double = float64
    int = int32
    long = int64

    pi = _math.pi

    class device:
        def __init__(self, value):
            if isinstance(value, device):
                value = value.type
            value = str(value)
            if value != "mps":
                raise ValueError("the MLX backend only accepts device='mps'")
            self.type = value
            self.index = None

        def __str__(self):
            return self.type

        def __repr__(self):
            return f"device(type={self.type!r})"

        def __eq__(self, other):
            return str(other) == self.type

    class _MPS:
        @staticmethod
        def is_available():
            return _platform.system() == "Darwin" and _platform.machine() == "arm64"

    class _CUDA:
        @staticmethod
        def is_available():
            return False

    backends = _SimpleNamespace(mps=_MPS())
    cuda = _CUDA()

    from ._mlx_tensor import install as _install_tensor_methods

    _install_tensor_methods(device)

    def _check_device(device):
        if device is not None and str(device) != "mps":
            raise ValueError("the MLX backend only accepts device='mps'")

    def tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
        _check_device(device)
        if requires_grad:
            raise RuntimeError(
                "requires_grad is not supported by the MLX backend; use torchmlx.Trainer"
            )
        if pin_memory:
            raise RuntimeError("pin_memory is not supported by the MLX backend")
        if dtype is None and not isinstance(data, (_native.array, _np.ndarray)):
            kind = _np.asarray(data).dtype.kind
            if kind in {"i", "u"}:
                dtype = int64
            elif kind == "b":
                dtype = bool
        return _native.array(data, dtype=dtype)

    def from_numpy(array):
        if not isinstance(array, _np.ndarray):
            raise TypeError("from_numpy expects a numpy.ndarray")
        return _native.array(array)

    def arange(start, end=None, step=1, *, dtype=None, device=None):
        _check_device(device)
        if end is None:
            start, end = 0, start
        if dtype is None and all(
            isinstance(value, _builtins.int) for value in (start, end, step)
        ):
            dtype = int64
        return _native.arange(start, end, step, dtype=dtype)

    def randint(low, high, size, *, dtype=None, device=None, requires_grad=False):
        _check_device(device)
        if requires_grad:
            raise RuntimeError(
                "requires_grad is not supported by the MLX backend; use torchmlx.Trainer"
            )
        result = _native.random.randint(low, high, shape=size)
        return result.astype(dtype or int64)

    def chunk(input, chunks, dim=0):
        return input.chunk(chunks, dim=dim)

    def transpose(input, dim0, dim1):
        return input.transpose(dim0, dim1)

    def zeros(*size, dtype=None, device=None, requires_grad=False):
        _check_device(device)
        if requires_grad:
            raise RuntimeError(
                "requires_grad is not supported by the MLX backend; use torchmlx.Trainer"
            )
        shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
        return _native.zeros(shape, dtype=dtype or float32)

    def ones(*size, dtype=None, device=None, requires_grad=False):
        _check_device(device)
        if requires_grad:
            raise RuntimeError(
                "requires_grad is not supported by the MLX backend; use torchmlx.Trainer"
            )
        shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
        return _native.ones(shape, dtype=dtype or float32)

    def zeros_like(input, *, dtype=None, device=None, requires_grad=False):
        _check_device(device)
        if requires_grad:
            raise RuntimeError(
                "requires_grad is not supported by the MLX backend; use torchmlx.Trainer"
            )
        return _native.zeros_like(input).astype(dtype or input.dtype)

    def ones_like(input, *, dtype=None, device=None, requires_grad=False):
        _check_device(device)
        if requires_grad:
            raise RuntimeError(
                "requires_grad is not supported by the MLX backend; use torchmlx.Trainer"
            )
        return _native.ones_like(input).astype(dtype or input.dtype)

    def cat(tensors, dim=0):
        return _native.concatenate(tensors, axis=dim)

    def softmax(input, dim, dtype=None):
        value = input.astype(dtype) if dtype is not None else input
        return _native.softmax(value, axis=dim)

    def triu(input, diagonal=0):
        return _native.triu(input, k=diagonal)

    def tril(input, diagonal=0):
        return _native.tril(input, k=diagonal)

    def topk(input, k, dim=None, largest=True, sorted=True):
        axis = -1 if dim is None else dim
        indices = _native.argsort(input, axis=axis)
        indices = _native.flip(indices, axis=axis) if largest else indices
        slices = [slice(None)] * input.ndim
        slices[axis] = slice(0, k)
        indices = indices[tuple(slices)].astype(int64)
        values = _native.take_along_axis(input, indices, axis=axis)
        return values, indices

    def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
        if return_inverse or return_counts or dim is not None:
            unsupported("torchmlx.unique with non-default options")
        values = _native.sort(input.reshape(-1))
        if values.shape[0] < 2:
            return values
        keep = _native.concatenate(
            [_native.array([True]), values[1:] != values[:-1]]
        )
        return values[keep]

    exp = _native.exp
    sin = _native.sin
    cos = _native.cos
    tanh = _native.tanh
    sqrt = _native.sqrt
    matmul = _native.matmul
    outer = _native.outer

    def pow(input, exponent):
        return _native.power(input, exponent)

    def polar(abs, angle):
        return abs * _native.exp(_native.array(1j) * angle)

    def categorical(logits, dim=-1, num_samples=1):
        if num_samples == 1:
            return _native.random.categorical(logits, axis=dim)[..., None].astype(int64)
        return _native.random.categorical(
            logits, axis=dim, num_samples=num_samples
        ).astype(int64)

    def __getattr__(name):
        unsupported(f"torchmlx.{name}")


import importlib as _importlib

nn = _importlib.import_module("torchmlx.nn")
optim = _importlib.import_module("torchmlx.optim")
from .trainer import Trainer

__all__ = [
    "Tensor",
    "Trainer",
    "arange",
    "backends",
    "bfloat16",
    "bool",
    "categorical",
    "cat",
    "chunk",
    "cuda",
    "current_backend",
    "device",
    "double",
    "float",
    "float16",
    "float32",
    "float64",
    "exp",
    "from_numpy",
    "int8",
    "int16",
    "int32",
    "int64",
    "half",
    "int",
    "long",
    "matmul",
    "nn",
    "ones",
    "ones_like",
    "optim",
    "outer",
    "pi",
    "polar",
    "pow",
    "randint",
    "tensor",
    "softmax",
    "sin",
    "cos",
    "sqrt",
    "tanh",
    "topk",
    "transpose",
    "tril",
    "triu",
    "uint8",
    "unique",
    "zeros",
    "zeros_like",
]
