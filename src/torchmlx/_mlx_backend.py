import builtins
import math
import platform
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from ._backend import unsupported
from ._mlx_tensor import Tensor, wrap


float16 = mx.float16
float32 = mx.float32
float64 = mx.float64
bfloat16 = mx.bfloat16
int8 = mx.int8
int16 = mx.int16
int32 = mx.int32
int64 = mx.int64
uint8 = mx.uint8
bool = mx.bool_
half = float16
float = float32
double = float64
int = int32
long = int64
pi = math.pi


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
        return platform.system() == "Darwin" and platform.machine() == "arm64"


class _CUDA:
    @staticmethod
    def is_available():
        return False


backends = SimpleNamespace(mps=_MPS())
cuda = _CUDA()


def _check_device(value):
    if value is not None and str(value) != "mps":
        raise ValueError("the MLX backend only accepts device='mps'")


def tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False):
    _check_device(device)
    if requires_grad:
        raise RuntimeError("requires_grad is not supported by the MLX backend")
    if pin_memory:
        raise RuntimeError("pin_memory is not supported by the MLX backend")
    if dtype is None and not isinstance(data, (mx.array, np.ndarray)):
        kind = np.asarray(data).dtype.kind
        if kind in {"i", "u"}:
            dtype = int64
        elif kind == "b":
            dtype = bool
    return wrap(mx.array(data, dtype=dtype))


def from_numpy(array):
    if not isinstance(array, np.ndarray):
        raise TypeError("from_numpy expects a numpy.ndarray")
    return wrap(mx.array(array))


def arange(start, end=None, step=1, *, dtype=None, device=None):
    _check_device(device)
    if end is None:
        start, end = 0, start
    if dtype is None and all(isinstance(value, builtins.int) for value in (start, end, step)):
        dtype = int64
    return wrap(mx.arange(start, end, step, dtype=dtype))


def randint(low, high, size, *, dtype=None, device=None, requires_grad=False):
    _check_device(device)
    if requires_grad:
        raise RuntimeError("requires_grad is not supported by the MLX backend")
    return wrap(mx.random.randint(low, high, shape=size).astype(dtype or int64))


def chunk(input, chunks, dim=0):
    return input.chunk(chunks, dim=dim)


def transpose(input, dim0, dim1):
    return input.transpose(dim0, dim1)


def zeros(*size, dtype=None, device=None, requires_grad=False):
    _check_device(device)
    if requires_grad:
        raise RuntimeError("requires_grad is not supported by the MLX backend")
    shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
    return wrap(mx.zeros(shape, dtype=dtype or float32))


def ones(*size, dtype=None, device=None, requires_grad=False):
    _check_device(device)
    if requires_grad:
        raise RuntimeError("requires_grad is not supported by the MLX backend")
    shape = size[0] if len(size) == 1 and isinstance(size[0], (tuple, list)) else size
    return wrap(mx.ones(shape, dtype=dtype or float32))


def zeros_like(input, *, dtype=None, device=None, requires_grad=False):
    _check_device(device)
    if requires_grad:
        raise RuntimeError("requires_grad is not supported by the MLX backend")
    return wrap(mx.zeros_like(input).astype(dtype or input.dtype))


def ones_like(input, *, dtype=None, device=None, requires_grad=False):
    _check_device(device)
    if requires_grad:
        raise RuntimeError("requires_grad is not supported by the MLX backend")
    return wrap(mx.ones_like(input).astype(dtype or input.dtype))


def cat(tensors, dim=0):
    return wrap(mx.concatenate(tensors, axis=dim))


def softmax(input, dim, dtype=None):
    value = input.astype(dtype) if dtype is not None else input
    return wrap(mx.softmax(value, axis=dim))


def triu(input, diagonal=0):
    return wrap(mx.triu(input, k=diagonal))


def tril(input, diagonal=0):
    return wrap(mx.tril(input, k=diagonal))


def topk(input, k, dim=None, largest=True, sorted=True):
    axis = -1 if dim is None else dim
    indices = mx.argsort(input, axis=axis)
    indices = mx.flip(indices, axis=axis) if largest else indices
    slices = [slice(None)] * input.ndim
    slices[axis] = slice(0, k)
    indices = indices[tuple(slices)].astype(int64)
    values = mx.take_along_axis(input, indices, axis=axis)
    return wrap((values, indices))


def unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None):
    if return_inverse or return_counts or dim is not None:
        unsupported("torchmlx.unique with non-default options")
    values = mx.sort(input.reshape(-1))
    if values.shape[0] < 2:
        return wrap(values)
    keep = mx.concatenate([mx.array([True]), values[1:] != values[:-1]])
    return wrap(values[keep])


def _unary(function):
    return lambda input: wrap(function(input))


exp = _unary(mx.exp)
sin = _unary(mx.sin)
cos = _unary(mx.cos)
tanh = _unary(mx.tanh)
sqrt = _unary(mx.sqrt)


def matmul(input, other):
    return wrap(mx.matmul(input, other))


def outer(input, other):
    return wrap(mx.outer(input, other))


def pow(input, exponent):
    return wrap(mx.power(input, exponent))


def polar(abs, angle):
    return wrap(abs * mx.exp(mx.array(1j) * angle))


def categorical(logits, dim=-1, num_samples=1):
    if num_samples == 1:
        return wrap(mx.random.categorical(logits, axis=dim)[..., None].astype(int64))
    return wrap(mx.random.categorical(logits, axis=dim, num_samples=num_samples).astype(int64))


def fallback(name):
    unsupported(f"torchmlx.{name}")
