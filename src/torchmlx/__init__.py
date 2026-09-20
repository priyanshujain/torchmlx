from ._backend import BACKEND


if BACKEND == "torch":
    from ._torch_backend import *
    from ._torch_backend import fallback as _fallback
else:
    from ._mlx_backend import *
    from ._mlx_backend import fallback as _fallback


def __getattr__(name):
    return _fallback(name)


import importlib as _importlib

nn = _importlib.import_module("torchmlx.nn")
optim = _importlib.import_module("torchmlx.optim")

__all__ = [
    "Tensor",
    "arange",
    "backends",
    "bfloat16",
    "bool",
    "categorical",
    "cat",
    "chunk",
    "cuda",
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
