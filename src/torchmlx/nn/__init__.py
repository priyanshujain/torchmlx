from torchmlx._backend import BACKEND, unsupported


if BACKEND == "torch":
    import torch.nn as _native

    Module = _native.Module
    ModuleList = _native.ModuleList
    Linear = _native.Linear
    Embedding = _native.Embedding
    LayerNorm = _native.LayerNorm
    Sequential = _native.Sequential
    GELU = _native.GELU
    Dropout = _native.Dropout
    Parameter = _native.Parameter

    def __getattr__(name):
        return getattr(_native, name)

else:
    from collections.abc import Iterable, MutableSequence

    import mlx.core as mx
    import mlx.nn as _native

    class _ParameterTree(dict):
        def __init__(self, values, model):
            super().__init__(values)
            self.model = model

    class Module(_native.Module):
        def __call__(self, *args, **kwargs):
            from torchmlx._autograd import abort_forward, begin_forward, end_forward

            tracking = begin_forward(self)
            try:
                output = self.forward(*args, **kwargs)
            except Exception:
                if tracking:
                    abort_forward()
                raise
            if tracking:
                end_forward(self, args, kwargs, output)
            return output

        def forward(self, *args, **kwargs):
            raise NotImplementedError(
                f"Module [{type(self).__name__}] is missing the required forward function"
            )

        def parameters(self):
            return _ParameterTree(super().parameters(), self)

        def to(self, *args, **kwargs):
            dtype = kwargs.pop("dtype", None)
            device = kwargs.pop("device", None)
            if kwargs:
                name = next(iter(kwargs))
                raise TypeError(f"to() got an unexpected keyword argument {name!r}")
            for value in args:
                if isinstance(value, mx.Dtype):
                    if dtype is not None:
                        raise TypeError("to() received dtype more than once")
                    dtype = value
                else:
                    if device is not None:
                        raise TypeError("to() received device more than once")
                    device = value
            if device is not None and str(device) != "mps":
                raise ValueError("the MLX backend only accepts device='mps'")
            if dtype is not None:
                self.set_dtype(dtype)
            return self

        def register_buffer(self, name, tensor, persistent=True):
            if not isinstance(name, str) or "." in name or name == "":
                raise KeyError("buffer name must be a non-empty string without dots")
            setattr(self, name, tensor)
            self.freeze(recurse=False, keys=name, strict=True)

    def Parameter(data=None, requires_grad=True):
        if data is None:
            return mx.array([])
        if not requires_grad:
            unsupported("torchmlx.nn.Parameter with requires_grad=False")
        return data

    class Linear(Module, _native.Linear):
        def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
            Module.__init__(self)
            if device is not None and str(device) != "mps":
                raise ValueError("the MLX backend only accepts device='mps'")
            scale = (1 / in_features) ** 0.5
            self.weight = mx.random.uniform(
                low=-scale, high=scale, shape=(out_features, in_features)
            )
            if bias:
                self.bias = mx.random.uniform(
                    low=-scale, high=scale, shape=(out_features,)
                )
            if dtype is not None:
                self.set_dtype(dtype)

        def forward(self, input):
            return _native.Linear.__call__(self, input)

    class Embedding(Module, _native.Embedding):
        def __init__(
            self,
            num_embeddings,
            embedding_dim,
            padding_idx=None,
            max_norm=None,
            norm_type=2.0,
            scale_grad_by_freq=False,
            sparse=False,
            device=None,
            dtype=None,
        ):
            if any(
                value is not None
                for value in (padding_idx, max_norm)
            ) or norm_type != 2.0 or scale_grad_by_freq or sparse:
                unsupported("torchmlx.nn.Embedding with non-default options")
            Module.__init__(self)
            if device is not None and str(device) != "mps":
                raise ValueError("the MLX backend only accepts device='mps'")
            scale = (1 / embedding_dim) ** 0.5
            self.weight = mx.random.normal(
                shape=(num_embeddings, embedding_dim), scale=scale
            )
            if dtype is not None:
                self.set_dtype(dtype)

        def forward(self, input):
            return self.weight[input]

    class LayerNorm(Module, _native.LayerNorm):
        def __init__(
            self,
            normalized_shape,
            eps=1e-5,
            elementwise_affine=True,
            bias=True,
            device=None,
            dtype=None,
        ):
            if isinstance(normalized_shape, Iterable) and not isinstance(
                normalized_shape, (str, bytes)
            ):
                shape = tuple(normalized_shape)
                if len(shape) != 1:
                    unsupported("torchmlx.nn.LayerNorm with multidimensional shape")
                normalized_shape = shape[0]
            if device is not None and str(device) != "mps":
                raise ValueError("the MLX backend only accepts device='mps'")
            _native.LayerNorm.__init__(
                self,
                normalized_shape,
                eps=eps,
                affine=elementwise_affine,
                bias=bias,
            )
            if dtype is not None:
                self.set_dtype(dtype)

        def forward(self, input):
            return _native.LayerNorm.__call__(self, input)

    class Sequential(Module):
        def __init__(self, *args):
            super().__init__()
            self.layers = list(args)

        def forward(self, input):
            for layer in self.layers:
                input = layer(input)
            return input

        def __len__(self):
            return len(self.layers)

        def __getitem__(self, index):
            if isinstance(index, str):
                return dict.__getitem__(self, index)
            return self.layers[index]

    class GELU(Module, _native.GELU):
        def __init__(self, approximate="none"):
            if approximate not in {"none", "tanh"}:
                raise ValueError("approximate must be 'none' or 'tanh'")
            _native.GELU.__init__(self, approx=approximate)

        def forward(self, input):
            return _native.GELU.__call__(self, input)

    class Dropout(Module, _native.Dropout):
        def __init__(self, p=0.5, inplace=False):
            if inplace:
                unsupported("torchmlx.nn.Dropout with inplace=True")
            _native.Dropout.__init__(self, p=p)

        def forward(self, input):
            return _native.Dropout.__call__(self, input)

    class ModuleList(Module, MutableSequence):
        def __init__(self, modules=None):
            super().__init__()
            self.layers = list(modules or [])

        def __getitem__(self, index):
            if isinstance(index, str):
                return dict.__getitem__(self, index)
            return self.layers[index]

        def __setitem__(self, index, module):
            if isinstance(index, str):
                dict.__setitem__(self, index, module)
                return
            self.layers[index] = module

        def __delitem__(self, index):
            if isinstance(index, str):
                dict.__delitem__(self, index)
                return
            del self.layers[index]

        def __len__(self):
            return len(self.layers)

        def __iter__(self):
            return iter(self.layers)

        def insert(self, index, module):
            self.layers.insert(index, module)

        def forward(self, *args, **kwargs):
            unsupported("torchmlx.nn.ModuleList.forward")

    def __getattr__(name):
        unsupported(f"torchmlx.nn.{name}")


import importlib as _importlib

functional = _importlib.import_module("torchmlx.nn.functional")

__all__ = [
    "Embedding",
    "Dropout",
    "GELU",
    "LayerNorm",
    "Linear",
    "Module",
    "ModuleList",
    "Parameter",
    "Sequential",
    "functional",
]
