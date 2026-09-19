from torchmlx._backend import BACKEND, unsupported


if BACKEND == "torch":
    import torch.optim as _native

    AdamW = _native.AdamW

    def __getattr__(name):
        return getattr(_native, name)

else:
    import mlx.optimizers as _native

    class AdamW:
        def __init__(
            self,
            params,
            lr=1e-3,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
            amsgrad=False,
            maximize=False,
            foreach=None,
            capturable=False,
            differentiable=False,
            fused=None,
        ):
            if amsgrad or maximize or foreach is not None or capturable or differentiable or fused is not None:
                unsupported("torchmlx.optim.AdamW with non-default options")
            self._parameters = params
            self._optimizer = _native.AdamW(
                learning_rate=lr,
                betas=list(betas),
                eps=eps,
                weight_decay=weight_decay,
                bias_correction=True,
            )

        @property
        def state(self):
            return self._optimizer.state

        def zero_grad(self, *args, **kwargs):
            unsupported("torchmlx.optim.AdamW.zero_grad on MLX; use torchmlx.Trainer")

        def step(self, *args, **kwargs):
            unsupported("torchmlx.optim.AdamW.step on MLX; use torchmlx.Trainer")

    def __getattr__(name):
        unsupported(f"torchmlx.optim.{name}")


__all__ = ["AdamW"]
