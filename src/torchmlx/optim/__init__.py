from torchmlx._backend import BACKEND, unsupported


if BACKEND == "torch":
    import torch.optim as _native

    AdamW = _native.AdamW

    def __getattr__(name):
        return getattr(_native, name)

else:
    import mlx.core as mx
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
            model = getattr(params, "model", None)
            if model is None:
                raise TypeError(
                    "MLX AdamW requires parameters returned directly by model.parameters()"
                )
            self._parameters = params
            self._model = model
            self._optimizer = _native.AdamW(
                learning_rate=lr,
                betas=list(betas),
                eps=eps,
                weight_decay=weight_decay,
                bias_correction=True,
            )
            self._pending_update = None
            self._random_before_forward = None
            self._compiled_backward = {}
            self._optimizer.init(model.trainable_parameters())
            state = [model.state, self._optimizer.state, mx.random.state]

            def update(gradients):
                self._optimizer.update(model, gradients)

            self._compiled_step = mx.compile(
                update,
                inputs=state,
                outputs=state,
            )
            self._step_state = state

        @property
        def state(self):
            return self._optimizer.state

        def zero_grad(self, *args, **kwargs):
            if args or kwargs:
                unsupported("torchmlx.optim.AdamW.zero_grad with arguments")
            from torchmlx._autograd import activate

            self._pending_update = None
            activate(self)

        def step(self, *args, **kwargs):
            if args or kwargs:
                unsupported("torchmlx.optim.AdamW.step with arguments")
            from torchmlx._autograd import step

            step(self)

    def __getattr__(name):
        unsupported(f"torchmlx.optim.{name}")


__all__ = ["AdamW"]
