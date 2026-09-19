from ._backend import BACKEND


if BACKEND == "torch":
    import torch

    class Trainer:
        def __init__(self, model, optimizer, loss_fn, compile=True):
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self._step = torch.compile(self._train_step) if compile else self._train_step

        def _train_step(self, x, y):
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(x), y)
            loss.backward()
            self.optimizer.step()
            return loss

        def step(self, x, y):
            return self._step(x, y)

else:
    import mlx.core as mx
    import mlx.nn as nn

    class Trainer:
        def __init__(self, model, optimizer, loss_fn, compile=True):
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            native_optimizer = optimizer._optimizer
            native_optimizer.init(model.trainable_parameters())

            def objective(x, y):
                return loss_fn(model(x), y)

            value_and_grad = nn.value_and_grad(model, objective)

            def train_step(x, y):
                loss, gradients = value_and_grad(x, y)
                native_optimizer.update(model, gradients)
                return loss

            if compile:
                state = [model.state, native_optimizer.state, mx.random.state]
                self._step = mx.compile(train_step, inputs=state, outputs=state)
            else:
                self._step = train_step

        def step(self, x, y):
            loss = self._step(x, y)
            mx.eval(
                loss,
                self.model.parameters(),
                self.optimizer._optimizer.state,
                mx.random.state,
            )
            return loss
