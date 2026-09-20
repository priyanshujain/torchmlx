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
    class Trainer:
        def __init__(self, model, optimizer, loss_fn, compile=True):
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn
            self.compile = compile

        def step(self, x, y):
            from ._autograd import trainer_step

            return trainer_step(self, x, y)
