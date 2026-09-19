import mlx.core as mx
import mlx.nn as nn


_active_optimizer = None
_forward_depth = 0
_latest_forward = None
_loss_plans = {}
_lineage = {}
_replayed_losses = {}
_suspended = False


def _snapshot_random_state():
    state = [key + mx.array(0, dtype=key.dtype) for key in mx.random.state]
    mx.eval(state)
    return state


def _restore_random_state(state):
    for current_key, saved_key in zip(mx.random.state, state):
        current_key[...] = saved_key
    mx.eval(mx.random.state)


def begin_forward():
    global _forward_depth
    _forward_depth += 1


def end_forward(model, args, kwargs, output):
    global _forward_depth, _latest_forward
    _forward_depth -= 1
    if (
        _forward_depth == 0
        and _active_optimizer is not None
        and model is _active_optimizer._model
        and not _suspended
        and isinstance(output, mx.array)
    ):
        _latest_forward = (model, args, kwargs, lambda value: value)
        _lineage[id(output)] = (output, _latest_forward)


def abort_forward():
    global _forward_depth
    _forward_depth -= 1


def activate(optimizer):
    global _active_optimizer, _latest_forward
    _active_optimizer = optimizer
    _latest_forward = None
    _loss_plans.clear()
    _lineage.clear()
    _replayed_losses.clear()
    optimizer._random_before_forward = _snapshot_random_state()


def propagate(source, result, operation=lambda value: value):
    entry = _lineage.get(id(source))
    if not _suspended and entry is not None and entry[0] is source:
        model, args, kwargs, previous = entry[1]
        _lineage[id(result)] = (
            result,
            (
                model,
                args,
                kwargs,
                lambda output: operation(previous(output)),
            ),
        )
    return result


def register_loss(loss, loss_input, rebuild):
    if _suspended or _active_optimizer is None:
        return loss
    entry = _lineage.get(id(loss_input))
    if entry is None or entry[0] is not loss_input:
        return loss
    model, args, kwargs, transform = entry[1]

    def plan():
        def objective():
            return rebuild(transform(model(*args, **kwargs)))

        return nn.value_and_grad(model, objective)()

    _loss_plans[id(loss)] = (loss, model, plan)
    return loss


def backward(loss):
    global _suspended
    if _active_optimizer is None:
        raise RuntimeError("optimizer.zero_grad() must be called before loss.backward()")
    entry = _loss_plans.get(id(loss))
    if entry is None or entry[0] is not loss:
        raise RuntimeError(
            "this MLX loss cannot use backward compatibility; compute it with a supported torchmlx loss function"
        )
    _, model, plan = entry
    random_after_forward = _snapshot_random_state()
    _restore_random_state(_active_optimizer._random_before_forward)
    _suspended = True
    try:
        replayed_loss, gradients = plan()
        mx.eval(replayed_loss, gradients, mx.random.state)
    finally:
        _suspended = False
        _restore_random_state(random_after_forward)
    _replayed_losses[id(loss)] = (loss, replayed_loss)
    _active_optimizer._pending_update = (model, gradients)


def replayed_value(value):
    entry = _replayed_losses.get(id(value))
    if entry is None or entry[0] is not value:
        return value
    return entry[1]


def step(optimizer):
    global _active_optimizer, _latest_forward
    if optimizer._pending_update is None:
        raise RuntimeError("loss.backward() must be called before optimizer.step()")
    model, gradients = optimizer._pending_update
    optimizer._optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer._optimizer.state, mx.random.state)
    optimizer._pending_update = None
    _active_optimizer = None
    _latest_forward = None
    _loss_plans.clear()
    _lineage.clear()
