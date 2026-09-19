import mlx.core as mx
import mlx.nn as nn


_active_optimizer = None
_forward_depth = 0
_latest_forward = None
_loss_plans = {}
_lineage = {}
_replayed_losses = {}
_suspended = False


class _ArraySlot:
    def __init__(self, index):
        self.index = index


class _BackwardPlan:
    def __init__(self, value_and_grad, state):
        self._value_and_grad = value_and_grad
        self._compiled = mx.compile(
            value_and_grad,
            inputs=state,
            outputs=state,
        )
        self._compile_enabled = True

    def __call__(self, *inputs):
        function = self._compiled if self._compile_enabled else self._value_and_grad
        return function(*inputs)

    def fallback(self, *inputs):
        self._compile_enabled = False
        return self._value_and_grad(*inputs)


def _partition_arrays(value, arrays):
    if isinstance(value, mx.array):
        slot = _ArraySlot(len(arrays))
        arrays.append(value)
        return slot
    if isinstance(value, tuple):
        return tuple(_partition_arrays(item, arrays) for item in value)
    if isinstance(value, list):
        return [_partition_arrays(item, arrays) for item in value]
    if isinstance(value, dict):
        return {
            key: _partition_arrays(item, arrays) for key, item in value.items()
        }
    return value


def _restore_arrays(template, arrays):
    if isinstance(template, _ArraySlot):
        return arrays[template.index]
    if isinstance(template, tuple):
        return tuple(_restore_arrays(item, arrays) for item in template)
    if isinstance(template, list):
        return [_restore_arrays(item, arrays) for item in template]
    if isinstance(template, dict):
        return {
            key: _restore_arrays(item, arrays) for key, item in template.items()
        }
    return template


def _template_key(template):
    if isinstance(template, _ArraySlot):
        return ("array",)
    if isinstance(template, tuple):
        return ("tuple", tuple(_template_key(item) for item in template))
    if isinstance(template, list):
        return ("list", tuple(_template_key(item) for item in template))
    if isinstance(template, dict):
        return (
            "dict",
            tuple(
                (type(key).__qualname__, repr(key), _template_key(item))
                for key, item in template.items()
            ),
        )
    return ("static", type(template).__qualname__, repr(template))


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
        _latest_forward = (model, args, kwargs, ())
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


def propagate(source, result, operation, signature, operands=()):
    entry = _lineage.get(id(source))
    if not _suspended and entry is not None and entry[0] is source:
        model, args, kwargs, transforms = entry[1]
        _lineage[id(result)] = (
            result,
            (
                model,
                args,
                kwargs,
                transforms + ((operation, signature, operands),),
            ),
        )
    return result


def register_loss(loss, loss_input, rebuild, operands, signature):
    if _suspended or _active_optimizer is None:
        return loss
    entry = _lineage.get(id(loss_input))
    if entry is None or entry[0] is not loss_input:
        return loss
    model, args, kwargs, transforms = entry[1]

    context = (
        args,
        kwargs,
        tuple(transform_operands for _, _, transform_operands in transforms),
        operands,
    )
    dynamic_inputs = []
    template = _partition_arrays(context, dynamic_inputs)
    cache_key = (
        getattr(model, "training", None),
        _template_key(template),
        tuple(transform_signature for _, transform_signature, _ in transforms),
        signature,
    )

    def build():
        state = [model.state, mx.random.state]

        def objective(*current_inputs):
            current_args, current_kwargs, current_transform_operands, current_operands = (
                _restore_arrays(template, current_inputs)
            )
            output = model(*current_args, **current_kwargs)
            for (operation, _, _), operation_operands in zip(
                transforms, current_transform_operands
            ):
                output = operation(output, *operation_operands)
            return rebuild(output, *current_operands)

        value_and_grad = nn.value_and_grad(model, objective)
        return _BackwardPlan(value_and_grad, state)

    _loss_plans[id(loss)] = (
        loss,
        model,
        cache_key,
        tuple(dynamic_inputs),
        build,
    )
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
    _, model, cache_key, dynamic_inputs, build = entry
    random_after_forward = _snapshot_random_state()
    _restore_random_state(_active_optimizer._random_before_forward)
    _suspended = True
    try:
        plan = _active_optimizer._compiled_backward.get(cache_key)
        if plan is None:
            plan = build()
            _active_optimizer._compiled_backward[cache_key] = plan
        parameters_before_replay = model.trainable_parameters()
        try:
            replayed_loss, gradients = plan(*dynamic_inputs)
        except ValueError as error:
            message = str(error)
            if (
                not plan._compile_enabled
                or "Attempting to eval an array" not in message
                or not (
                    "function transformations" in message
                    or "without a primitive" in message
                )
            ):
                raise
            model.update(parameters_before_replay)
            _restore_random_state(_active_optimizer._random_before_forward)
            replayed_loss, gradients = plan.fallback(*dynamic_inputs)
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
    optimizer._compiled_step(gradients)
    mx.eval(optimizer._step_state)
    optimizer._pending_update = None
    _active_optimizer = None
    _latest_forward = None
    _loss_plans.clear()
    _lineage.clear()
