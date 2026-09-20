import mlx.core as mx
import mlx.nn as nn


_active_optimizer = None
_forward_depth = 0
_loss_plans = {}
_lineage = {}
_replayed_losses = {}
_suspended = False


class _ArraySlot:
    def __init__(self, index):
        self.index = index


class _TrainingPlan:
    def __init__(self, model, optimizer, objective, compile=True):
        self.model = model
        self.optimizer = optimizer
        self._value_and_grad = nn.value_and_grad(model, objective)
        self._compile_enabled = compile
        self._compiled_backward = None
        self._compiled_fused = None
        self._backward_signatures = set()
        self._fused_signatures = set()
        if compile:
            backward_state = [model.state, mx.random.state]
            fused_state = [model.state, optimizer.state, mx.random.state]
            self._compiled_backward = mx.compile(
                self._value_and_grad,
                inputs=backward_state,
                outputs=backward_state,
            )
            self._compiled_fused = mx.compile(
                self._fused,
                inputs=fused_state,
                outputs=fused_state,
            )

    def _fused(self, *inputs):
        loss, gradients = self._value_and_grad(*inputs)
        self.optimizer.update(self.model, gradients)
        return loss

    def backward(self, *inputs):
        if self._compile_enabled:
            return self._compiled_backward(*inputs)
        return self._value_and_grad(*inputs)

    def fused(self, *inputs):
        if self._compile_enabled:
            return self._compiled_fused(*inputs)
        return self._fused(*inputs)

    def disable_compile(self):
        self._compile_enabled = False


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
        return {key: _partition_arrays(item, arrays) for key, item in value.items()}
    return value


def _restore_arrays(template, arrays):
    if isinstance(template, _ArraySlot):
        return arrays[template.index]
    if isinstance(template, tuple):
        return tuple(_restore_arrays(item, arrays) for item in template)
    if isinstance(template, list):
        return [_restore_arrays(item, arrays) for item in template]
    if isinstance(template, dict):
        return {key: _restore_arrays(item, arrays) for key, item in template.items()}
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


def _copy_tree(value):
    if isinstance(value, dict):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    return value


def _snapshot_random_state():
    state = [key + mx.array(0, dtype=key.dtype) for key in mx.random.state]
    mx.eval(state)
    return state


def _restore_random_state(state):
    for current_key, saved_key in zip(mx.random.state, state):
        current_key[...] = saved_key
    mx.eval(mx.random.state)


def _compile_failure(error):
    message = str(error)
    return "Attempting to eval an array" in message and (
        "function transformations" in message or "without a primitive" in message
    )


def begin_forward(model):
    global _forward_depth
    if (
        _suspended
        or _active_optimizer is None
        or model is not _active_optimizer._model
    ):
        return False
    if _forward_depth:
        _active_optimizer._recursive_forward = True
    _forward_depth += 1
    return True


def note_eager_evaluation():
    if _forward_depth and _active_optimizer is not None and not _suspended:
        _active_optimizer._eager_forward = True


def end_forward(model, args, kwargs, output):
    global _forward_depth
    _forward_depth -= 1
    if _forward_depth != 0 or not isinstance(output, mx.array):
        return
    random_before = _active_optimizer._random_before_forward
    stochastic = any(
        current is not previous
        for current, previous in zip(mx.random.state, random_before)
    )
    descriptor = (
        model,
        args,
        kwargs,
        (),
        random_before if stochastic else None,
        not (
            _active_optimizer._recursive_forward
            or _active_optimizer._eager_forward
        ),
    )
    _lineage[id(output)] = (output, descriptor)


def abort_forward():
    global _forward_depth
    _forward_depth -= 1


def activate(optimizer):
    global _active_optimizer, _forward_depth
    _active_optimizer = optimizer
    _forward_depth = 0
    _loss_plans.clear()
    _lineage.clear()
    _replayed_losses.clear()
    optimizer._random_before_forward = tuple(mx.random.state)
    optimizer._recursive_forward = False
    optimizer._eager_forward = False


def propagate(source, result, operation, signature, operands=()):
    if _suspended:
        return result
    entry = _lineage.get(id(source))
    if entry is None or entry[0] is not source:
        return result
    model, args, kwargs, transforms, random_before, compile_allowed = entry[1]
    _lineage[id(result)] = (
        result,
        (
            model,
            args,
            kwargs,
            transforms + ((operation, signature, operands),),
            random_before,
            compile_allowed,
        ),
    )
    return result


def _plan_for(
    optimizer, model, cache_key, template, transforms, rebuild, compile_allowed
):
    plan = optimizer._training_plans.get(cache_key)
    if plan is not None:
        return plan

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

    plan = _TrainingPlan(
        model, optimizer._optimizer, objective, compile=compile_allowed
    )
    optimizer._training_plans[cache_key] = plan
    return plan


def register_loss(loss, loss_input, rebuild, operands, signature):
    if _suspended or _active_optimizer is None:
        return loss
    entry = _lineage.get(id(loss_input))
    if entry is None or entry[0] is not loss_input:
        return loss
    model, args, kwargs, transforms, random_before, compile_allowed = entry[1]
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
        compile_allowed,
    )
    _loss_plans[id(loss)] = (
        loss,
        model,
        cache_key,
        tuple(dynamic_inputs),
        template,
        transforms,
        rebuild,
        random_before,
        compile_allowed,
    )
    return loss


def backward(loss):
    if _active_optimizer is None:
        raise RuntimeError("optimizer.zero_grad() must be called before loss.backward()")
    entry = _loss_plans.get(id(loss))
    if entry is None or entry[0] is not loss:
        raise RuntimeError(
            "this MLX loss cannot use backward compatibility; compute it with a supported torchmlx loss function"
        )
    (
        _,
        model,
        cache_key,
        dynamic_inputs,
        template,
        transforms,
        rebuild,
        random_before,
        compile_allowed,
    ) = entry
    plan = _plan_for(
        _active_optimizer,
        model,
        cache_key,
        template,
        transforms,
        rebuild,
        compile_allowed,
    )
    _active_optimizer._pending_update = (
        "staged",
        loss,
        plan,
        dynamic_inputs,
        random_before,
    )


def _restore_execution(model, optimizer, parameters, optimizer_state):
    model.update(parameters)
    if optimizer_state is not None:
        _restore_tree(optimizer.state, optimizer_state)


def _restore_tree(current, saved):
    if isinstance(current, dict) and isinstance(saved, dict):
        for key in tuple(current):
            if key not in saved:
                del current[key]
        for key, value in saved.items():
            if key in current:
                restored = _restore_tree(current[key], value)
                if restored is not current[key]:
                    current[key] = restored
            else:
                current[key] = value
        return current
    if isinstance(current, list) and isinstance(saved, list):
        restored = [
            _restore_tree(old, value) for old, value in zip(current, saved)
        ]
        current[:] = restored + saved[len(restored) :]
        return current
    return saved


def _execute(plan, inputs, fused, random_before, advance_random=False):
    global _suspended
    model = plan.model
    optimizer = plan.optimizer
    signatures = plan._fused_signatures if fused else plan._backward_signatures
    signature = tuple((value.shape, value.dtype) for value in inputs)
    protect_state = not plan._compile_enabled or signature not in signatures
    parameters = (
        _copy_tree(model.trainable_parameters())
        if protect_state
        else getattr(optimizer, "_torchmlx_parameters", None)
    )
    optimizer_state = None
    if fused:
        optimizer_state = (
            _copy_tree(optimizer.state)
            if protect_state
            else getattr(optimizer, "_torchmlx_state", None)
        )
    random_after = _snapshot_random_state() if random_before is not None else None
    random_rollback = tuple(mx.random.state) if advance_random else None
    if random_before is not None:
        _restore_random_state(random_before)
    _suspended = True
    try:
        try:
            result = plan.fused(*inputs) if fused else plan.backward(*inputs)
            current_parameters = model.parameters()
            current_optimizer_state = _copy_tree(optimizer.state) if fused else []
            mx.eval(
                result,
                current_parameters,
                current_optimizer_state,
                mx.random.state
                if random_before is not None or advance_random
                else [],
            )
            signatures.add(signature)
            if fused:
                optimizer._torchmlx_parameters = current_parameters
                optimizer._torchmlx_state = current_optimizer_state
        except (ValueError, RuntimeError) as error:
            if not plan._compile_enabled or not _compile_failure(error):
                raise
            _restore_execution(model, optimizer, parameters, optimizer_state)
            if random_before is not None:
                _restore_random_state(random_before)
            elif random_rollback is not None:
                _restore_random_state(random_rollback)
            plan.disable_compile()
            result = plan.fused(*inputs) if fused else plan.backward(*inputs)
            current_parameters = model.parameters()
            current_optimizer_state = _copy_tree(optimizer.state) if fused else []
            mx.eval(
                result,
                current_parameters,
                current_optimizer_state,
                mx.random.state
                if random_before is not None or advance_random
                else [],
            )
            if fused:
                optimizer._torchmlx_parameters = current_parameters
                optimizer._torchmlx_state = current_optimizer_state
    except Exception:
        if parameters is not None:
            _restore_execution(model, optimizer, parameters, optimizer_state)
        if random_rollback is not None:
            _restore_random_state(random_rollback)
        raise
    finally:
        _suspended = False
        if random_after is not None:
            _restore_random_state(random_after)
    return result


def _materialize(optimizer):
    pending = optimizer._pending_update
    _, loss, plan, dynamic_inputs, random_before = pending
    replayed_loss, gradients = _execute(
        plan, dynamic_inputs, fused=False, random_before=random_before
    )
    _replayed_losses[id(loss)] = (loss, replayed_loss)
    optimizer._pending_update = ("materialized", loss, plan.model, gradients)
    return replayed_loss


def replayed_value(value):
    if _active_optimizer is not None:
        pending = _active_optimizer._pending_update
        if pending is not None and pending[0] == "staged" and pending[1] is value:
            return _materialize(_active_optimizer)
    entry = _replayed_losses.get(id(value))
    if entry is None or entry[0] is not value:
        return value
    return entry[1]


def _finish_step(loss, replayed_loss):
    global _active_optimizer, _forward_depth
    _replayed_losses[id(loss)] = (loss, replayed_loss)
    _active_optimizer._pending_update = None
    _active_optimizer = None
    _forward_depth = 0
    _loss_plans.clear()
    _lineage.clear()


def step(optimizer):
    pending = optimizer._pending_update
    if pending is None:
        raise RuntimeError("loss.backward() must be called before optimizer.step()")
    if pending[0] == "staged":
        _, loss, plan, dynamic_inputs, random_before = pending
        replayed_loss = _execute(
            plan, dynamic_inputs, fused=True, random_before=random_before
        )
    else:
        _, loss, model, gradients = pending
        parameters = _copy_tree(model.trainable_parameters())
        optimizer_state = _copy_tree(optimizer._optimizer.state)
        try:
            optimizer._compiled_step(gradients)
            current_parameters = model.parameters()
            current_optimizer_state = _copy_tree(optimizer._optimizer.state)
            mx.eval(current_parameters, current_optimizer_state)
        except Exception:
            _restore_execution(
                model, optimizer._optimizer, parameters, optimizer_state
            )
            raise
        optimizer._optimizer._torchmlx_parameters = current_parameters
        optimizer._optimizer._torchmlx_state = current_optimizer_state
        replayed_loss = _replayed_losses[id(loss)][1]
    _finish_step(loss, replayed_loss)


def trainer_step(trainer, x, y):
    context = (x, y)
    dynamic_inputs = []
    template = _partition_arrays(context, dynamic_inputs)
    cache_key = (
        getattr(trainer.model, "training", None),
        _template_key(template),
        (),
        ("trainer", id(trainer.loss_fn), trainer.compile),
    )
    plan = trainer.optimizer._training_plans.get(cache_key)
    if plan is None:

        def objective(*current_inputs):
            current_x, current_y = _restore_arrays(template, current_inputs)
            return trainer.loss_fn(trainer.model(current_x), current_y)

        plan = _TrainingPlan(
            trainer.model,
            trainer.optimizer._optimizer,
            objective,
            compile=trainer.compile,
        )
        trainer.optimizer._training_plans[cache_key] = plan
    return _execute(
        plan,
        tuple(dynamic_inputs),
        fused=True,
        random_before=None,
        advance_random=True,
    )
