from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, cast

import mlx.core as mx
import mlx.nn as nn

from ._mlx_tensor import Tensor


_active_optimizer: ContextVar[Any | None] = ContextVar(
    "torchmlx_active_optimizer", default=None
)
_forward_depth: ContextVar[int] = ContextVar("torchmlx_forward_depth", default=0)
_suspended: ContextVar[bool] = ContextVar("torchmlx_suspended", default=False)


@dataclass(frozen=True)
class _ArraySlot:
    index: int


@dataclass(frozen=True)
class _Lineage:
    model: Any
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    transforms: tuple[Any, ...]
    random_before: tuple[mx.array, ...] | None
    compile_allowed: bool


@dataclass(frozen=True)
class _LossPlan:
    model: Any
    cache_key: tuple[Any, ...]
    dynamic_inputs: tuple[mx.array, ...]
    template: Any
    transforms: tuple[Any, ...]
    rebuild: Callable[..., mx.array]
    random_before: tuple[mx.array, ...] | None
    compile_allowed: bool


class _TrainingPlan:
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        objective: Callable[..., mx.array],
        compile: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self._value_and_grad = nn.value_and_grad(model, objective)
        self._compile_enabled = compile
        self._compiled_backward: Callable[..., Any] | None = None
        self._compiled_fused: Callable[..., Any] | None = None
        self._backward_signatures: set[Any] = set()
        self._fused_signatures: set[Any] = set()
        if compile:
            backward_state = [model.state, mx.random.state]
            fused_state = [model.state, optimizer.state, mx.random.state]
            self._compiled_backward = mx.compile(
                self._value_and_grad, inputs=backward_state, outputs=backward_state
            )
            self._compiled_fused = mx.compile(
                self._fused, inputs=fused_state, outputs=fused_state
            )

    def _fused(self, *inputs: mx.array) -> mx.array:
        loss, gradients = self._value_and_grad(*inputs)
        self.optimizer.update(self.model, gradients)
        return loss

    def backward(self, *inputs: mx.array) -> Any:
        if self._compile_enabled:
            assert self._compiled_backward is not None
            return self._compiled_backward(*inputs)
        return self._value_and_grad(*inputs)

    def fused(self, *inputs: mx.array) -> mx.array:
        if self._compile_enabled:
            assert self._compiled_fused is not None
            return self._compiled_fused(*inputs)
        return self._fused(*inputs)

    def disable_compile(self) -> None:
        self._compile_enabled = False


def _partition_arrays(value: Any, arrays: list[mx.array]) -> Any:
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


def _restore_arrays(template: Any, arrays: tuple[mx.array, ...]) -> Any:
    if isinstance(template, _ArraySlot):
        return arrays[template.index]
    if isinstance(template, tuple):
        return tuple(_restore_arrays(item, arrays) for item in template)
    if isinstance(template, list):
        return [_restore_arrays(item, arrays) for item in template]
    if isinstance(template, dict):
        return {key: _restore_arrays(item, arrays) for key, item in template.items()}
    return template


def _template_key(template: Any) -> tuple[Any, ...]:
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


def _copy_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_tree(item) for item in value)
    return value


def _snapshot_random_state() -> list[mx.array]:
    random_state = cast(Any, mx.random.state)
    state = [key + mx.array(0, dtype=key.dtype) for key in random_state]
    mx.eval(state)
    return state


def _restore_random_state(state: Any) -> None:
    for current_key, saved_key in zip(cast(Any, mx.random.state), state):
        current_key[...] = saved_key
    mx.eval(mx.random.state)


def _compile_failure(error: Exception) -> bool:
    message = str(error)
    return "Attempting to eval an array" in message and (
        "function transformations" in message or "without a primitive" in message
    )


def begin_forward(model: Any) -> bool:
    optimizer = _active_optimizer.get()
    if _suspended.get() or optimizer is None or model is not optimizer._model:
        return False
    depth = _forward_depth.get()
    if depth:
        optimizer._recursive_forward = True
    _forward_depth.set(depth + 1)
    return True


def note_eager_evaluation() -> None:
    optimizer = _active_optimizer.get()
    if _forward_depth.get() and optimizer is not None and not _suspended.get():
        optimizer._eager_forward = True


def end_forward(
    model: Any, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any
) -> None:
    optimizer = _active_optimizer.get()
    depth = _forward_depth.get() - 1
    _forward_depth.set(depth)
    if depth != 0 or optimizer is None or not isinstance(output, Tensor):
        return
    random_before = optimizer._random_before_forward
    stochastic = any(
        current is not previous
        for current, previous in zip(cast(Any, mx.random.state), random_before)
    )
    output._lineage = _Lineage(
        model,
        args,
        kwargs,
        (),
        random_before if stochastic else None,
        not (optimizer._recursive_forward or optimizer._eager_forward),
    )


def abort_forward() -> None:
    depth = max(0, _forward_depth.get() - 1)
    _forward_depth.set(depth)
    if depth == 0:
        optimizer = _active_optimizer.get()
        if optimizer is not None:
            optimizer._pending_update = None
        _active_optimizer.set(None)


def activate(optimizer: Any) -> None:
    previous = _active_optimizer.get()
    if previous is not None and previous is not optimizer:
        previous._pending_update = None
    _active_optimizer.set(optimizer)
    _forward_depth.set(0)
    optimizer._random_before_forward = tuple(cast(Any, mx.random.state))
    optimizer._recursive_forward = False
    optimizer._eager_forward = False


def propagate(
    source: Tensor,
    result: Tensor,
    operation: Callable[..., mx.array],
    signature: tuple[Any, ...],
    operands: tuple[mx.array, ...] = (),
) -> Tensor:
    if _suspended.get() or source._lineage is None:
        return result
    lineage: _Lineage = source._lineage
    result._lineage = _Lineage(
        lineage.model,
        lineage.args,
        lineage.kwargs,
        lineage.transforms + ((operation, signature, operands),),
        lineage.random_before,
        lineage.compile_allowed,
    )
    return result


def _plan_for(optimizer: Any, loss_plan: _LossPlan) -> _TrainingPlan:
    plan = optimizer._training_plans.get(loss_plan.cache_key)
    if plan is not None:
        return plan

    def objective(*current_inputs: mx.array) -> mx.array:
        current_args, current_kwargs, transform_operands, current_operands = (
            _restore_arrays(loss_plan.template, current_inputs)
        )
        output = loss_plan.model(*current_args, **current_kwargs)
        for (operation, _, _), operands in zip(
            loss_plan.transforms, transform_operands
        ):
            output = operation(output, *operands)
        return loss_plan.rebuild(output, *current_operands)

    plan = _TrainingPlan(
        loss_plan.model,
        optimizer._optimizer,
        objective,
        compile=loss_plan.compile_allowed,
    )
    optimizer._training_plans[loss_plan.cache_key] = plan
    return plan


def register_loss(
    loss: Tensor,
    loss_input: Tensor,
    rebuild: Callable[..., mx.array],
    operands: tuple[mx.array, ...],
    signature: tuple[Any, ...],
) -> Tensor:
    optimizer = _active_optimizer.get()
    if _suspended.get() or optimizer is None or loss_input._lineage is None:
        return loss
    lineage: _Lineage = loss_input._lineage
    context = (
        lineage.args,
        lineage.kwargs,
        tuple(cast(Any, item)[2] for item in lineage.transforms),
        operands,
    )
    dynamic_inputs: list[mx.array] = []
    template = _partition_arrays(context, dynamic_inputs)
    cache_key = (
        getattr(lineage.model, "training", None),
        _template_key(template),
        tuple(item[1] for item in lineage.transforms),
        signature,
        lineage.compile_allowed,
    )
    loss._loss_plan = _LossPlan(
        lineage.model,
        cache_key,
        tuple(dynamic_inputs),
        template,
        lineage.transforms,
        rebuild,
        lineage.random_before,
        lineage.compile_allowed,
    )
    return loss


def backward(loss: Tensor) -> None:
    optimizer = _active_optimizer.get()
    if optimizer is None:
        raise RuntimeError("optimizer.zero_grad() must be called before loss.backward()")
    loss_plan = loss._loss_plan
    if loss_plan is None:
        _clear(optimizer)
        raise RuntimeError(
            "this MLX loss cannot use backward compatibility; compute one supported loss from one model forward"
        )
    try:
        plan = _plan_for(optimizer, loss_plan)
    except Exception:
        _clear(optimizer)
        raise
    optimizer._pending_update = (
        "staged",
        loss,
        plan,
        loss_plan.dynamic_inputs,
        loss_plan.random_before,
    )


def _restore_tree(current: Any, saved: Any) -> Any:
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
        restored = [_restore_tree(old, value) for old, value in zip(current, saved)]
        current[:] = restored + saved[len(restored) :]
        return current
    return saved


def _restore_execution(
    model: Any, optimizer: Any, parameters: Any, optimizer_state: Any
) -> None:
    model.update(parameters)
    if optimizer_state is not None:
        _restore_tree(optimizer.state, optimizer_state)


def _execute(
    plan: _TrainingPlan,
    inputs: tuple[mx.array, ...],
    fused: bool,
    random_before: Any,
) -> Any:
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
    if random_before is not None:
        _restore_random_state(random_before)
    token = _suspended.set(True)
    try:
        try:
            result = plan.fused(*inputs) if fused else plan.backward(*inputs)
            current_parameters = model._native_parameters()
            current_optimizer_state = _copy_tree(optimizer.state) if fused else []
            mx.eval(
                result,
                current_parameters,
                current_optimizer_state,
                mx.random.state if random_before is not None else [],
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
            plan.disable_compile()
            result = plan.fused(*inputs) if fused else plan.backward(*inputs)
            current_parameters = model._native_parameters()
            current_optimizer_state = _copy_tree(optimizer.state) if fused else []
            mx.eval(
                result,
                current_parameters,
                current_optimizer_state,
                mx.random.state if random_before is not None else [],
            )
            if fused:
                optimizer._torchmlx_parameters = current_parameters
                optimizer._torchmlx_state = current_optimizer_state
    except Exception:
        if parameters is not None:
            _restore_execution(model, optimizer, parameters, optimizer_state)
        raise
    finally:
        _suspended.reset(token)
        if random_after is not None:
            _restore_random_state(random_after)
    return result


def _materialize(optimizer: Any) -> mx.array:
    _, loss, plan, dynamic_inputs, random_before = optimizer._pending_update
    replayed_loss, gradients = _execute(
        plan, dynamic_inputs, fused=False, random_before=random_before
    )
    loss._replayed_value = replayed_loss
    optimizer._pending_update = ("materialized", loss, plan.model, gradients)
    return replayed_loss


def replayed_value(value: Tensor) -> mx.array:
    optimizer = _active_optimizer.get()
    if optimizer is not None:
        pending = optimizer._pending_update
        if pending is not None and pending[0] == "staged" and pending[1] is value:
            return _materialize(optimizer)
    return value._replayed_value if value._replayed_value is not None else value


def _clear(optimizer: Any) -> None:
    optimizer._pending_update = None
    _active_optimizer.set(None)
    _forward_depth.set(0)


def step(optimizer: Any) -> None:
    pending = optimizer._pending_update
    if pending is None:
        _clear(optimizer)
        raise RuntimeError("loss.backward() must be called before optimizer.step()")
    try:
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
                current_parameters = model._native_parameters()
                current_optimizer_state = _copy_tree(optimizer._optimizer.state)
                mx.eval(current_parameters, current_optimizer_state)
            except Exception:
                _restore_execution(model, optimizer._optimizer, parameters, optimizer_state)
                raise
            optimizer._optimizer._torchmlx_parameters = current_parameters
            optimizer._optimizer._torchmlx_state = current_optimizer_state
            replayed_loss = loss._replayed_value
        loss._replayed_value = replayed_loss
    finally:
        _clear(optimizer)
