# pyright: reportAssignmentType=false, reportRedeclaration=false

import math

from torchmlx._backend import BACKEND, unsupported


if BACKEND == "torch":
    import torch.nn.functional as _native

    cross_entropy = _native.cross_entropy
    scaled_dot_product_attention = _native.scaled_dot_product_attention
    silu = _native.silu
    softmax = _native.softmax

    def __getattr__(name):
        return getattr(_native, name)

else:
    import mlx.core as mx
    from torchmlx._mlx_tensor import wrap

    def cross_entropy(
        input,
        target,
        weight=None,
        size_average=None,
        ignore_index=-100,
        reduce=None,
        reduction="mean",
        label_smoothing=0.0,
    ):
        original_input = input

        def finish(loss):
            from torchmlx._autograd import register_loss

            def rebuild(recomputed_input, current_target, *current_weight):
                return cross_entropy(
                    recomputed_input,
                    current_target,
                    weight=current_weight[0] if current_weight else None,
                    size_average=size_average,
                    ignore_index=ignore_index,
                    reduce=reduce,
                    reduction=reduction,
                    label_smoothing=label_smoothing,
                )

            operands = (target,) if weight is None else (target, weight)
            signature = (
                "cross_entropy",
                weight is not None,
                size_average,
                ignore_index,
                reduce,
                reduction,
                label_smoothing,
            )
            return register_loss(
                wrap(loss),
                original_input,
                rebuild,
                operands,
                signature,
            )

        if size_average is not None or reduce is not None:
            unsupported("torchmlx.nn.functional.cross_entropy legacy reductions")
        if input.ndim < 2:
            raise ValueError("cross_entropy input must have at least 2 dimensions")
        if input.ndim == 2:
            logits = input
            targets = target.reshape(-1)
            output_shape = target.shape
        else:
            axes = [0] + list(range(2, input.ndim)) + [1]
            logits = input.transpose(axes).reshape(-1, input.shape[1])
            targets = target.reshape(-1)
            output_shape = target.shape
        valid = targets != ignore_index
        safe_targets = mx.where(valid, targets, mx.zeros_like(targets))
        log_probabilities = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target_losses = -mx.take_along_axis(
            log_probabilities, safe_targets[:, None], axis=-1
        ).squeeze(-1)
        if weight is None:
            smooth_losses = -mx.mean(log_probabilities, axis=-1)
        else:
            target_losses = target_losses * weight[safe_targets]
            smooth_losses = -mx.sum(
                log_probabilities * weight[None, :], axis=-1
            ) / logits.shape[-1]
        losses = (1 - label_smoothing) * target_losses + label_smoothing * smooth_losses
        losses = mx.where(valid, losses, mx.zeros_like(losses))
        if reduction == "none":
            return finish(losses.reshape(output_shape))
        if reduction == "sum":
            return finish(mx.sum(losses))
        if reduction == "mean":
            if weight is None:
                denominator = mx.sum(valid)
            else:
                denominator = mx.sum(mx.where(valid, weight[safe_targets], 0))
            return finish(mx.sum(losses) / denominator)
        raise ValueError(f"invalid reduction {reduction!r}")

    def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        enable_gqa=False,
    ):
        if dropout_p != 0.0:
            unsupported(
                "torchmlx.nn.functional.scaled_dot_product_attention with dropout"
            )
        if enable_gqa:
            unsupported(
                "torchmlx.nn.functional.scaled_dot_product_attention with enable_gqa"
            )
        if is_causal and attn_mask is not None:
            raise ValueError("attn_mask cannot be set when is_causal=True")
        if scale is None:
            scale = 1 / math.sqrt(query.shape[-1])
        mask = "causal" if is_causal else attn_mask
        return wrap(
            mx.fast.scaled_dot_product_attention(
                query, key, value, scale=scale, mask=mask
            )
        )

    def silu(input, inplace=False):
        if inplace:
            unsupported("torchmlx.nn.functional.silu with inplace=True")
        return wrap(input * mx.sigmoid(input))

    def softmax(input, dim=None, dtype=None):
        value = input.astype(dtype) if dtype is not None else input
        return wrap(mx.softmax(value, axis=dim))

    def __getattr__(name):
        unsupported(f"torchmlx.nn.functional.{name}")


__all__ = ["cross_entropy", "scaled_dot_product_attention", "silu", "softmax"]
