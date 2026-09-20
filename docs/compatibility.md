# Compatibility

TorchMLX targets the common transformer operations used by GPT-2, Llama 3, Qwen 3, and GPT-OSS style implementations.

Supported MLX operations include embeddings, linear layers, normalization building blocks, dropout, activations, causal attention, tensor shape operations, masks, top-k routing, and AdamW training.

On MLX, TorchMLX tensors are an internal MLX array subclass. Native MLX arrays and their methods are not modified. Model parameters remain native MLX arrays internally and are presented as TorchMLX tensors through the PyTorch-shaped interface.

Boolean expert routing and `unique` execute eagerly because their output shapes control Python flow.

The standard training sequence works with `cross_entropy` losses applied directly or after reshape, view, slicing, transpose, squeeze, unsqueeze, or flatten operations:

```python
optimizer.zero_grad()
logits = model(input_tokens)
loss = F.cross_entropy(logits.reshape(-1, vocabulary_size), targets.reshape(-1))
loss.backward()
optimizer.step()
```

MLX implements this sequence by recording the outer model call and replaying the forward, backward, and AdamW update inside one cached compiled graph during `step`. Inputs and loss operands remain dynamic, so batches are not captured as constants. Random state is restored for the replay so dropout uses the same mask. Reading `loss.item()` after `backward` and before `step` materializes gradients through a separate compiled path. Models with eager data-dependent operations fall back to an uncompiled replay.

The supported subset includes tensor construction, dtype conversion, indexing, arithmetic, comparisons, reshape and view operations, transpose, squeeze and unsqueeze, flatten, expand, chunk, masking, common transformer math, embeddings, linear layers, layer normalization, dropout, GELU, causal attention, cross entropy, and AdamW with its default feature set.

Unrecorded loss expressions, losses combining multiple model forwards, gradient hooks, parameter `.grad`, higher-order gradients, serialization, additional optimizers, and additional losses remain unsupported on MLX. Use native PyTorch when a program needs behavior outside this subset.

TorchMLX selects MLX on Apple silicon and PyTorch elsewhere. `TORCHMLX_BACKEND=torch` and `TORCHMLX_BACKEND=mlx` are internal validation overrides. TorchMLX never changes backend during an operation.

The referenced OpenArch model files contain source errors independent of TorchMLX, including invalid constructor calls and undefined attributes. Correct those errors before using either backend.
