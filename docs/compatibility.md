# Compatibility

TorchMLX targets the common transformer operations used by GPT-2, Llama 3, Qwen 3, and GPT-OSS style implementations.

Supported MLX operations include embeddings, linear layers, normalization building blocks, dropout, activations, causal attention, tensor shape operations, masks, top-k routing, and AdamW training.

MLX arrays remain native arrays. Torch-style tensor methods are installed on the native array type for the supported subset.

Boolean expert routing and `unique` execute eagerly because their output shapes control Python flow.

The standard training sequence works with `cross_entropy` losses applied directly or after reshape, view, slicing, transpose, squeeze, unsqueeze, or flatten operations:

```python
optimizer.zero_grad()
logits = model(input_tokens)
loss = F.cross_entropy(logits.reshape(-1, vocabulary_size), targets.reshape(-1))
loss.backward()
optimizer.step()
```

MLX implements this sequence by recording the outer model call and replaying the forward, backward, and AdamW update inside one cached compiled graph during `step`. Inputs and loss operands remain dynamic, so batches are not captured as constants. Random state is restored for the replay so dropout uses the same mask. Reading `loss.item()` before `step` materializes gradients through a separate compiled path. Models with eager data-dependent operations fall back to an uncompiled replay. Unrecorded loss expressions, gradient hooks, parameter `.grad`, higher-order gradients, and multiple-forward losses remain unsupported.

Set `TORCHMLX_BACKEND=torch` before import to use native PyTorch for unsupported programs. TorchMLX never changes backend during an operation.

The referenced OpenArch model files contain source errors independent of TorchMLX, including invalid constructor calls and undefined attributes. Correct those errors before using either backend.
