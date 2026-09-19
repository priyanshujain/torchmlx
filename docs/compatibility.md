# Compatibility

TorchMLX targets the common transformer operations used by GPT-2, Llama 3, Qwen 3, and GPT-OSS style implementations.

Supported MLX operations include embeddings, linear layers, normalization building blocks, dropout, activations, causal attention, tensor shape operations, masks, top-k routing, and AdamW training through `Trainer`.

MLX arrays remain native arrays. Torch-style tensor methods are installed on the native array type for the supported subset.

Boolean expert routing and `unique` execute eagerly because their output shapes control Python flow.

Set `TORCHMLX_BACKEND=torch` before import to use native PyTorch for unsupported programs. TorchMLX never changes backend during an operation.

The referenced OpenArch model files contain source errors independent of TorchMLX, including invalid constructor calls and undefined attributes. Correct those errors before using either backend.
