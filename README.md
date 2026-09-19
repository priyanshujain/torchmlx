# TorchMLX

TorchMLX is a PyTorch-shaped compatibility layer that uses MLX on Apple silicon and PyTorch elsewhere.

It is experimental and targets transformer inference and training without per-operation backend fallback.

```python
import torchmlx as torch
from torchmlx import nn, optim

model = nn.Linear(4, 2)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
```

See the [TinyStories example](examples/tinystories-llm/train.py) and [compatibility details](docs/compatibility.md).
