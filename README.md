# torchmlx

torchmlx is a pytorch-shaped compatibility layer that uses mlx on apple silicon and pytorch elsewhere.

NOTE: it is experimental and built first for educational use. If you find a bug, please create an issue on the repo.

```python
import torchmlx as torch
from torchmlx import nn, optim

model = nn.Linear(4, 2)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
```

see the [tinystories example](examples/tinystories-llm/train.py) and [compatibility details](docs/compatibility.md).
