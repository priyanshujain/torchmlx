# torchmlx

torchmlx is a pytorch-shaped compatibility layer that uses mlx on apple silicon and pytorch elsewhere. it is experimental and built first for educational use.

```bash
pip install pytorchmlx
```

```python
import torchmlx as torch
from torchmlx import nn, optim
```

see the [tinystories example](examples/tinystories-llm/train.py) and [compatibility details](docs/compatibility.md).
