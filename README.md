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

| backend | 5,000 steps + generation | effective steps/s |
| --- | ---: | ---: |
| torchmlx mlx, m3 pro | 170.27 s | 29.4 |
| pytorch mps, m3 pro | 184.42 s | 27.1 |
