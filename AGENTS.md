# Project instructions

- Keep code minimal, reliable, comment-free, and focused on user experience.
- Do not add tests. Validate with `examples/tinystories-llm/train.py` using the existing uv environment.
- Keep the same user code working comfortably on macOS with MLX and Colab with PyTorch.
- Treat TorchMLX as an educational experiment and state that clearly in the short README.
- Do not use em dashes anywhere in the project.
- Work on `main`. Do not commit or push unless explicitly asked.
- Use short lowercase commit messages.
- Expose only PyTorch-compatible user concepts. Do not introduce TorchMLX-specific abstractions, workflows, or configuration objects that users must learn. Backend machinery must remain private.
