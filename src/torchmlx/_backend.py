import os
import platform


def _select_backend():
    override = os.environ.get("TORCHMLX_BACKEND")
    if override is not None:
        backend = override.lower()
        if backend not in {"mlx", "torch"}:
            raise RuntimeError(
                "TORCHMLX_BACKEND must be either 'mlx' or 'torch', "
                f"not {override!r}"
            )
        return backend
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mlx"
    return "torch"


BACKEND = _select_backend()


def unsupported(api):
    raise NotImplementedError(
        f"{api} is not supported by the MLX backend. "
        "Set TORCHMLX_BACKEND=torch before importing torchmlx to use PyTorch."
    )
