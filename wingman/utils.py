"""Some simple Wingman functionality."""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
    ) from e

__device = "cpu"
__device = "mps" if torch.cuda.is_available() else __device
__device = "cuda:0" if torch.backends.mps.is_available() else __device


def gpuize(
    input, device: str = __device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """gpuize.

    Args:
        input: the array that we want to gpuize
        device: a string of the device we want to move the thing to
        dtype: the datatype that the returned tensor should be
    """
    if torch.is_tensor(input):
        return input.to(device=device, dtype=dtype)
    else:
        return torch.tensor(input, device=device, dtype=dtype)


def cpuize(input) -> np.ndarray:
    """cpuize.

    Args:
        input: the array of the thing we want to put on the cpu
    """
    if torch.is_tensor(input):
        return input.detach().cpu().numpy()
    else:
        return input


def shutdown_handler(*_):
    """shutdown_handler.

    Args:
        _:
    """
    print("ctrl-c invoked")
    exit(0)
