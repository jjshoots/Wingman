"""Some simple Wingman functionality."""
from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
    ) from e


def gpuize(input, device: str = "cuda:0", keeptype: bool = False) -> torch.Tensor:
    """gpuize.

    Args:
        input: the array that we want to gpuize
        device: a string of the device we want to move the thing to
        keeptype: a boolean on whether to keep the original dtype, otherwise, the tensor is converted to torch.float64
    """
    dtype = input.dtype if keeptype else torch.float64
    return input.to(device, dtype=dtype)


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
