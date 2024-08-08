"""Some simple Wingman functionality."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
    ) from e

__device = "cpu"
__device = "mps" if torch.backends.mps.is_available() else __device
__device = "cuda:0" if torch.cuda.is_available() else __device


def gpuize(
    input: np.ndarray | torch.Tensor,
    device: str | torch.device = __device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """gpuize.

    Args:
    ----
        input (np.ndarray | torch.Tensor): the array that we want to gpuize
        device (str | torch.device): a string of the device we want to move the thing to
        dtype (torch.dtype): the datatype that the returned tensor should be

    """
    if torch.is_tensor(input):
        return input.to(device=device, dtype=dtype)  # pyright: ignore[reportAttributeAccessIssue]
    else:
        return torch.tensor(input, device=device, dtype=dtype)


def nested_gpuize(
    input: dict[str, Any],
    device: str | torch.device = __device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Gpuize but for nested dictionaries of elements.

    Args:
    ----
        input (dict[str, Any]): the array that we want to gpuize
        device (str | torch.device): a string of the device we want to move the thing to
        dtype (torch.dtype): the datatype that the returned tensor should be

    """
    output = {}
    for key, value in input.items():
        if isinstance(value, dict):
            output[key] = nested_gpuize(value, device=device, dtype=dtype)
        else:
            output[key] = gpuize(value)
    return output


def cpuize(input: np.ndarray | torch.Tensor) -> np.ndarray:
    """cpuize.

    Args:
    ----
        input (np.ndarray | torch.Tensor): the array of the thing we want to put on the cpu

    """
    if torch.is_tensor(input):
        return input.detach().cpu().numpy()  # pyright: ignore[reportAttributeAccessIssue]
    else:
        return input  # pyright: ignore[reportReturnType]


def nested_cpuize(input: dict[str, Any]) -> dict[str, Any]:
    """Gpuize but for nested dictionaries of elements.

    Args:
    ----
        input (dict[str, Any]): the array of the thing we want to put on the cpu

    """
    output = {}
    for key, value in input.items():
        if isinstance(value, dict):
            output[key] = nested_cpuize(value)
        else:
            output[key] = cpuize(value)
    return output


def shutdown_handler(*_):
    """shutdown_handler.

    Args:
    ----
        _:

    """
    print("ctrl-c invoked")
    exit(0)
