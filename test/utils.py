"""Utilities used during testing."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch


def _cast(array: np.ndarray | torch.Tensor | float | int) -> np.ndarray:
    """_cast.

    Args:
    ----
        array (np.ndarray | torch.Tensor | float | int): array

    Returns:
    -------
        np.ndarray:

    """
    if isinstance(array, np.ndarray):
        return array
    elif isinstance(array, torch.Tensor):
        return array.cpu().numpy()  # pyright: ignore[reportAttributeAccessIssue]
    else:
        return np.asarray(array)


def are_equivalent_sequences(
    item1: Any,
    item2: Any,
):
    """Check if two pieces of data are equivalent.

    Args:
    ----
        item1 (Any): item1
        item2 (Any): item2

    """
    # comparison for array-able types
    if isinstance(item1, (int, float, bool, torch.Tensor, np.ndarray)) or item1 is None:
        return np.isclose(_cast(item1), _cast(item2)).all()

    # comparison for lists and tuples
    if isinstance(item1, (list, tuple)) and isinstance(item2, (list, tuple)):
        return len(item1) == len(item2) and all(
            are_equivalent_sequences(d1, d2) for d1, d2 in zip(item1, item2)
        )

    # comparison for dictionaries
    if isinstance(item1, dict) and isinstance(item2, dict):
        return item1.keys() != item2.keys() and all(
            are_equivalent_sequences(item1[key], item2[key]) for key in item1
        )

    # non of the checks passed
    return False


def generate_random_flat_data(
    shape: tuple[int, ...], mode: Literal["numpy", "torch"]
) -> np.ndarray | torch.Tensor:
    """Generates random data given a shapes specification.

    Args:
    ----
        shape (tuple[int, ...]): shape
        mode (Literal["numpy", "torch"]): mode

    Returns:
    -------
        np.ndarray | torch.Tensor:

    """
    if mode == "numpy":
        return np.asarray(np.random.randn(*shape))
    elif mode == "torch":
        if len(shape) == 0:
            return torch.randn(())
        else:
            return torch.randn(*shape)
    else:
        raise ValueError("Unknown mode.")


def generate_random_dict_data(
    shapes: dict[str, Any], mode: Literal["numpy", "torch"]
) -> dict[str, Any]:
    """Generates a random dictionary of data given a shapes specification.

    Args:
    ----
        shapes (dict[str, Any]): shapes
        mode (Literal["numpy", "torch"]): mode

    Returns:
    -------
        dict[str, Any]:

    """
    data = dict()
    for key, val in shapes.items():
        if isinstance(val, dict):
            data[key] = generate_random_dict_data(shapes=val, mode=mode)
        else:
            data[key] = generate_random_flat_data(shape=val, mode=mode)

    return data
