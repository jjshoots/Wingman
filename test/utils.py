"""Utilities used during testing."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch

from wingman.replay_buffer import FlatReplayBuffer
from wingman.replay_buffer.core import ReplayBuffer
from wingman.replay_buffer.wrappers.dict_wrapper import DictReplayBufferWrapper


def create_shapes(
    use_dict: bool, bulk_size: int = 0
) -> list[tuple[int, ...] | dict[str, Any]]:
    """create_shapes.

    Args:
    ----
        use_dict (bool): use_dict
        bulk_size (int): bulk_size

    Returns:
    -------
        list[tuple[int, ...] | dict[str, Any]]:

    """
    if bulk_size:
        bulk_shape = (bulk_size,)
    else:
        bulk_shape = ()

    if use_dict:
        return [
            (*bulk_shape, 3, 3),
            (
                *bulk_shape,
                3,
            ),
            (*bulk_shape,),
            {
                "a": (*bulk_shape, 4, 3),
                "b": (*bulk_shape,),
                "c": {
                    "d": (*bulk_shape, 11, 2),
                },
            },
            {
                "e": (*bulk_shape, 3, 2),
            },
            (*bulk_shape, 4),
        ]
    else:
        return [
            (*bulk_shape, 3, 3),
            (*bulk_shape, 3),
            (*bulk_shape,),
        ]


def create_memory(
    mem_size: int,
    mode: Literal["numpy", "torch"],
    device: torch.device,
    store_on_device: bool,
    random_rollover: bool,
    use_dict: bool,
) -> ReplayBuffer:
    """create_memory.

    Args:
    ----
        mem_size (int): mem_size
        mode (Literal["numpy", "torch"]): mode
        device (torch.device): device
        store_on_device (bool): store_on_device
        random_rollover (bool): random_rollover
        use_dict (bool): use_dict

    Returns:
    -------
        ReplayBuffer:

    """
    memory = FlatReplayBuffer(
        mem_size=mem_size,
        mode=mode,
        device=device,
        store_on_device=store_on_device,
        random_rollover=random_rollover,
    )

    if use_dict:
        memory = DictReplayBufferWrapper(
            replay_buffer=memory,
        )

    return memory


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


def _dict_element_to_bulk_dim_swap(
    data_dict: dict[str, Any],
    bulk_size: int,
) -> list[dict[str, Any]]:
    """Given a nested dictionary where each leaf is an n-long array, returns an n-long sequence where each item is the same nested dictionary structure.

    Args:
    ----
        data_dict (dict[str, Any]): data_dict
        bulk_size (int): bulk_size

    Returns:
    -------
        list[dict[str, Any]]:

    """
    bulk_first_dicts: list[dict[str, Any]] = [dict() for _ in range(bulk_size)]
    for key, value in data_dict.items():
        if isinstance(value, dict):
            for i, element in enumerate(
                _dict_element_to_bulk_dim_swap(data_dict=value, bulk_size=bulk_size)
            ):
                bulk_first_dicts[i][key] = element
        else:
            for i, element in enumerate(value):
                bulk_first_dicts[i][key] = element

    return bulk_first_dicts


def element_to_bulk_dim_swap(
    element_first_data: list[Any],
    bulk_size: int,
) -> list[Any]:
    """Given a tuple of elements, each with `bulk_size` items, returns a `bulk_size` sequence, with each item being the size of the tuple.

    Args:
    ----
        element_first_data (list[Any]): element_first_data
        bulk_size (int): bulk_size

    Returns:
    -------
        list[Any]:

    """
    bulk_first_data = [[] for _ in range(bulk_size)]
    for element in element_first_data:
        # if not a dictionary, can do a plain axis extract
        if not isinstance(element, dict):
            for i in range(bulk_size):
                bulk_first_data[i].append(element[i])

        # if it's a dictionary, then we need to unpack the dictionary into each item
        else:
            for i, dict_element in enumerate(
                _dict_element_to_bulk_dim_swap(
                    data_dict=element,
                    bulk_size=bulk_size,
                )
            ):
                bulk_first_data[i].append(dict_element)

    return bulk_first_data


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


def are_equivalent(
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
            are_equivalent(d1, d2) for d1, d2 in zip(item1, item2)
        )

    # comparison for dictionaries
    if isinstance(item1, dict) and isinstance(item2, dict):
        return item1.keys() == item2.keys() and all(
            are_equivalent(item1[key], item2[key]) for key in item1.keys()
        )

    # non of the checks passed
    return False
