"""Tests the replay buffer module."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pprint import pformat
from typing import Literal, Sequence

import numpy as np
import pytest
import torch

from wingman.replay_buffer import ReplayBuffer


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


def _is_equivalent_tuple(
    item1: Sequence[np.ndarray | torch.Tensor | float | int],
    item2: Sequence[np.ndarray | torch.Tensor | float | int],
) -> bool:
    """Checks whether a two tuples of np.ndarrays are equivalent to each other.

    Args:
    ----
        item1 (tuple[np.ndarray | torch.Tensor | float | int]): item1
        item2 (tuple[np.ndarray | torch.Tensor | float | int]): item2

    Returns:
    -------
        bool:

    """
    equivalence = True

    item1 = [_cast(array) for array in item1]
    item2 = [_cast(array) for array in item2]
    for i1, i2 in zip(item1, item2):
        equivalence = np.isclose(i1, i2).all() and equivalence
    return bool(equivalence)


def _randn(
    shape: tuple[int], mode: Literal["numpy", "torch"]
) -> np.ndarray | torch.Tensor:
    """_randn.

    Args:
    ----
        shape (tuple[int]): shape
        mode (Literal["numpy", "torch"]): mode

    Returns:
    -------
        np.ndarray | torch.Tensor:

    """
    if mode == "numpy":
        return np.random.randn(*shape)
    elif mode == "torch":
        if len(shape) == 0:
            return torch.randn(())
        else:
            return torch.randn(*shape)
    else:
        raise ValueError("Unknown mode.")


# define the test configurations
_random_rollovers = [True, False]
_modes = ["numpy", "torch"]
_devices = [torch.device("cpu")]
if torch.cuda.is_available():
    _devices.append(torch.device("cuda:0"))
_store_on_devices = [True, False]
ALL_CONFIGURATIONS = product(
    _random_rollovers,
    _modes,
    _devices,
    _store_on_devices,
)


@pytest.mark.parametrize(
    "random_rollover, mode, device, store_on_device",
    ALL_CONFIGURATIONS,
)
def test_bulk(
    random_rollover: bool,
    mode: Literal["numpy", "torch"],
    device: torch.device,
    store_on_device: bool,
):
    """Tests repeatedly bulking the buffer and whether it rollovers correctly."""
    bulk_size = 7
    mem_size = 11
    element_shapes = [(3, 3), (3,), ()]
    memory = ReplayBuffer(
        mem_size=mem_size, mode=mode, device=device, store_on_device=store_on_device
    )

    for iteration in range(10):
        # try to stuff:
        # a) (bulk_size, 3) array
        # b) (bulk_size,) array
        data = []
        for shape in element_shapes:
            data.append(_randn(shape=(bulk_size, *shape), mode=mode))
        print([d.shape for d in data])
        memory.push(data, bulk=True, random_rollover=random_rollover)

        # reverse the data to make indexing for checking easier
        reversed_data = [list(item) for item in zip(*data)]

        # if random rollover and we're more than full, different matching technique
        if random_rollover and memory.is_full:
            num_matches = 0
            # match according to meshgrid
            for item1 in reversed_data:
                for item2 in memory:
                    num_matches += int(_is_equivalent_tuple(item1, item2))

            assert (
                num_matches == bulk_size
            ), f"""Expected {bulk_size} matches inside the memory, got {num_matches}."""

            continue

        for step in range(bulk_size):
            item1 = reversed_data[step]
            item2 = memory.__getitem__((iteration * bulk_size + step) % mem_size)
            assert _is_equivalent_tuple(
                item1, item2
            ), f"""Something went wrong with rollover at iteration {iteration},
                step {step}, expected \n{pformat(item1)}, got \n{pformat(item2)}."""


@pytest.mark.parametrize(
    "random_rollover, mode, device, store_on_device",
    ALL_CONFIGURATIONS,
)
def test_non_bulk(
    random_rollover: bool,
    mode: Literal["numpy", "torch"],
    device: torch.device,
    store_on_device: bool,
):
    """Tests the replay buffer generically."""
    mem_size = 11
    element_shapes = [(3, 3), (3,), ()]
    memory = ReplayBuffer(
        mem_size=mem_size, mode=mode, device=device, store_on_device=store_on_device
    )

    previous_data = []
    for iteration in range(20):
        current_data = []
        for shape in element_shapes:
            current_data.append(_randn(shape=shape, mode=mode))
        memory.push(current_data, random_rollover=random_rollover)

        # if random rollover and we're more than full, different matching method
        if random_rollover and memory.is_full:
            num_current_matches = 0
            num_previous_matches = 0
            for item in memory:
                num_current_matches += int(_is_equivalent_tuple(item, current_data))
                num_previous_matches += int(_is_equivalent_tuple(item, previous_data))

            assert (
                num_current_matches == 1
            ), f"""Expected 1 match for current_data, got {num_current_matches}."""
            assert (
                num_previous_matches <= 1
            ), f"""Expected 1 or 0 match for previous_data, got {num_previous_matches}."""

            continue

        # check the current data
        output = memory.__getitem__(iteration % mem_size)
        assert _is_equivalent_tuple(
            output, current_data
        ), f"""Something went wrong with rollover at iteration {iteration},
            expected \n{pformat(current_data)}, got \n{pformat(output)}."""

        # check the previous data
        if iteration > 0:
            output = memory.__getitem__((iteration - 1) % mem_size)
            assert _is_equivalent_tuple(
                output, previous_data
            ), f"""Something went wrong with rollover at iteration {iteration},
                expected \n{pformat(previous_data)}, got \n{pformat(output)}."""

        previous_data = deepcopy(current_data)
