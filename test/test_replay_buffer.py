"""Tests the replay buffer module."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pprint import pformat
from typing import Literal

import pytest
import torch
from utils import (
    are_equivalent,
    create_memory,
    create_shapes,
    element_to_bulk_dim_swap,
    generate_random_dict_data,
    generate_random_flat_data,
)

# define the test configurations
_random_rollovers = [True, False]
_modes = ["numpy", "torch"]
_devices = [torch.device("cpu")]
if torch.cuda.is_available():
    _devices.append(torch.device("cuda:0"))
_store_on_devices = [True, False]
_use_dict = [False, True]
ALL_CONFIGURATIONS = list(
    product(
        _random_rollovers,
        _modes,
        _devices,
        _store_on_devices,
        _use_dict,
    )
)


@pytest.mark.parametrize(
    "random_rollover, mode, device, store_on_device, use_dict",
    ALL_CONFIGURATIONS,
)
def test_bulk(
    random_rollover: bool,
    mode: Literal["numpy", "torch"],
    device: torch.device,
    store_on_device: bool,
    use_dict: bool,
):
    """Tests repeatedly bulking the buffer and whether it rollovers correctly."""
    bulk_size = 7
    mem_size = 11
    shapes = create_shapes(use_dict=use_dict, bulk_size=bulk_size)
    memory = create_memory(
        mem_size=mem_size,
        mode=mode,
        device=device,
        store_on_device=store_on_device,
        random_rollover=random_rollover,
        use_dict=use_dict,
    )

    for iteration in range(10):
        # try to stuff:
        # a) (bulk_size, 3) array
        # b) (bulk_size,) array
        data = []
        for shape in shapes:
            if isinstance(shape, (list, tuple)):
                data.append(
                    generate_random_flat_data(shape=(bulk_size, *shape), mode=mode)
                )
            elif isinstance(shape, dict):
                data.append(generate_random_dict_data(shapes=shape, mode=mode))
            else:
                raise ValueError
        memory.push(data, bulk=True)

        # reverse the data
        # on insertion we had [element_dim, bulk_dim, *data_shapes]
        # on comparison we want [bulk_dim, element_dim, *data_shapes]
        serialized_data = element_to_bulk_dim_swap(
            element_first_data=data,
            bulk_size=bulk_size,
        )

        # if random rollover and we're more than full, different matching technique
        if random_rollover and memory.is_full:
            num_matches = 0
            # match according to meshgrid
            for item1 in serialized_data:
                for item2 in memory:
                    num_matches += int(are_equivalent(item1, item2))

            assert (
                num_matches == bulk_size
            ), f"""Expected {bulk_size} matches inside the memory, got {num_matches}."""

            continue

        for step in range(bulk_size):
            item1 = serialized_data[step]
            item2 = memory[(iteration * bulk_size + step) % mem_size]
            assert are_equivalent(
                item1, item2
            ), f"""Something went wrong with rollover at iteration {iteration},
                step {step}, expected \n{pformat(item1)}, got \n{pformat(item2)}."""


@pytest.mark.parametrize(
    "random_rollover, mode, device, store_on_device, use_dict",
    ALL_CONFIGURATIONS,
)
def test_non_bulk(
    random_rollover: bool,
    mode: Literal["numpy", "torch"],
    device: torch.device,
    store_on_device: bool,
    use_dict: bool,
):
    """Tests the replay buffer generically."""
    mem_size = 11
    shapes = create_shapes(use_dict=use_dict)
    memory = create_memory(
        mem_size=mem_size,
        mode=mode,
        device=device,
        store_on_device=store_on_device,
        random_rollover=random_rollover,
        use_dict=use_dict,
    )

    previous_data = []
    for iteration in range(20):
        current_data = []
        for shape in shapes:
            if isinstance(shape, (list, tuple)):
                current_data.append(generate_random_flat_data(shape=shape, mode=mode))
            elif isinstance(shape, dict):
                current_data.append(generate_random_dict_data(shapes=shape, mode=mode))
            else:
                raise ValueError
        memory.push(current_data)

        # if random rollover and we're more than full, different matching method
        if random_rollover and memory.is_full:
            num_current_matches = 0
            num_previous_matches = 0
            for item in memory:
                num_current_matches += int(are_equivalent(item, current_data))
                num_previous_matches += int(are_equivalent(item, previous_data))

            assert (
                num_current_matches == 1
            ), f"""Expected 1 match for current_data, got {num_current_matches}."""
            assert (
                num_previous_matches <= 1
            ), f"""Expected 1 or 0 match for previous_data, got {num_previous_matches}."""

            continue

        # check the current data
        output = memory.__getitem__(iteration % mem_size)
        assert are_equivalent(
            output, current_data
        ), f"""Something went wrong with rollover at iteration {iteration},
            expected \n{pformat(current_data)}, got \n{pformat(output)}."""

        # check the previous data
        if iteration > 0:
            output = memory[(iteration - 1) % mem_size]
            assert are_equivalent(
                output, previous_data
            ), f"""Something went wrong with rollover at iteration {iteration},
                expected \n{pformat(previous_data)}, got \n{pformat(output)}."""

        previous_data = deepcopy(current_data)
