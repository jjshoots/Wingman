"""Tests the replay buffer module."""
from copy import deepcopy

import numpy as np
import pytest

from wingman import ReplayBuffer


@pytest.mark.parametrize("random_rollover", [True, False])
def test_bulk(random_rollover):
    """Tests repeatedly bulking the buffer and whether it rollovers correctly."""
    bulk_size = 7
    mem_size = 11
    element_shapes = [(3, 3), (3,), ()]
    memory = ReplayBuffer(mem_size=mem_size)

    for iteration in range(10):
        # try to stuff:
        # a) (bulk_size, 3) array
        # b) (bulk_size,) array
        data = []
        for shape in element_shapes:
            data.append(np.random.randn(bulk_size, *shape))
        memory.push(data, bulk=True, random_rollover=random_rollover)

        # if random rollover and we're more than full, don't care anymore
        if random_rollover and memory.is_full:
            return

        for step in range(bulk_size):
            output = memory.__getitem__((iteration * bulk_size + step) % mem_size)
            for idx in range(len(data)):
                assert (
                    output[idx] == data[idx][step]
                ).all(), f"""Something went wrong with rollover at iteration {iteration},
                step {step}, expected {data[idx][step]}, got {output[idx]}."""


@pytest.mark.parametrize("random_rollover", [True, False])
def test_non_bulk(random_rollover):
    """Tests the replay buffer generically."""
    mem_size = 11
    element_shapes = [(3, 3), (3,), ()]
    memory = ReplayBuffer(mem_size=mem_size)

    previous_data = []
    for iteration in range(50):
        current_data = []
        for shape in element_shapes:
            current_data.append(np.random.randn(*shape))
        memory.push(current_data, random_rollover=random_rollover)

        # if random rollover and we're more than full, don't care anymore
        if random_rollover and memory.is_full:
            return

        # check the current data
        output = memory.__getitem__(iteration % mem_size)
        for idx in range(len(current_data)):
            assert (
                output[idx] == current_data[idx]
            ).all(), f"""Something went wrong with rollover at iteration {iteration},
            expected {current_data[idx]}, got {output[idx]}."""

        # check the previous data
        if iteration > 0:
            output = memory.__getitem__((iteration - 1) % mem_size)
            for idx in range(len(current_data)):
                assert (
                    output[idx] == previous_data[idx]
                ).all(), f"""Something went wrong with rollover at iteration {iteration},
                expected {previous_data[idx]}, got {output[idx]}."""

        previous_data = deepcopy(current_data)
