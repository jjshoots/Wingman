"""Tests the replay buffer module."""
from copy import deepcopy

import numpy as np
import pytest

from wingman.replay_buffer import ReplayBuffer


def is_equivalent_tuple(item1, item2):
    """Checks whether a two tuples of np.ndarrays are equivalent to each other."""
    equivalence = True
    for i1, i2 in zip(item1, item2):
        equivalence = (i1 == i2).all() and equivalence
    return equivalence


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

        # reverse the data to make indexing for checking easier
        reversed_data = list(map(list, zip(*data)))

        # if random rollover and we're more than full, different matching technique
        if random_rollover and memory.is_full:
            num_matches = 0
            # match according to meshgrid
            for item1 in reversed_data:
                for item2 in memory:
                    num_matches += int(is_equivalent_tuple(item1, item2))

            assert (
                num_matches == bulk_size
            ), f"""Expected {bulk_size} matches inside the memory, got {num_matches}."""

            continue

        for step in range(bulk_size):
            item1 = reversed_data[step]
            item2 = memory.__getitem__((iteration * bulk_size + step) % mem_size)
            assert is_equivalent_tuple(
                item1, item2
            ), f"""Something went wrong with rollover at iteration {iteration},
                step {step}, expected {item1}, got {item2}."""


@pytest.mark.parametrize("random_rollover", [True, False])
def test_non_bulk(random_rollover):
    """Tests the replay buffer generically."""
    mem_size = 11
    element_shapes = [(3, 3), (3,), ()]
    memory = ReplayBuffer(mem_size=mem_size)

    previous_data = []
    for iteration in range(20):
        current_data = []
        for shape in element_shapes:
            current_data.append(np.random.randn(*shape))
        memory.push(current_data, random_rollover=random_rollover)

        # if random rollover and we're more than full, different matching method
        if random_rollover and memory.is_full:
            num_current_matches = 0
            num_previous_matches = 0
            for item in memory:
                num_current_matches += int(is_equivalent_tuple(item, current_data))
                num_previous_matches += int(is_equivalent_tuple(item, previous_data))

            assert (
                num_current_matches == 1
            ), f"""Expected 1 match for current_data, got {num_current_matches}."""
            assert (
                num_previous_matches <= 1
            ), f"""Expected 1 or 0 match for previous_data, got {num_previous_matches}."""

            continue

        # check the current data
        output = memory.__getitem__(iteration % mem_size)
        assert is_equivalent_tuple(
            output, current_data
        ), f"""Something went wrong with rollover at iteration {iteration},
            expected {current_data}, got {output}."""

        # check the previous data
        if iteration > 0:
            output = memory.__getitem__((iteration - 1) % mem_size)
            assert is_equivalent_tuple(
                output, previous_data
            ), f"""Something went wrong with rollover at iteration {iteration},
                expected {previous_data}, got {output}."""

        previous_data = deepcopy(current_data)
