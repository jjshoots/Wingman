"""Tests the replay buffer utilities."""

from __future__ import annotations

import numpy as np
import pytest
from utils import are_equivalent

from wingman.replay_buffer.utils import listed_dict_to_dicted_list


@pytest.mark.parametrize(
    "stack",
    [True, False],
)
def test_dicted_list_to_listed_dict(stack: bool) -> None:
    """Tests the dicted_list_to_listed_dict function."""
    # the input
    listed_dict = [
        {"a": {"x": 1, "y": 2}, "b": [3, 4]},
        {"a": {"x": 5, "y": 6}, "b": [7, 8]},
        {"a": {"x": 9, "y": 10}, "b": [11, 12]},
    ]

    # the target output
    first = np.asarray([1, 5, 9])
    second = np.asarray([2, 6, 10])
    third = np.asarray(
        [
            [3, 4],
            [7, 8],
            [11, 12],
        ]
    )
    if stack:
        first = np.expand_dims(first, axis=0)
        second = np.expand_dims(second, axis=0)
        third = np.expand_dims(third, axis=0)
    target_dicted_list = {
        "a": {
            "x": first,
            "y": second,
        },
        "b": third,
    }

    # convert and check
    created_dicted_list = listed_dict_to_dicted_list(listed_dict, stack=stack)

    assert are_equivalent(
        target_dicted_list, created_dicted_list
    ), f"Expected {target_dicted_list=} to equal {created_dicted_list=}."
