"""Replay buffer implementation with push, automatic overflow, and automatic torch dataset functionality."""
from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually"
    ) from e

from .print_utils import cstr, wm_print


class ReplayBuffer(Dataset):
    """Replay Buffer implementation of a Torch dataset."""

    def __init__(self, mem_size: int):
        """__init__.

        Args:
            mem_size (int): number of transitions the replay buffer aims to hold
        """
        self.memory = []
        self.mem_size = int(mem_size)
        self.count = 0

    def __len__(self):
        """__len__."""
        return min(self.mem_size, self.count)

    def __getitem__(self, idx):
        """__getitem__.

        Args:
            idx: index of the tuple to get from the replay buffer
        """
        return list(d[idx] for d in self.memory)

    def push(self, data: Sequence[np.ndarray | float | int | bool], bulk: bool = False):
        """Adds transition tuples into the replay buffer.

        The data must be either:
        - an n-long tuple of a single transition
        - an n-long tuple of m transitions, ie: a list of [m, ...] np arrays with the `bulk` flag set to True

        Args:
            data (Sequence[np.ndarray | float | int | bool]): data
            bulk (bool): whether to bulk add stuff into the replay buffer
        """
        # check if we are bulk adding things in and assert lengths
        bulk_size = 1
        if bulk:
            assert all([isinstance(d, np.ndarray) for d in data]), cstr(
                "All things must be np.ndarray for bulk data.", "FAIL"
            )

            bulk_size = data[0].shape[0]  # pyright: ignore

            assert all([len(d) == bulk_size for d in data]), cstr(  # pyright: ignore
                f"All things in data must have same len for the first dimension for bulk data. Received data with {[len(d) for d in data]} items respectively.",  # pyright: ignore
                "FAIL",
            )

        # expand dims of things that only have 1 dim
        def _ensure_dims(thing) -> np.ndarray:
            """Ensures that all arrays are at least [n, ...] and not [n, ].

            Args:
                thing: input

            Returns:
                np.ndarray: output
            """
            thing = np.asarray(thing)
            if len(thing.shape) == int(bulk):
                thing = np.expand_dims(thing, axis=-1)
            return thing

        np_data = list(map(_ensure_dims, data))

        # instantiate the memory if it does not exist
        if self.count == 0:
            self.memory = []
            for thing in np_data:
                if not bulk:
                    self.memory.append(
                        np.zeros((self.mem_size, *thing.shape), dtype=np.float32)
                    )
                else:
                    self.memory.append(
                        np.zeros((self.mem_size, *thing.shape[1:]), dtype=np.float32)
                    )

            mem_size = sum([d.nbytes for d in self.memory])
            wm_print(cstr(f"Replay Buffer Size: {mem_size / 1e9} gigabytes.", "OKCYAN"))

        # assert that the number of lists in memory is same as data to push
        assert len(np_data) == len(self.memory), cstr(
            f"Data length not similar to memory buffer length. Replay buffer has {len(self.memory)} items, but received {len(np_data)} items.",
            "FAIL",
        )

        # put stuff in memory
        i = self.count % self.mem_size
        for memory, thing in zip(self.memory, np_data):
            if not bulk:
                memory[i] = thing
            else:
                # get the remaining space of the replay buffer and
                # whether we need to wrap around
                remaining_space = self.mem_size - i
                front = min(remaining_space, bulk_size)
                back = remaining_space - bulk_size

                # add stuff to the buffer
                memory[i : i + front] = thing[:front]
                if back < 0:
                    memory[: abs(back)] = thing[back:]

        self.count += bulk_size

    @property
    def is_full(self) -> bool:
        """Whether or not the replay buffer has reached capacity.

        Returns:
            bool: whether the buffer is full
        """
        return self.count >= self.mem_size
