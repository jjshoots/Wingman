"""Replay buffer implementation with push, automatic overflow, and automatic torch dataset functionality."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
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

    def __repr__(self):
        """Printouts parameters of this replay buffer."""
        return f"""ReplayBuffer of size {self.mem_size} with {len(self.memory)} elements. \n
        Element shapes are {[elem.shape[1:] for elem in self.memory]}. \n
        A brief view of the memory: \n
        {self.memory}
        """

    @staticmethod
    def __ensure_dims(thing: Any, bulk: bool) -> np.ndarray:
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

    def push(
        self,
        data: Sequence[np.ndarray | float | int | bool],
        bulk: bool = False,
        random_rollover: bool = False,
    ):
        """Adds transition tuples into the replay buffer.

        The data must be either:
        - an n-long tuple of a single transition
        - an n-long tuple of m transitions, ie: a list of [m, ...] np arrays with the `bulk` flag set to True

        Args:
            data (Sequence[np.ndarray | float | int | bool]): data
            bulk (bool): whether to bulk add stuff into the replay buffer
            random_rollover (bool): whether to rollover the data in the replay buffer once full or to randomly insert
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

            # assert on memory lengths
            assert self.mem_size >= bulk_size, cstr(
                f"Bulk size ({bulk_size}) should be less than or equal to memory size ({self.mem_size}).",
                "FAIL",
            )

        np_data = list(map(lambda x: self.__ensure_dims(x, bulk), data))

        # instantiate the memory if it does not exist
        if self.count == 0:
            self.memory = []
            for thing in np_data:
                if not bulk:
                    self.memory.append(
                        np.zeros((self.mem_size, *thing.shape), dtype=np.float64)
                    )
                else:
                    self.memory.append(
                        np.zeros((self.mem_size, *thing.shape[1:]), dtype=np.float64)
                    )

            mem_size_bytes = sum([d.nbytes for d in self.memory])
            wm_print(
                cstr(f"Replay Buffer Size: {mem_size_bytes / 1e9} gigabytes.", "OKCYAN")
            )

        # assert that the number of lists in memory is same as data to push
        assert len(np_data) == len(self.memory), cstr(
            f"Data length not similar to memory buffer length. Replay buffer has {len(self.memory)} items, but received {len(np_data)} items.",
            "FAIL",
        )

        # indexing for memory positions
        start = self.count % self.mem_size
        stop = min(start + bulk_size, self.mem_size)
        rollover = -(self.mem_size - start - bulk_size)
        if random_rollover:
            if not self.is_full:
                idx_start = np.arange(start, stop)
                idx_stop = np.random.choice(
                    start, size=np.maximum(rollover, 0), replace=False
                )
            else:
                idx_start = np.random.choice(
                    self.mem_size, size=bulk_size, replace=False
                )
                idx_stop = []
        else:
            idx_start = np.arange(start, stop)
            idx_stop = np.arange(0, rollover)
        indices = np.append(idx_start, idx_stop).astype(np.int64)

        # put things in memory
        for memory, thing in zip(self.memory, np_data):
            memory[indices] = thing

        self.count += bulk_size

    @property
    def is_full(self) -> bool:
        """Whether or not the replay buffer has reached capacity.

        Returns:
            bool: whether the buffer is full
        """
        return self.count >= self.mem_size
