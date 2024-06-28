"""Replay buffer implementation with push, automatic overflow, and automatic torch dataset functionality."""
from __future__ import annotations

from enum import Enum
from typing import Generator, Literal, Sequence

import numpy as np
from prefetch_generator import prefetch

from wingman.exceptions import ReplayBufferException

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
    ) from e

from .print_utils import cstr, wm_print


class _Mode(Enum):
    """_Mode."""

    TORCH = 1
    NUMPY = 2


class ReplayBuffer(Dataset):
    """Replay Buffer implementation of a Torch or Numpy dataset."""

    def __init__(
        self,
        mem_size: int,
        mode: Literal["numpy", "torch"] = "numpy",
        device: torch.device = torch.device("cpu"),
    ):
        """__init__.

        Args:
            mem_size (int): number of transitions the replay buffer aims to hold
        """
        self.memory = []
        self.mem_size = int(mem_size)
        self.count = 0

        # store the device
        self.device = device
        self.cpu_device = torch.device("cpu")

        # store the mode
        if mode == "numpy":
            self.mode = _Mode.NUMPY
            self.mode_type = np.ndarray
            self.mode_caller = np
            self.mode_dtype = np.float32
        elif mode == "torch":
            self.mode = _Mode.TORCH
            self.mode_type = torch.Tensor
            self.mode_caller = torch
            self.mode_dtype = torch.float32
        else:
            raise ReplayBufferException(
                f"Unknown mode {mode}. Only `'numpy'` and `'torch'` are allowed."
            )

    def __len__(self) -> int:
        """The number of memory items this replay buffer is holding.

        Returns:
            int:
        """
        return min(self.mem_size, self.count)

    def __getitem__(self, idx) -> list[np.ndarray | torch.Tensor]:
        return list(d[idx] for d in self.memory)

    def __repr__(self) -> str:
        """Printouts parameters of this replay buffer.

        Returns:
            str:
        """
        return f"""ReplayBuffer of size {self.mem_size} with {len(self.memory)} elements. \n
        Element shapes are {[elem.shape[1:] for elem in self.memory]}. \n
        A brief view of the memory: \n
        {self.memory}
        """

    @property
    def is_full(self) -> bool:
        """Whether or not the replay buffer has reached capacity.

        Returns:
            bool: whether the buffer is full
        """
        return self.count >= self.mem_size

    def _format_data(
        self, thing: np.ndarray | torch.Tensor | float | int | bool, bulk: bool
    ) -> np.ndarray | torch.Tensor:
        """_format_data.

        Args:
            thing (np.ndarray | torch.Tensor | float | int | bool): thing
            bulk (bool): bulk

        Returns:
            np.ndarray | torch.Tensor:
        """
        if self.mode == _Mode.NUMPY:
            # cast to the right dtype
            data = np.asarray(
                thing,
                dtype=self.mode_dtype,  # pyright: ignore[reportGeneralTypeIssues]
            )

            # dim check
            if bulk and len(data.shape) < 1:
                data = np.expand_dims(data, axis=-1)
        elif self.mode == _Mode.TORCH:
            # cast to the right dtype, store on CPU intentionally
            data = torch.asarray(
                thing,
                device=self.cpu_device,
                dtype=self.mode_dtype,  # pyright: ignore[reportGeneralTypeIssues]
            )
            data.requires_grad_(False)

            # dim check
            if bulk and len(data.shape) < 1:
                data = data.unsqueeze(-1)
        else:
            raise ReplayBufferException(
                f"Unknown mode {self.mode}. Only `'numpy'` and `'torch'` are allowed."
            )

        return data

    def push(
        self,
        data: Sequence[torch.Tensor | np.ndarray | float | int | bool],
        bulk: bool = False,
        random_rollover: bool = False,
    ):
        """Adds transition tuples into the replay buffer.

        The data must be either:
        - an n-long tuple of a single transition
        - an n-long tuple of m transitions, ie: a list of [m, ...] np arrays with the `bulk` flag set to True

        Args:
            data (Sequence[torch.Tensor | np.ndarray | float | int | bool]): data
            bulk (bool): whether to bulk add stuff into the replay buffer
            random_rollover (bool): whether to rollover the data in the replay buffer once full or to randomly insert
        """
        # cast to dtype and conditionally add batch dim
        array_data = [self._format_data(item, bulk=True) for item in data]

        if not bulk:
            bulk_size = 1
        else:
            bulk_size = data[0].shape[0]  # pyright: ignore
            # assert all items have same length
            if not all([len(item) == bulk_size for item in array_data]):  # pyright: ignore[reportArgumentType]
                raise ReplayBufferException(
                    "All things in data must have same len for the first dimension for bulk data. "
                    f"Received data with {[len(item) for item in array_data]} items respectively.",
                )

            # assert on memory lengths
            if self.mem_size < bulk_size:
                raise ReplayBufferException(
                    f"Bulk size ({bulk_size}) should be less than or equal to memory size ({self.mem_size}).",
                )

        # instantiate the memory if it does not exist
        if self.count == 0:
            self.memory = []
            if not bulk:
                self.memory.extend(
                    [
                        self.mode_caller.zeros(
                            (self.mem_size, *item.shape),
                            dtype=self.mode_dtype,  # pyright: ignore[reportGeneralTypeIssues]
                        )
                        for item in array_data
                    ]
                )
            else:
                self.memory.extend(
                    [
                        self.mode_caller.zeros(
                            (self.mem_size, *item.shape[1:]),
                            dtype=self.mode_dtype,  # pyright: ignore[reportGeneralTypeIssues]
                        )
                        for item in array_data
                    ]
                )

            mem_size_bytes = sum([d.nbytes for d in self.memory])
            wm_print(
                cstr(f"Replay Buffer Size: {mem_size_bytes / 1e9} gigabytes.", "OKCYAN")
            )

        # assert that the number of lists in memory is same as data to push
        if len(array_data) != len(self.memory):
            raise ReplayBufferException(
                f"Data length not similar to memory buffer length. Replay buffer has {len(self.memory)} items. "
                f"But received {len(array_data)} items.",
            )

        # indexing for memory positions
        start = self.count % self.mem_size
        stop = min(start + bulk_size, self.mem_size)
        rollover = -(self.mem_size - start - bulk_size)
        if random_rollover:
            if not self.is_full:
                idx_front = np.arange(start, stop)
                idk_back = np.random.choice(
                    start, size=np.maximum(rollover, 0), replace=False
                )
            else:
                idx_front = np.random.choice(
                    self.mem_size, size=bulk_size, replace=False
                )
                idk_back = []
        else:
            idx_front = np.arange(start, stop)
            idk_back = np.arange(0, rollover)
        idx = np.append(idx_front, idk_back).astype(np.int64)

        # put things in memory
        for memory, item in zip(self.memory, array_data):
            memory[idx] = item

        self.count += bulk_size

    def sample(self, batch_size: int) -> list[np.ndarray | torch.Tensor]:
        """sample.

        Args:
            batch_size (int): batch_size

        Returns:
            list[np.ndarray | torch.Tensor]:
        """
        idx = np.random.choice(
            len(self), size=np.minimum(len(self), batch_size), replace=False
        )
        if self.mode == _Mode.TORCH:
            return [item[idx].to(self.device) for item in self.memory]
        else:
            return [item[idx] for item in self.memory]

    @prefetch(max_prefetch=1)
    def iter_sample(
        self, batch_size: int, num_iter: int
    ) -> Generator[list[np.ndarray | torch.Tensor], None, None]:
        """iter_sample.

        Args:
            batch_size (int): batch_size
            num_iter (int): num_iter

        Returns:
            Generator[list[np.ndarray | torch.Tensor], None, None]:
        """
        for _ in range(num_iter):
            yield (self.sample(batch_size=batch_size))
