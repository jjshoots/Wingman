from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from .print_utils import cstr, wm_print


class ReplayBuffer(Dataset):
    """Replay Buffer implementation of a Torch dataset."""

    def __init__(self, mem_size):
        self.memory = []
        self.mem_size = int(mem_size)
        self.count = 0

    def __len__(self):
        return min(self.mem_size, self.count)

    def __getitem__(self, idx):
        return (d[idx] for d in self.memory)

    def push(self, data: list[np.ndarray | float | int | bool], bulk: bool = False):
        # check if we are bulk adding things in and assert lengths
        bulk_size = 1
        if bulk:
            assert all([isinstance(data, np.ndarray)]), cstr(
                "All things must be np.ndarray for bulk data.", "FAIL"
            )

            bulk_size = data[0].shape[0]  # pyright: ignore

            assert all([len(d) == bulk_size for d in data]), cstr(  # pyright: ignore
                f"All things in data must have same len for the first dimension for bulk data. Received data with {[len(d) for d in data]} items respectively.",  # pyright: ignore
                "FAIL",
            )

        # expand dims of things that only have 1 dim
        def _ensure_dims(thing) -> np.ndarray:
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
        return self.count >= self.mem_size
