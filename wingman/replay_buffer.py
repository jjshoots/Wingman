import numpy as np
from torch.utils.data import Dataset


class ReplayBuffer(Dataset):
    """Replay Buffer implementation of a Torch dataset."""

    def __init__(self, mem_size):
        self.memory = []
        self.mem_size = int(mem_size)
        self.count = 0

    def __len__(self):
        return min(self.mem_size, self.count)

    def __getitem__(self, idx):
        data = []
        for item in self.memory:
            data.append(item[idx])

        return (*data,)

    def push(self, data, bulk=False):
        # check if we are bulk adding things in
        bulk_size = 1
        if bulk:
            bulk_size = data[0].shape[0]
            for thing in data:
                if len(thing) != bulk_size:
                    raise AssertionError(
                        "All things in data must have same len for the first dimension."
                    )

        # expand dims of things that only have 1 dim
        def _ensure_dims(thing):
            thing = np.array(thing)
            if len(thing.shape) == 0 + (bulk * 1):
                thing = np.expand_dims(thing, axis=-1)
            return thing

        data = list(map(_ensure_dims, data))

        # instantiate the memory if it does not exist
        if self.count == 0:
            self.memory = []
            for thing in data:
                if not bulk:
                    self.memory.append(
                        np.zeros((self.mem_size, *thing.shape), dtype=np.float32)
                    )
                else:
                    self.memory.append(
                        np.zeros((self.mem_size, *thing.shape[1:]), dtype=np.float32)
                    )

            mem_size = 0
            for thing in self.memory:
                mem_size += thing.nbytes
            print(f"Replay Buffer Size: {mem_size / 1e9} gigabytes.")

        # assert that the number of lists in memory is same as data to push
        assert len(data) == len(
            self.memory
        ), "data length not similar to memory buffer length"

        # put stuff in memory
        i = self.count % self.mem_size
        for memory, thing in zip(self.memory, data):
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
    def is_full(self):
        return self.count >= self.mem_size
