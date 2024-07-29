"""Base replay buffer class."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Generator, Sequence

from prefetch_generator import prefetch


class ReplayBuffer:
    """Base replay buffer class."""

    def __init__(self, mem_size: int):
        """__init__.

        Args:
        ----
            mem_size (int): mem_size

        """
        self.memory = []
        self.mem_size = int(mem_size)
        self.count = 0

    def __len__(self) -> int:
        """The number of memory items this replay buffer is holding.

        Returns
        -------
            int:

        """
        return min(self.mem_size, self.count)

    def __repr__(self) -> str:
        """Printouts parameters of this replay buffer.

        Returns
        -------
            str:

        """
        return f"""ReplayBuffer of size {self.mem_size} with {len(self.memory)} elements. \n
        A brief view of the memory: \n
        {self.memory}
        """

    @property
    def is_full(self) -> bool:
        """Whether or not the replay buffer has reached capacity.

        Returns
        -------
            bool: whether the buffer is full

        """
        return self.count >= self.mem_size

    @prefetch(max_prefetch=1)
    def iter_sample(
        self, batch_size: int, num_iter: int
    ) -> Generator[Sequence[Any], None, None]:
        """iter_sample.

        Args:
        ----
            batch_size (int): batch_size
            num_iter (int): num_iter

        Returns:
        -------
            Generator[Sequence[np.ndarray | torch.Tensor], None, None]:

        """
        for _ in range(num_iter):
            yield (self.sample(batch_size=batch_size))

    @abstractmethod
    def sample(self, batch_size: int) -> Sequence[Any]:
        """sample.

        Args:
        ----
            batch_size (int): batch_size

        Returns:
        -------
            Sequence[Any]:

        """
        raise NotImplementedError

    @abstractmethod
    def push(
        self,
        data: Sequence[Any],
        bulk: bool = False,
    ) -> None:
        """push.

        Args:
        ----
            data (Sequence[Any]): data
            bulk (bool): bulk

        Returns:
        -------
            None:

        """
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx: int) -> Sequence[Any]:
        """__getitem__.

        Args:
        ----
            idx (int): idx

        Returns:
        -------
            Sequence[Any]:

        """
        raise NotImplementedError


class ReplayBufferWrapper(ReplayBuffer):
    """ReplayBufferWrapper."""

    def __init__(self, base_buffer: ReplayBuffer):
        """__init__.

        Args:
        ----
            base_buffer (ReplayBuffer): base_buffer

        """
        self.base_buffer = base_buffer

    def __len__(self) -> int:
        """The number of memory items this replay buffer is holding.

        Returns
        -------
            int:

        """
        return len(self.base_buffer)

    def __repr__(self) -> str:
        """Printouts parameters of this replay buffer.

        Returns
        -------
            str:

        """
        return f"""ReplayBuffer of size {self.mem_size} with {len(self.memory)} elements. \n
        A brief view of the memory: \n
        {self.base_buffer}
        """

    def __getitem__(self, idx: int) -> Sequence[Any]:
        """__getitem__.

        Args:
        ----
            idx (int): idx

        Returns:
        -------
            Sequence[Any]:

        """
        return self.wrap_data(self.base_buffer[idx])

    @property
    def is_full(self) -> bool:
        """Whether or not the replay buffer has reached capacity.

        Returns
        -------
            bool: whether the buffer is full

        """
        return self.base_buffer.is_full

    def push(
        self,
        data: Sequence[Any],
        bulk: bool = False,
    ) -> None:
        """push.

        Args:
        ----
            data (Sequence[Any]): data
            bulk (bool): bulk

        Returns:
        -------
            None:

        """
        self.base_buffer.push(
            data=self.unwrap_data(data, bulk),
            bulk=bulk,
        )

    def sample(self, batch_size: int) -> Sequence[Any]:
        """sample.

        Args:
        ----
            batch_size (int): batch_size

        Returns:
        -------
            Sequence[Any]:

        """
        return self.wrap_data(self.base_buffer.sample(batch_size=batch_size))

    @abstractmethod
    def unwrap_data(self, wrapped_data: Sequence[Any], bulk: bool) -> Sequence[Any]:
        """Unwraps data from the underlying data into an unwrapped format.

        This is called when packing the data into the `base_buffer`.

        Args:
        ----
            wrapped_data (Sequence[Any]): wrapped_data
            bulk (bool): bulk

        Returns:
        -------
            Sequence[Any]:

        """
        raise NotImplementedError

    @abstractmethod
    def wrap_data(self, unwrapped_data: Sequence[Any]) -> Sequence[Any]:
        """Wraps data from the underlying data into a wrapped format.

        This is called when sampling data from `base_buffer`.

        Args:
        ----
            unwrapped_data (Sequence[Any]): unwrapped_data

        Returns:
        -------
            Sequence[Any]:

        """
        raise NotImplementedError
