"""Dict Replay buffer implementation with push, automatic overflow, and automatic torch dataset functionality."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from wingman.exceptions import ReplayBufferException
from wingman.replay_buffer import ReplayBuffer

try:
    import torch
    from torch.utils.data import Dataset
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
    ) from e


class DictReplayBuffer(ReplayBuffer):
    """Replay Buffer implementation that also allows taking in nested dicts.

    Only dictionaries with depth of 1 are accepted at the moment.
    """

    def __init__(
        self,
        mem_size: int,
        mode: Literal["numpy", "torch"] = "numpy",
        device: torch.device = torch.device("cpu"),
        store_on_device: bool = False,
        random_rollover: bool = False,
    ) -> None:
        super().__init__(
            mem_size=mem_size,
            mode=mode,
            device=device,
            store_on_device=store_on_device,
            random_rollover=random_rollover,
        )

        # these are n-long lists of:
        # a) if int, this corresponds to which element in the memory this item comes from
        # b) if dict, this is either a key2idx or idx2key mapping
        self.key_to_idx: list[int | dict[str, int]] = []
        self.idx_to_key: list[int | dict[int, str]] = []

    def push(
        self,
        data: Sequence[
            dict[str, torch.Tensor | np.ndarray | float | int | bool]
            | torch.Tensor
            | np.ndarray
            | float
            | int
            | bool
        ],
        bulk: bool = False,
    ):
        if self.count == 0:
            flattened_data: list[torch.Tensor | np.ndarray | float | int | bool] = []
        else:
            # placeholder
            flattened_data = [1] * len(self.memory)

        # iterate through each item
        for item in data:
            # if non-dict, treat like normal
            if isinstance(item, (torch.Tensor, np.ndarray, float, int, bool)):
                flattened_data.append(item)
                # if the count is zero, start recording
                if self.count == 0:
                    self.key_to_idx.append(len(flattened_data))
                    self.idx_to_key.append(len(flattened_data))

            # if dict, then unpack flatten
            elif isinstance(item, dict):
                for i, (key, value) in enumerate(item.items()):
                    # if count is zero, do the usual thing
                    if self.count == 0:
                        flattened_data.append(value)
                        self.key_to_idx.append({key: len(flattened_data)})
                        self.idx_to_key.append({len(flattened_data): key})

                    # if the count is not zero, we need to unpack carefully
                    else:
                        # first, get the supposed mapping out
                        mapping = self.key_to_idx[i]

                        # make sure that the mapping is a dict
                        if not isinstance(mapping, dict):
                            raise ReplayBufferException(
                                f"Did not expect a dict in position {i} of the inserted data."
                            )

                        # insert the data into the position specified by the mapping
                        flattened_data[mapping[key]] = value

            else:
                raise NotImplementedError(
                    f"Don't know how to deal with type {type(item)} with data:\n{item}."
                )

        # insert the flattened data normally
        super().push(flattened_data, bulk=bulk)

    def sample(
        self, batch_size: int
    ) -> Sequence[
        dict[str, torch.Tensor | np.ndarray | float | int | bool]
        | torch.Tensor
        | np.ndarray
        | float
        | int
        | bool
    ]:
        # sample some flattened data
        flattened_data = super().sample(batch_size=batch_size)

        # unflatten the data
        data: list[
            dict[str, torch.Tensor | np.ndarray | float | int | bool]
            | torch.Tensor
            | np.ndarray
            | float
            | int
            | bool
        ] = []

        for mapping in self.idx_to_key:
            # if int, normal unpacking
            if isinstance(mapping, int):
                data.append(flattened_data[mapping])

            # if dict, then...
            elif isinstance(mapping, dict):
                # iterate through the mapping and reconstruct the dictionary at the location
                data_dict = dict()
                for idx, key in mapping.items():
                    data_dict[key] = flattened_data[idx]
                data.append(data_dict)

        return data
