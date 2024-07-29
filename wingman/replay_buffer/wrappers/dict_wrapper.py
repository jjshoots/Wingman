"""Wrapper to convert a FlatReplayBuffer into one that accepts nested dicts."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Union

import numpy as np

from wingman.exceptions import ReplayBufferException
from wingman.replay_buffer.core import ReplayBufferWrapper
from wingman.replay_buffer.flat_replay_buffer import FlatReplayBuffer

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
    ) from e

_NestedDict = Mapping[str, Union[int, "_NestedDict"]]


class DictReplayBufferWrapper(ReplayBufferWrapper):
    """Replay Buffer Wrapper that allows the underlying replay buffer to take in nested dicts."""

    def __init__(self, replay_buffer: FlatReplayBuffer) -> None:
        """__init__.

        If bulk adding items, this expects dictionary items to be a dictionary of lists, NOT a list of dictionaries.

        Args:
        ----
            self:
            replay_buffer (FlatReplayBuffer): replay_buffer

        Returns:
        -------
            None:

        """
        super().__init__(replay_buffer)

        # this is a list where:
        # - if the element is an integer, specifies the location in the unwrapped flat data that the wrapped element in this index should be
        # - if it's a dictionary, specifies that the data at this index in the wrapped data should be a dictionary, where
        #   - if the value is an int, is the location of this data in the flat dict
        #   - if the value is a dict, means that this data holds a nested dict
        self.mapping: list[int | _NestedDict] = []
        self.total_elements = 0

    @staticmethod
    def _recursive_unpack_dict_mapping(
        data_dict: dict[str, np.ndarray | torch.Tensor],
        start_idx: int,
    ) -> tuple[_NestedDict, int]:
        """Recursively unpacks a dictionary into a mapping.

        Args:
        ----
            data_dict (dict[str, np.ndarray | torch.Tensor]): data_dict
            start_idx (int): start_idx

        Returns:
        -------
            tuple[_NestedDict, int]:

        """
        mapping: _NestedDict = dict()
        idx = start_idx

        for key, value in data_dict.items():
            if isinstance(value, dict):
                mapping[key], idx = (
                    DictReplayBufferWrapper._recursive_unpack_dict_mapping(
                        value, start_idx=idx
                    )
                )
            else:
                mapping[key] = idx
                idx += 1

        return mapping, idx

    @staticmethod
    def _generate_mapping(
        wrapped_data: Sequence[
            dict[str, np.ndarray | torch.Tensor]
            | np.ndarray
            | torch.Tensor
            | float
            | int
            | bool
        ],
    ) -> tuple[list[int | _NestedDict], int]:
        """Generates a mapping from wrapped data.

        For example:
            data = [
                32,
                65,
                {
                    "a": 5,
                    "b": {
                        "c": 6,
                        "d": 7,
                        "e": {
                            "f": 8
                        }
                    },
                    "g": {
                        "h": 9
                    }
                },
                100,
            ]

            becomes:

            [
                0,
                1,
                {
                    'a': 2,
                    'b': {
                        'c': 3,
                        'd': 4,
                        'e': {'f': 5}
                    },
                    'g': {'h': 6}
                },
                7
            ]

        Args:
        ----
            wrapped_data (Sequence[dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | float | int | bool]): wrapped_data

        Returns:
        -------
            tuple[list[int | _NestedDict], int]:

        """
        mapping: list[int | _NestedDict] = []
        idx = 0

        for item in wrapped_data:
            if isinstance(item, dict):
                dict_mapping, idx = (
                    DictReplayBufferWrapper._recursive_unpack_dict_mapping(
                        item, start_idx=idx
                    )
                )
                mapping.append(dict_mapping)
            else:
                mapping.append(idx)
                idx += 1

        return mapping, idx

    @staticmethod
    def _recursive_unpack_dict_data(
        data_dict: dict[str, np.ndarray | torch.Tensor],
        unwrapped_data_target: list[Any],
        mapping: _NestedDict,
    ) -> list[Any]:
        """Recursively unpacks dictionary data into a sequence of items that FlatReplayBuffer can use.

        Args:
        ----
            data_dict (dict[str, np.ndarray | torch.Tensor]): data_dict
            unwrapped_data_target (list[Any]): unwrapped_data_target
            mapping (_NestedDict): mapping

        Returns:
        -------
            list[Any]:

        """
        for key, value in data_dict.items():
            if isinstance((idx_map := mapping[key]), int):
                unwrapped_data_target[idx_map] = value

            elif isinstance((idx_map := mapping[key]), dict):
                if not isinstance(value, dict):
                    raise ReplayBufferException(
                        "Something went wrong with data unwrapping.\n"
                        f"Expected a dictionary for key {key} within the data, but got {type(value)}."
                    )

                unwrapped_data_target = (
                    DictReplayBufferWrapper._recursive_unpack_dict_data(
                        data_dict=value,
                        unwrapped_data_target=unwrapped_data_target,
                        mapping=idx_map,
                    )
                )
            else:
                raise ValueError("Not supposed to be here")

        return unwrapped_data_target

    def unwrap_data(
        self,
        wrapped_data: Sequence[
            Sequence[dict[str, Any] | np.ndarray | torch.Tensor | float | int | bool]
            | dict[str, Any]
            | np.ndarray
            | torch.Tensor
            | float
            | int
            | bool
        ],
        bulk: bool,
    ) -> Sequence[np.ndarray | torch.Tensor | float | int | bool]:
        """Unwraps dictionary data into a sequence of items that FlatReplayBuffer can use.

        If bulk adding items, this expects dictionary items to be a dictionary of lists, NOT a list of dictionaries.

        Args:
        ----
            wrapped_data (Sequence[dict[str, Any] | np.ndarray | torch.Tensor | float | int | bool]): wrapped_data
            bulk (bool): bulk

        Returns:
        -------
            Sequence[np.ndarray | torch.Tensor | float | int | bool]:

        """
        if len(self) == 0:
            self.mapping, self.total_elements = self._generate_mapping(
                wrapped_data=wrapped_data
            )

        if len(self.mapping) != len(wrapped_data):
            raise ReplayBufferException(
                "Something went wrong with data unwrapping.\n"
                f"Expected `wrapped_data` to have {len(self.mapping)} items, but got {len(wrapped_data)}."
            )

        # holder for the unwrapped data
        unwrapped_data: list[Any] = [None] * self.total_elements

        for i, (mapping, data_item) in enumerate(zip(self.mapping, wrapped_data)):
            # if the mapping says it's an int, then just set the data without nesting
            if isinstance(mapping, int):
                unwrapped_data[mapping] = data_item

            # if it's a dict, then we need to recursively unpack
            elif isinstance(mapping, dict):
                if not isinstance(data_item, dict):
                    raise ReplayBufferException(
                        "Something went wrong with data unwrapping.\n"
                        f"Expected `wrapped_data` at element {i} to be a dict, but got {type(data_item)}."
                    )

                unwrapped_data = self._recursive_unpack_dict_data(
                    data_dict=data_item,
                    unwrapped_data_target=unwrapped_data,
                    mapping=mapping,
                )

        return unwrapped_data

    @staticmethod
    def _recursive_pack_dict_data(
        unwrapped_data: Sequence[Any],
        mapping: _NestedDict,
    ) -> dict[str, Any]:
        """Packs back a sequence of items into a dictionary structure according to a mapping.

        Args:
        ----
            unwrapped_data (Sequence[Any]): unwrapped_data
            mapping (_NestedDict): mapping

        Returns:
        -------
            dict[str, Any]:

        """
        data_dict = dict()
        for key, idx_map in mapping.items():
            if isinstance(idx_map, int):
                data_dict[key] = unwrapped_data[idx_map]
            elif isinstance(idx_map, dict):
                data_dict[key] = DictReplayBufferWrapper._recursive_pack_dict_data(
                    unwrapped_data=unwrapped_data,
                    mapping=idx_map,
                )
            else:
                raise ValueError("Not supposed to be here")

        return data_dict

    def wrap_data(
        self, unwrapped_data: Sequence[np.ndarray | torch.Tensor | float | int | bool]
    ) -> Sequence[
        dict[str, np.ndarray | torch.Tensor]
        | np.ndarray
        | torch.Tensor
        | float
        | int
        | bool
    ]:
        """Converts a sequence of items into a dictionary structure that is similar to that used during `push`.

        Args:
        ----
            self:
            unwrapped_data (Sequence[np.ndarray | torch.Tensor | float | int | bool]): unwrapped_data

        Returns:
        -------
            Sequence[dict[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor | float | int | bool]:

        """
        wrapped_data: list[Any] = [None] * len(self.mapping)

        for i, idx_map in enumerate(self.mapping):
            if isinstance(idx_map, int):
                wrapped_data[i] = unwrapped_data[idx_map]
            elif isinstance(idx_map, dict):
                wrapped_data[i] = DictReplayBufferWrapper._recursive_pack_dict_data(
                    unwrapped_data=unwrapped_data,
                    mapping=idx_map,
                )
            else:
                raise ValueError("Not supposed to be here")

        return wrapped_data
