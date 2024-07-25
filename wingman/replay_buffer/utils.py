"""Some replay buffer utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np

from wingman.exceptions import ReplayBufferException
from wingman.replay_buffer.core import ReplayBuffer
from wingman.replay_buffer.flat_replay_buffer import FlatReplayBuffer, _Mode

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
    ) from e


def flat_rb_swap_mode(
    replay_buffer: ReplayBuffer, mode: Literal["numpy", "torch"] | int
) -> ReplayBuffer:
    """Swaps the mode of a replay buffer that has an underlying flat replay buffer.

    Useful for pushing and sampling in episodic contexts from numpy to torch.

    Args:
    ----
        replay_buffer (ReplayBuffer): replay_buffer
        mode (Literal["numpy", "torch"]): mode

    Returns:
    -------
        ReplayBuffer:

    """

    # we want the internal flat replay buffer
    flat_rb_handle = replay_buffer
    while not isinstance(flat_rb_handle, FlatReplayBuffer):
        flat_rb_handle = getattr(flat_rb_handle, "base_buffer", None)
        if flat_rb_handle is None:
            raise ReplayBufferException(
                f"This replay buffer does not have a flat buffer internally."
            )

    # grab the original mode first
    original_mode = flat_rb_handle.mode

    # store the mode
    if mode == "numpy" or mode == _Mode.NUMPY:
        flat_rb_handle.mode = _Mode.NUMPY
        flat_rb_handle.mode_type = np.ndarray
        flat_rb_handle.mode_caller = np  # pyright: ignore[reportAttributeAccessIssue]
        flat_rb_handle.mode_dtype = np.float32
    elif mode == "torch" or mode == _Mode.TORCH:
        flat_rb_handle.mode = _Mode.TORCH
        flat_rb_handle.mode_type = torch.Tensor
        flat_rb_handle.mode_caller = torch  # pyright: ignore[reportAttributeAccessIssue]
        flat_rb_handle.mode_dtype = torch.float32
    else:
        raise ReplayBufferException(
            f"Unknown mode {mode}. Only `'numpy'` and `'torch'` are allowed."
        )

    # convert over the memory buffer
    if original_mode == _Mode.NUMPY and flat_rb_handle.mode == _Mode.TORCH:
        # numpy to torch conversion
        flat_rb_handle.memory = [
            torch.asarray(
                data,
                dtype=flat_rb_handle.mode_dtype,  # pyright: ignore[reportArgumentType]
                device=flat_rb_handle.storage_device,
            )
            for data in flat_rb_handle.memory
        ]
    elif original_mode == _Mode.TORCH and flat_rb_handle.mode == _Mode.NUMPY:
        # torch to numpy conversion
        flat_rb_handle.memory = [data.cpu().numpy() for data in flat_rb_handle.memory]

    return replay_buffer
