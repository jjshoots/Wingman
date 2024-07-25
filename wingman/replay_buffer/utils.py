"""Some replay buffer utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np

from wingman.exceptions import ReplayBufferException
from wingman.replay_buffer.flat_replay_buffer import FlatReplayBuffer, _Mode

try:
    import torch
except ImportError as e:
    raise ImportError(
        "Could not import torch, this is not bundled as part of Wingman and has to be installed manually."
    ) from e


def flat_rb_swap_mode(
    replay_buffer: FlatReplayBuffer, mode: Literal["numpy", "torch"]
) -> FlatReplayBuffer:
    """Swaps the mode of a flat replay buffer, useful for pushing and sampling in episodic contexts from numpy to torch.

    Args:
    ----
        replay_buffer (FlatReplayBuffer): replay_buffer
        mode (Literal["numpy", "torch"]): mode

    Returns:
    -------
        FlatReplayBuffer:

    """
    # grab the original mode first
    original_mode = replay_buffer.mode

    # store the mode
    if mode == "numpy":
        replay_buffer.mode = _Mode.NUMPY
        replay_buffer.mode_type = np.ndarray
        replay_buffer.mode_caller = np  # pyright: ignore[reportAttributeAccessIssue]
        replay_buffer.mode_dtype = np.float32
    elif mode == "torch":
        replay_buffer.mode = _Mode.TORCH
        replay_buffer.mode_type = torch.Tensor
        replay_buffer.mode_caller = torch  # pyright: ignore[reportAttributeAccessIssue]
        replay_buffer.mode_dtype = torch.float32
    else:
        raise ReplayBufferException(
            f"Unknown mode {mode}. Only `'numpy'` and `'torch'` are allowed."
        )

    # convert over the memory buffer
    if original_mode == _Mode.NUMPY and replay_buffer.mode == _Mode.TORCH:
        # numpy to torch conversion
        replay_buffer.memory = [
            torch.asarray(
                data,
                dtype=replay_buffer.mode_dtype,  # pyright: ignore[reportArgumentType]
                device=replay_buffer.storage_device,
            )
            for data in replay_buffer.memory
        ]
    elif original_mode == _Mode.TORCH and replay_buffer.mode == _Mode.NUMPY:
        # torch to numpy conversion
        replay_buffer.memory = [data.cpu().numpy() for data in replay_buffer.memory]

    return replay_buffer
