# ruff: noqa: F401
from .wingman import Wingman

try:
    import torch as _

    from .neural_blocks import NeuralBlocks
    from .replay_buffer import ReplayBuffer
except ImportError:
    import warnings

    warnings.warn(
        "Could not import torch, "
        "this is not bundled as part of Wingman and has to be installed manually, "
        "as a result, `NeuralBlocks` and `ReplayBuffer` are unavailable.",
        category=RuntimeWarning,
    )
