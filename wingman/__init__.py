"""The main Wingman package."""

# ruff: noqa: F401
from .wingman import Wingman as Wingman

try:
    import torch as _

    from .neural_blocks import NeuralBlocks as NeuralBlocks
except ImportError:
    import warnings

    warnings.warn(
        "Could not import torch, "
        "this is not bundled as part of Wingman and has to be installed manually.",
        category=RuntimeWarning,
    )
