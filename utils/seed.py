"""
Reproducibility utilities for EduTune AI.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> int:
    """
    Set random seeds across supported libraries.

    This function configures:
    - Python random
    - NumPy
    - PyTorch CPU
    - PyTorch CUDA, when available

    Returns:
        int: The seed that was applied.
    """
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer.")

    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    return seed


def enable_deterministic_mode() -> None:
    """
    Enable deterministic PyTorch behavior where supported.

    This can reduce performance but improves experiment reproducibility.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def disable_deterministic_mode() -> None:
    """Restore standard PyTorch performance settings."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True