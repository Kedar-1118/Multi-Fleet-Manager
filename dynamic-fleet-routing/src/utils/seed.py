"""Global random seed management for reproducibility.

Seeds Python, NumPy, PyTorch, and Gymnasium to ensure
reproducible experiments across all random sources.
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_global_seed(seed: int) -> None:
    """Set random seeds across all libraries for reproducibility.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed


def get_rng(seed: Optional[int] = None) -> np.random.RandomState:
    """Create a new NumPy RandomState with an optional seed.

    Args:
        seed: Optional seed. Uses random seed if None.

    Returns:
        A NumPy RandomState instance.
    """
    if seed is not None:
        return np.random.RandomState(seed)
    return np.random.RandomState()
