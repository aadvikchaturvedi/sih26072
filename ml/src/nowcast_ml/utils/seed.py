"""Deterministic seeding helpers."""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (if importable)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - legacy global RNG for third-party code
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)
