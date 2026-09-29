"""Refiner interface: turns the deterministic forecast into ensemble members.

Defined up front so :class:`~nowcast_ml.inference.Predictor` can return
``reflectivity_members`` later without changing its API.
"""

from __future__ import annotations

from typing import Protocol

import torch


class Refiner(Protocol):
    name: str

    def sample(
        self,
        cond: torch.Tensor,
        deterministic: torch.Tensor,
        n_members: int,
        seed: int | None = None,
    ) -> torch.Tensor | None:
        """Return (B, n_members, T_out, H, W) normalized reflectivity members, or None if unsupported.

        ``cond`` is the model input (B, T_in, 2C, H, W); ``deterministic`` is the backbone
        forecast (B, T_out, H, W) in normalized units. Members are ``deterministic + residual``.
        """
        ...
