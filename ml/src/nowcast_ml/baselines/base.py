"""Common forecaster protocol shared by baselines, the model and evaluation.

Every forecaster maps a :class:`ForecastInput` (the last ``t_in`` frames in
physical units, NaN where unavailable) to :class:`ForecastArrays`, so evaluation
scores all of them with identical code.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from nowcast_ml.data import channels as ch
from nowcast_ml.inference.schema import ForecastArrays


@dataclass
class ForecastInput:
    x: np.ndarray  # (t_in, C, H, W) physical units, NaN where unavailable
    avail: np.ndarray  # (t_in, C, H, W) bool
    channels: list[str]
    past_ltg: np.ndarray  # (H, W) bool: flash within radius in the last 30 min
    grid_spacing_km: float = 2.0
    step_minutes: int = 10

    @classmethod
    def from_sample(
        cls, sample: dict, channels: list[str], grid_spacing_km: float, step_minutes: int = 10
    ):
        return cls(
            x=sample["x"],
            avail=sample["avail"],
            channels=list(channels),
            past_ltg=sample["past_ltg"],
            grid_spacing_km=grid_spacing_km,
            step_minutes=step_minutes,
        )

    def channel(self, name: str) -> np.ndarray | None:
        """(t_in, H, W) values of ``name`` (NaN where unavailable) or None if not in the list."""
        if name not in self.channels:
            return None
        j = self.channels.index(name)
        return np.where(self.avail[:, j], self.x[:, j], np.nan)

    def has_radar(self, min_coverage: float = 0.05) -> bool:
        z = self.channel(ch.TARGET_CHANNEL)
        return z is not None and np.isfinite(z[-1]).mean() >= min_coverage


class Forecaster(Protocol):
    name: str

    def forecast(
        self, inp: ForecastInput, n_leads: int = 12, ltg_leads_min: tuple[int, ...] = (30, 60)
    ) -> ForecastArrays: ...


# ------------------------------------------------------------------ shared helpers
def motion_field(frames: np.ndarray) -> np.ndarray:
    """Lucas-Kanade dense motion (2, H, W) in px/step from (n>=2, H, W) frames.

    Returns zero motion when there are no trackable features (e.g. empty scene).
    """
    from pysteps.motion.lucaskanade import dense_lucaskanade

    f = np.nan_to_num(frames.astype(np.float64), nan=0.0)
    H, W = f.shape[-2:]
    if np.ptp(f) < 1e-6:
        return np.zeros((2, H, W))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            v = dense_lucaskanade(f, verbose=False)
        except Exception:  # noqa: BLE001 - pysteps raises generic errors on degenerate input
            return np.zeros((2, H, W))
    return np.nan_to_num(v)


def tracking_frames(inp: ForecastInput, n: int = 3) -> np.ndarray:
    """Frames used for motion: radar reflectivity if available, else cold cloud tops (TIR1)."""
    if inp.has_radar():
        z = inp.channel(ch.TARGET_CHANNEL)[-n:]
        return np.clip(np.nan_to_num(z, nan=0.0), 0, None)
    bt = inp.channel("tir1_bt")
    if bt is not None and np.isfinite(bt[-1]).any():
        return np.clip(300.0 - np.nan_to_num(bt[-n:], nan=300.0), 0, None)
    H, W = inp.x.shape[-2:]
    return np.zeros((n, H, W))


def advect(
    field: np.ndarray, velocity: np.ndarray, n_steps: int, outval: float = 0.0
) -> np.ndarray:
    from pysteps.extrapolation.semilagrangian import extrapolate

    f = np.nan_to_num(field.astype(np.float64), nan=outval)
    out = extrapolate(f, velocity, n_steps, outval=outval)
    return np.nan_to_num(out, nan=outval)


def advected_lightning(
    past_ltg: np.ndarray, velocity: np.ndarray, ltg_leads_min, step_minutes: int
) -> np.ndarray:
    """ "Lightning persists" advected with the storm motion; max over each lead window."""
    n_max = max(ltg_leads_min) // step_minutes
    fr = advect(past_ltg.astype(np.float64), velocity, n_max)
    return np.stack([fr[: L // step_minutes].max(axis=0) for L in ltg_leads_min]).clip(0, 1)


def last_reflectivity(inp: ForecastInput) -> np.ndarray:
    z = inp.channel(ch.TARGET_CHANNEL)
    H, W = inp.x.shape[-2:]
    return np.full((H, W), np.nan) if z is None else z[-1]
