"""Lagrangian persistence: Lucas-Kanade optical flow + semi-Lagrangian advection (pysteps)."""

from __future__ import annotations

import numpy as np

from nowcast_ml.baselines.base import (
    ForecastInput,
    advect,
    advected_lightning,
    last_reflectivity,
    motion_field,
    tracking_frames,
)
from nowcast_ml.inference.schema import ForecastArrays


class ExtrapolationForecaster:
    """Areas advected in from outside the domain are filled with 0 dBZ (no echo)."""

    name = "extrapolation"

    def __init__(self, n_motion_frames: int = 3):
        self.n_motion_frames = n_motion_frames

    def forecast(
        self, inp: ForecastInput, n_leads: int = 12, ltg_leads_min=(30, 60)
    ) -> ForecastArrays:
        v = motion_field(tracking_frames(inp, self.n_motion_frames))
        z = last_reflectivity(inp)
        if np.isfinite(z).any():
            refl = advect(np.clip(np.nan_to_num(z, nan=0.0), 0, None), v, n_leads)
        else:
            refl = np.full((n_leads, *z.shape), np.nan)
        ltg = advected_lightning(inp.past_ltg, v, ltg_leads_min, inp.step_minutes)
        return ForecastArrays(reflectivity=refl, lightning=ltg)
