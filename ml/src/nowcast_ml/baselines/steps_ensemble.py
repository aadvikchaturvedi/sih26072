"""pysteps STEPS stochastic ensemble (default 20 members).

Reflectivity is converted to rain rate (Marshall-Palmer Z = 200 R^1.6), log-
transformed to dBR as STEPS expects, forecast, and converted back to dBZ. The
deterministic ``reflectivity`` output is the ensemble mean taken in rain-rate
space (averaging dBZ across displaced members collapses the peaks); members are
returned in ``reflectivity_members``. Lightning uses the advected "persists" field.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np

from nowcast_ml.baselines.base import (
    ForecastInput,
    advected_lightning,
    motion_field,
    tracking_frames,
)
from nowcast_ml.inference.schema import ForecastArrays

R_THR = 0.1  # mm/h
DBR_ZERO = -15.0


def dbz_to_rain(dbz: np.ndarray) -> np.ndarray:
    z = 10.0 ** (np.asarray(dbz, dtype=np.float64) / 10.0)
    r = (z / 200.0) ** (1 / 1.6)
    return np.where(np.nan_to_num(dbz, nan=0.0) > 0, r, 0.0)


def rain_to_dbz(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    with np.errstate(divide="ignore"):
        dbz = 10.0 * np.log10(200.0 * np.power(np.clip(r, 0, None), 1.6))
    return np.where(r >= R_THR, np.clip(dbz, 0, 80), 0.0)


def to_dbr(r):
    with np.errstate(divide="ignore"):
        return np.where(r >= R_THR, 10.0 * np.log10(np.clip(r, R_THR, None)), DBR_ZERO)


def from_dbr(x):
    r = 10.0 ** (np.asarray(x) / 10.0)
    return np.where(r >= R_THR, r, 0.0)


def n_cascade_levels(h: int, w: int) -> int:
    return int(np.clip(int(np.log2(min(h, w))) - 2, 1, 6))


class StepsForecaster:
    name = "steps"

    def __init__(self, n_members: int = 20, seed: int = 0):
        self.n_members = n_members
        self.seed = seed

    def forecast(
        self, inp: ForecastInput, n_leads: int = 12, ltg_leads_min=(30, 60)
    ) -> ForecastArrays:
        from pysteps.nowcasts import steps

        H, W = inp.x.shape[-2:]
        v = motion_field(tracking_frames(inp, 3))
        ltg = advected_lightning(inp.past_ltg, v, ltg_leads_min, inp.step_minutes)
        z = inp.channel("maxz")
        if z is None or not np.isfinite(z[-1]).any():
            nan = np.full((n_leads, H, W), np.nan)
            return ForecastArrays(nan, ltg, np.repeat(nan[None], self.n_members, 0))

        rain = dbz_to_rain(np.nan_to_num(z[-3:], nan=0.0))
        if (
            rain[-1] >= R_THR
        ).mean() < 0.005:  # no precipitation: STEPS is undefined, forecast none
            members = np.zeros((self.n_members, n_leads, H, W))
        else:
            with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                out = steps.forecast(
                    to_dbr(rain),
                    v,
                    n_leads,
                    n_ens_members=self.n_members,
                    n_cascade_levels=n_cascade_levels(H, W),
                    precip_thr=10 * np.log10(R_THR),
                    kmperpixel=inp.grid_spacing_km,
                    timestep=inp.step_minutes,
                    noise_method="nonparametric",
                    vel_pert_method="bps",
                    mask_method="incremental",
                    seed=self.seed,
                )
            rain_members = from_dbr(np.nan_to_num(out, nan=DBR_ZERO))
            members = rain_to_dbz(rain_members)
            return ForecastArrays(rain_to_dbz(rain_members.mean(axis=0)), ltg, members)
        return ForecastArrays(reflectivity=members.mean(axis=0), lightning=ltg, members=members)
