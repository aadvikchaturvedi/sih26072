"""Synthetic convective events for tests and smoke training.

Storms are Gaussian reflectivity cores that move with a common steering flow
(plus a small per-storm perturbation), grow, peak and decay. Satellite,
lightning and NWP channels are derived from the same storms with plausible
physics so every channel carries signal:

* cloud-top TIR1 cools *before* reflectivity peaks (anvil leads the core),
* flashes occur where reflectivity exceeds ~40 dBZ,
* CAPE is elevated in the storm environment.

This is **not** a meteorological simulator; it exists to exercise the code
paths and let tiny models learn advection in seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from nowcast_ml.data import channels as ch

LAT0, LON0 = 22.0, 78.0  # centre of the synthetic domain
KM_PER_DEG = 111.0


@dataclass
class SyntheticEvent:
    event_id: str
    times: pd.DatetimeIndex
    fields: dict[str, np.ndarray]  # name -> (T, H, W) float32, physical units
    missing: dict[str, np.ndarray]  # group -> (T, H, W) bool, True = missing
    flashes: np.ndarray  # (N, 3) float: time index, y (px), x (px)
    lat: np.ndarray  # (H, W)
    lon: np.ndarray  # (H, W)
    grid_spacing_km: float
    motion_px: tuple[float, float] = (0.0, 0.0)  # steering flow (dy, dx) per frame
    meta: dict = field(default_factory=dict)


def _latlon(h: int, w: int, dx_km: float) -> tuple[np.ndarray, np.ndarray]:
    dlat = dx_km / KM_PER_DEG
    dlon = dx_km / (KM_PER_DEG * np.cos(np.deg2rad(LAT0)))
    lat = LAT0 + (np.arange(h) - h / 2) * dlat
    lon = LON0 + (np.arange(w) - w / 2) * dlon
    lon2, lat2 = np.meshgrid(lon, lat)
    # Row 0 is the northern edge (image order), as in the input contract.
    return lat2[::-1].astype(np.float64), lon2.astype(np.float64)


def _smooth_field(r: np.random.Generator, h: int, w: int, scale: float) -> np.ndarray:
    """Cheap smooth random field in [0, 1] via low-frequency cosines."""
    yy, xx = np.mgrid[0:h, 0:w]
    out = np.zeros((h, w))
    for _ in range(3):
        ky, kx = r.uniform(0.2, 1.0, 2) * 2 * np.pi / scale
        ph = r.uniform(0, 2 * np.pi)
        out += np.cos(ky * yy + kx * xx + ph)
    out = (out - out.min()) / (np.ptp(out) + 1e-9)
    return out


def generate_event(
    seed: int,
    size: int = 64,
    n_frames: int = 30,
    n_storms: tuple[int, int] = (1, 4),
    speed_px: tuple[float, float] = (0.5, 2.5),
    grid_spacing_km: float = 2.0,
    radar_missing_prob: float = 0.0,
    start: str = "2026-05-12T06:00",
    step_minutes: int = 10,
    event_id: str | None = None,
    motion: tuple[float, float] | None = None,
) -> SyntheticEvent:
    """Generate one synthetic event. Deterministic for a given ``seed``."""
    r = np.random.default_rng(seed)
    h = w = size
    T = n_frames
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    # Steering flow shared by all storms.
    if motion is None:
        speed = r.uniform(*speed_px)
        ang = r.uniform(0, 2 * np.pi)
        vy, vx = speed * np.sin(ang), speed * np.cos(ang)
    else:
        vy, vx = motion

    k = int(r.integers(n_storms[0], n_storms[1] + 1))
    refl = np.zeros((T, h, w))
    anvil = np.zeros((T, h, w))
    for _ in range(k):
        # Start upstream so storms cross the domain.
        y0 = r.uniform(0.2, 0.8) * h - vy * T / 2
        x0 = r.uniform(0.2, 0.8) * w - vx * T / 2
        dvy, dvx = r.normal(0, 0.15, 2)
        peak_dbz = r.uniform(42, 62)
        t_peak = r.uniform(0.25, 0.75) * T
        life = r.uniform(0.25, 0.5) * T
        sigma0 = r.uniform(2.0, 4.0) * (size / 64)
        for t in range(T):
            cy = y0 + (vy + dvy) * t
            cx = x0 + (vx + dvx) * t
            env = np.exp(-0.5 * ((t - t_peak) / life) ** 2)
            env_anvil = np.exp(-0.5 * ((t + 2 - t_peak) / life) ** 2)  # anvil leads core
            sig = sigma0 * (0.7 + 0.6 * env)
            d2 = (yy - cy) ** 2 + (xx - cx) ** 2
            refl[t] = np.maximum(refl[t], peak_dbz * env * np.exp(-d2 / (2 * sig**2)))
            anvil[t] = np.maximum(anvil[t], env_anvil * np.exp(-d2 / (2 * (1.8 * sig) ** 2)))

    noise = r.normal(0, 0.8, refl.shape)
    maxz = np.where(refl > 5.0, refl + noise, 0.0)
    maxz = np.clip(maxz, 0.0, 75.0)
    cappi = np.clip(np.where(maxz > 0, maxz - 3.0, 0.0), 0.0, 75.0)

    tir1 = 295.0 - 95.0 * anvil + r.normal(0, 0.5, anvil.shape)
    tir2 = tir1 - 1.5
    wv = 240.0 - 35.0 * anvil + r.normal(0, 0.3, anvil.shape)
    mir = tir1 + 6.0 * (1.0 - anvil)
    cooling = np.zeros_like(tir1)
    cooling[1:] = tir1[1:] - tir1[:-1]
    tir1_minus_wv = tir1 - wv

    # Lightning: Poisson flashes where reflectivity > 40 dBZ.
    px_area = grid_spacing_km**2
    rate = np.clip((maxz - 40.0) / 20.0, 0.0, None) * 0.6  # expected flashes / pixel / step
    counts = r.poisson(rate)
    flash_density = (counts / px_area).astype(np.float64)
    ti, yi, xi = np.nonzero(counts)
    reps = counts[ti, yi, xi]
    flashes = np.stack(
        [
            np.repeat(ti, reps).astype(float),
            np.repeat(yi, reps) + r.uniform(-0.5, 0.5, reps.sum()),
            np.repeat(xi, reps) + r.uniform(-0.5, 0.5, reps.sum()),
        ],
        axis=1,
    ) if reps.sum() else np.zeros((0, 3))

    # NWP environment: smooth, slowly varying.
    base = _smooth_field(r, h, w, scale=size * 1.5)
    storm_env = np.clip(anvil.max(axis=0), 0, 1)
    cape = np.broadcast_to(800 + 2200 * base + 800 * storm_env, (T, h, w)).copy()
    cin = np.broadcast_to(-20 - 80 * (1 - base), (T, h, w)).copy()
    shear = np.broadcast_to(8 + 15 * _smooth_field(r, h, w, size), (T, h, w)).copy()
    pwat = np.broadcast_to(35 + 25 * base, (T, h, w)).copy()
    frz = np.broadcast_to(np.full((h, w), 4800.0) + 300 * base, (T, h, w)).copy()

    fields = {
        "maxz": maxz,
        "cappi3km": cappi,
        "tir1_bt": tir1,
        "tir2_bt": tir2,
        "wv_bt": wv,
        "mir_bt": mir,
        "tir1_cooling": cooling,
        "tir1_minus_wv": tir1_minus_wv,
        "flash_density": flash_density,
        "cape": cape,
        "cin": cin,
        "shear_0_6km": shear,
        "pwat": pwat,
        "freezing_level": frz,
    }
    fields = {n: fields[n].astype(np.float32) for n in ch.ALL_CHANNELS}

    missing = {g: np.zeros((T, h, w), dtype=bool) for g in ch.GROUPS}
    # Radar outages: whole frames, and a fixed beam-blocked wedge.
    if radar_missing_prob > 0:
        outage = r.random(T) < radar_missing_prob
        missing["radar"][outage] = True
        if r.random() < 0.5:
            ang = np.arctan2(yy - h / 2, xx - w / 2)
            a0 = r.uniform(-np.pi, np.pi)
            wedge = np.abs(np.angle(np.exp(1j * (ang - a0)))) < 0.25
            missing["radar"][:, wedge] = True
    for name in ch.channels_in_group("radar"):
        fields[name][missing["radar"]] = np.nan

    lat, lon = _latlon(h, w, grid_spacing_km)
    times = pd.date_range(start, periods=T, freq=f"{step_minutes}min")
    return SyntheticEvent(
        event_id=event_id or f"synth_{seed:05d}",
        times=times,
        fields=fields,
        missing=missing,
        flashes=flashes,
        lat=lat,
        lon=lon,
        grid_spacing_km=grid_spacing_km,
        motion_px=(float(vy), float(vx)),
        meta={"n_storms": k, "seed": seed},
    )


def generate_events(n: int, seed: int = 0, **kwargs) -> list[SyntheticEvent]:
    return [generate_event(seed * 100_003 + i, **kwargs) for i in range(n)]
