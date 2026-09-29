"""Event: a contract-format Dataset viewed through a model's channel list.

This is the single place where contract data becomes model arrays:

* channels are reordered to the model's list; absent channels become NaN,
* ``avail`` (T, C, H, W) = channel present AND group not missing AND finite,
* lightning occurrence comes from the flash point table if present, else
  ``flash_density > 0``.

Used by training datasets, evaluation and :class:`~nowcast_ml.inference.Predictor`.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pandas as pd
import xarray as xr

from nowcast_ml.data import channels as ch
from nowcast_ml.data import labels as lb
from nowcast_ml.data.schema import MISSING_VAR, X_VAR


class Event:
    def __init__(self, ds: xr.Dataset, channels: list[str] | tuple[str, ...]):
        self.ds = ds
        self.channels = list(channels)
        self.event_id = str(ds.attrs.get("event_id", "unknown"))
        self.grid_spacing_km = float(ds.attrs.get("grid_spacing_km", 2.0))
        ds_chans = [str(c) for c in ds["channel"].values]
        self._src_index = [ds_chans.index(c) if c in ds_chans else -1 for c in self.channels]
        groups = [str(g) for g in ds["group"].values] if "group" in ds.coords else []
        self._group_index = [
            groups.index(ch.group_of(c)) if ch.group_of(c) in groups else -1 for c in self.channels
        ]
        self.missing_channels = [
            c for c, i in zip(self.channels, self._src_index, strict=True) if i < 0
        ]
        self.extra_channels = [c for c in ds_chans if c not in self.channels]

    @classmethod
    def open(cls, path, channels) -> Event:
        return cls(xr.open_zarr(str(path), consolidated=None), channels)

    # ------------------------------------------------------------------ basic info
    @property
    def times(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.ds["time"].values)

    @property
    def n_times(self) -> int:
        return int(self.ds.sizes["time"])

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.ds.sizes["y"]), int(self.ds.sizes["x"])

    @property
    def lat(self) -> np.ndarray:
        return self.ds["lat"].values

    @property
    def lon(self) -> np.ndarray:
        return self.ds["lon"].values

    def index_of(self, t0) -> int:
        """Index of the frame at time ``t0`` (exact match required)."""
        t0 = pd.Timestamp(t0)
        if t0.tzinfo is not None:
            t0 = t0.tz_convert("UTC").tz_localize(None)
        idx = self.times.get_indexer([t0])[0]
        if idx < 0:
            raise KeyError(f"t0={t0} not found in event {self.event_id}")
        return int(idx)

    # ------------------------------------------------------------------ windows
    def window(self, start: int, stop: int) -> tuple[np.ndarray, np.ndarray]:
        """Frames [start, stop) -> (x (n, C, H, W) float32 with NaN where unavailable, avail bool)."""
        if start < 0 or stop > self.n_times:
            raise IndexError(f"window [{start}, {stop}) outside 0..{self.n_times}")
        sub = self.ds.isel(time=slice(start, stop))
        xs = sub[X_VAR].values.astype(np.float32, copy=False)
        n = stop - start
        H, W = self.shape
        C = len(self.channels)
        x = np.full((n, C, H, W), np.nan, dtype=np.float32)
        avail = np.zeros((n, C, H, W), dtype=bool)
        miss = sub[MISSING_VAR].values.astype(bool) if MISSING_VAR in sub else None
        for j, (si, gi) in enumerate(zip(self._src_index, self._group_index, strict=True)):
            if si < 0:
                continue
            v = xs[:, si]
            ok = np.isfinite(v)
            if miss is not None and gi >= 0:
                ok &= ~miss[:, gi]
            x[:, j] = np.where(ok, v, np.nan)
            avail[:, j] = ok
        return x, avail

    def target(self, start: int, stop: int, channel: str = ch.TARGET_CHANNEL):
        """(n, H, W) target values and validity for one channel."""
        x, avail = self.window(start, stop)
        j = self.channels.index(channel)
        return x[:, j], avail[:, j]

    # ------------------------------------------------------------------ lightning
    @cached_property
    def lightning_occurrence(self) -> np.ndarray:
        """(T, H, W) bool flash occurrence per frame."""
        if "flash_time" in self.ds:
            return lb.occurrence_from_points(
                self.ds["flash_time"].values,
                self.ds["flash_lat"].values,
                self.ds["flash_lon"].values,
                self.ds["time"].values,
                self.lat,
                self.lon,
            )
        ds_chans = [str(c) for c in self.ds["channel"].values]
        if ch.LIGHTNING_CHANNEL in ds_chans:
            fd = self.ds[X_VAR].isel(channel=ds_chans.index(ch.LIGHTNING_CHANNEL)).values
            return np.nan_to_num(fd, nan=0.0) > 0
        return np.zeros((self.n_times, *self.shape), dtype=bool)

    @cached_property
    def lightning_missing(self) -> np.ndarray:
        """(T, H, W) bool: lightning observations missing (labels untrustworthy)."""
        has_points = "flash_time" in self.ds
        ds_chans = [str(c) for c in self.ds["channel"].values]
        if not has_points and ch.LIGHTNING_CHANNEL not in ds_chans:
            return np.ones((self.n_times, *self.shape), dtype=bool)
        groups = [str(g) for g in self.ds["group"].values] if "group" in self.ds.coords else []
        if MISSING_VAR in self.ds and "lightning" in groups:
            return self.ds[MISSING_VAR].sel(group="lightning").values.astype(bool)
        return np.zeros((self.n_times, *self.shape), dtype=bool)


def radar_coverage(avail: np.ndarray, channels: list[str]) -> float:
    """Fraction of pixels with any radar channel available in the last frame of ``avail``."""
    idx = [i for i, c in enumerate(channels) if ch.group_of(c) == "radar"]
    if not idx:
        return 0.0
    return float(avail[-1, idx].any(axis=0).mean())
