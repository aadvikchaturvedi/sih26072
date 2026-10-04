"""Rolling observation buffer: in memory, with a contract-format Zarr snapshot per domain
so a restart resumes with the frames it already had."""

from __future__ import annotations

import logging
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from nowcast_ml.data import channels as ch
from nowcast_ml.data.schema import FLASH_VARS, MISSING_VAR, X_VAR, build_event_dataset

from nowcast_backend.domain.errors import Conflict, NotFound
from nowcast_backend.domain.grid import Grid
from nowcast_backend.domain.models import DomainStatus
from nowcast_backend.domain.timeutil import aware_utc, naive_utc

log = logging.getLogger(__name__)


@dataclass
class _Buffer:
    grid: Grid
    channels: list[str]
    frames: dict[pd.Timestamp, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    flashes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # time, lat, lon

    def times(self) -> list[pd.Timestamp]:
        return sorted(self.frames)


class RollingObservationStore:
    def __init__(self, root: Path | None, max_frames: int, step_minutes: int = 10):
        self._root = root
        self._max_frames = max_frames
        self._step = pd.Timedelta(minutes=step_minutes)
        self._buffers: dict[str, _Buffer] = {}
        self._lock = threading.RLock()
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            for path in sorted(root.glob("*.zarr")):
                try:
                    self._insert(path.stem, xr.open_zarr(str(path), consolidated=None).load())
                except Exception:  # noqa: BLE001 - a bad snapshot must not stop startup
                    log.exception("ignoring unreadable observation snapshot %s", path)

    # ------------------------------------------------------------------ write
    def ingest(self, domain: str, ds: xr.Dataset) -> list[pd.Timestamp]:
        with self._lock:
            times = self._insert(domain, ds)
            self._snapshot(domain)
            return times

    def _insert(self, domain: str, ds: xr.Dataset) -> list[pd.Timestamp]:
        grid = Grid(
            np.asarray(ds["lat"].values, dtype=np.float64),
            np.asarray(ds["lon"].values, dtype=np.float64),
            float(ds.attrs["grid_spacing_km"]),
        )
        channels = [str(c) for c in ds["channel"].values]
        buf = self._buffers.get(domain)
        if buf is None:
            buf = _Buffer(grid, channels)
        elif not buf.grid.same_as(grid):
            raise Conflict(
                f"domain {domain!r} has a {buf.grid.shape} grid at {buf.grid.spacing_km} km; "
                "the new frames are on a different grid (use another domain id)"
            )
        elif set(buf.channels) != set(channels):
            raise Conflict(
                f"domain {domain!r} carries channels {buf.channels}; got {channels}. Keep the "
                "channel list fixed and flag unavailable data in 'missing' instead"
            )
        order = [channels.index(c) for c in buf.channels]
        x = ds[X_VAR].values.astype(np.float32, copy=False)[:, order]
        missing = self._group_mask(ds, x.shape)
        times = [naive_utc(t) for t in ds["time"].values]
        for i, t in enumerate(times):
            # A fully-missing frame is a gap filler, not an observation.
            if not missing[i].all():
                buf.frames[t] = (x[i].copy(), missing[i].copy())
        if all(v in ds for v in FLASH_VARS):
            self._merge_flashes(buf, ds)
        self._trim(buf)
        self._buffers[domain] = buf
        return [t for t in times if t in buf.frames]

    @staticmethod
    def _group_mask(ds: xr.Dataset, x_shape) -> np.ndarray:
        """(T, len(GROUPS), H, W) uint8 in canonical group order; absent groups count as valid."""
        T, _, H, W = x_shape
        out = np.zeros((T, len(ch.GROUPS), H, W), dtype=np.uint8)
        groups = [str(g) for g in ds["group"].values]
        src = ds[MISSING_VAR].values
        for j, g in enumerate(ch.GROUPS):
            if g in groups:
                out[:, j] = src[:, groups.index(g)] != 0
        return out

    @staticmethod
    def _merge_flashes(buf: _Buffer, ds: xr.Dataset) -> None:
        new = np.rec.fromarrays(
            [
                ds["flash_time"].values.astype("datetime64[ns]"),
                ds["flash_lat"].values.astype(np.float64),
                ds["flash_lon"].values.astype(np.float64),
            ],
            names="time,lat,lon",
        )
        if buf.flashes is not None:
            old = np.rec.fromarrays(list(buf.flashes), names="time,lat,lon")
            new = np.unique(np.concatenate([old, new]))  # frames are re-sent with overlap
        buf.flashes = (new["time"], new["lat"], new["lon"])

    def _trim(self, buf: _Buffer) -> None:
        if not buf.frames:
            return
        oldest = max(buf.frames) - (self._max_frames - 1) * self._step
        for t in [t for t in buf.frames if t < oldest]:
            del buf.frames[t]
        if buf.flashes is not None:
            keep = buf.flashes[0] > np.datetime64(oldest - self._step)
            buf.flashes = tuple(a[keep] for a in buf.flashes)

    def _snapshot(self, domain: str) -> None:
        buf = self._buffers[domain]
        if self._root is None or not buf.frames:
            return
        times = buf.times()
        ds, _ = self._dataset(domain, buf, times[0], times[-1])
        final, tmp = self._root / f"{domain}.zarr", self._root / f".{domain}.zarr.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        ds.to_zarr(str(tmp), mode="w", consolidated=True)
        shutil.rmtree(final, ignore_errors=True)
        tmp.rename(final)

    # ------------------------------------------------------------------ read
    def _buffer(self, domain: str) -> _Buffer:
        buf = self._buffers.get(domain)
        if buf is None or not buf.frames:
            raise NotFound(f"no observations for domain {domain!r}")
        return buf

    def _dataset(self, domain: str, buf: _Buffer, start: pd.Timestamp, stop: pd.Timestamp):
        times = pd.date_range(start, stop, freq=self._step)
        H, W = buf.grid.shape
        x = np.full((len(times), len(buf.channels), H, W), np.nan, dtype=np.float32)
        missing = np.ones((len(times), len(ch.GROUPS), H, W), dtype=np.uint8)
        observed = 0
        for i, t in enumerate(times):
            if t in buf.frames:
                x[i], missing[i] = buf.frames[t]
                observed += 1
        ds = build_event_dataset(
            x,
            buf.channels,
            times,
            buf.grid.lat,
            buf.grid.lon,
            missing,
            event_id=domain,
            grid_spacing_km=buf.grid.spacing_km,
            flashes=buf.flashes,
        )
        return ds, observed

    def domains(self) -> list[str]:
        with self._lock:
            return sorted(d for d, b in self._buffers.items() if b.frames)

    def grid(self, domain: str) -> Grid:
        with self._lock:
            return self._buffer(domain).grid

    def status(self, domain: str) -> DomainStatus:
        with self._lock:
            buf = self._buffer(domain)
            times = buf.times()
            return DomainStatus(
                domain=domain,
                shape=buf.grid.shape,
                grid_spacing_km=buf.grid.spacing_km,
                bounds=buf.grid.bounds,
                channels=list(buf.channels),
                times=[aware_utc(t) for t in times],
                latest_time=aware_utc(times[-1]),
            )

    def window(self, domain: str, t0: pd.Timestamp | None, n_frames: int):
        with self._lock:
            buf = self._buffer(domain)
            t0 = max(buf.frames) if t0 is None else naive_utc(t0)
            if t0 not in buf.frames:
                raise NotFound(f"domain {domain!r} has no observation frame at {t0} UTC")
            return self._dataset(domain, buf, t0 - (n_frames - 1) * self._step, t0)

    def flashes(self, domain: str, start: pd.Timestamp, stop: pd.Timestamp):
        with self._lock:
            buf = self._buffer(domain)
            if buf.flashes is None:
                empty = np.array([], dtype=np.float64)
                return np.array([], dtype="datetime64[ns]"), empty, empty
            t, lat, lon = buf.flashes
            keep = (t > np.datetime64(naive_utc(start))) & (t <= np.datetime64(naive_utc(stop)))
            return t[keep], lat[keep], lon[keep]

    def frame(self, domain: str, time: pd.Timestamp, channel: str) -> np.ndarray:
        """(H, W) float32 field with NaN where the channel's group is missing."""
        with self._lock:
            buf = self._buffer(domain)
            time = naive_utc(time)
            if time not in buf.frames:
                raise NotFound(f"domain {domain!r} has no observation frame at {time} UTC")
            if channel not in buf.channels:
                raise NotFound(f"domain {domain!r} has no channel {channel!r}")
            x, missing = buf.frames[time]
            gone = missing[ch.GROUPS.index(ch.group_of(channel))].astype(bool)
            return np.where(gone, np.nan, x[buf.channels.index(channel)])
