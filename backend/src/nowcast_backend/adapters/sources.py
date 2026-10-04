"""Frame sources the scheduler polls.

Decoding raw radar / INSAT / lightning-network files is the data team's job; they
write input-contract stores (``ml/docs/data_contract.md``) and the backend reads those.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def _open(path: Path) -> xr.Dataset:
    return xr.open_zarr(str(path), consolidated=None)


class ZarrDirectorySource:
    """Watches a directory of live ``<domain>.zarr`` contract stores that a producer keeps
    appending to, and yields the frames added since the previous poll."""

    def __init__(self, directory: Path, initial_frames: int):
        self._dir = directory
        self._initial = initial_frames
        self._seen: dict[str, pd.Timestamp] = {}

    def poll(self):
        for path in sorted(self._dir.glob("*.zarr")):
            domain = path.stem
            try:
                ds = _open(path)
                times = pd.DatetimeIndex(ds["time"].values)
                last = self._seen.get(domain)
                start = (
                    max(0, len(times) - self._initial)
                    if last is None
                    else int(times.searchsorted(last, side="right"))
                )
                if start >= len(times):
                    continue
                new = ds.isel(time=slice(start, None)).load()
            except Exception:  # noqa: BLE001 - e.g. the producer is mid-write; retry next poll
                log.exception("cannot read %s; will retry", path)
                continue
            self._seen[domain] = times[-1]
            yield domain, new


class ReplaySource:
    """Replays an archived event as if it were arriving live: the first poll yields
    ``initial_frames`` frames, every later poll yields the next one."""

    def __init__(self, store: Path, domain: str, initial_frames: int):
        self._ds = _open(store)
        self._domain = domain
        self._next = 0
        self._initial = initial_frames

    def poll(self):
        n = self._ds.sizes["time"]
        if self._next >= n:
            return
        stop = min(n, self._next + (self._initial if self._next == 0 else 1))
        frames = self._ds.isel(time=slice(self._next, stop)).load()
        self._next = stop
        yield self._domain, frames
