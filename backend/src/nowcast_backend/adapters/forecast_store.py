"""Forecasts on disk::

    <root>/<domain>/<YYYYmmddTHHMMZ>/<source>/forecast.zarr   output-contract Dataset
                                              meta.json       ForecastMeta
                                              cells.json      storm cells

``meta.json`` is written last, so a directory with it is complete.
"""

from __future__ import annotations

import json
import shutil
import threading
from collections import OrderedDict
from pathlib import Path

import pandas as pd
import xarray as xr

from nowcast_backend.domain.errors import NotFound
from nowcast_backend.domain.models import ForecastMeta, StormCell
from nowcast_backend.domain.timeutil import naive_utc, stamp


class ZarrForecastStore:
    def __init__(self, root: Path, cache_size: int = 8):
        self._root = root
        self._cache: OrderedDict[Path, xr.Dataset] = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        root.mkdir(parents=True, exist_ok=True)

    def _dir(self, domain: str, t0: pd.Timestamp, source: str) -> Path:
        return self._root / domain / stamp(t0) / source

    def _require(self, domain: str, t0: pd.Timestamp, source: str) -> Path:
        d = self._dir(domain, t0, source)
        if not (d / "meta.json").exists():
            raise NotFound(f"no {source} forecast for domain {domain!r} at {naive_utc(t0)} UTC")
        return d

    def save(self, meta: ForecastMeta, forecast: xr.Dataset, cells: list[StormCell]) -> None:
        d = self._dir(meta.domain, meta.t0, meta.source)
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        forecast.to_zarr(str(d / "forecast.zarr"), mode="w", consolidated=True)
        (d / "cells.json").write_text(json.dumps([c.model_dump(mode="json") for c in cells]))
        (d / "meta.json").write_text(meta.model_dump_json())
        with self._lock:
            self._cache[d] = forecast
            self._cache.move_to_end(d)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

    def exists(self, domain: str, t0: pd.Timestamp, source: str) -> bool:
        return (self._dir(domain, t0, source) / "meta.json").exists()

    def meta(self, domain: str, t0: pd.Timestamp, source: str) -> ForecastMeta:
        d = self._require(domain, t0, source)
        return ForecastMeta.model_validate_json((d / "meta.json").read_text())

    def cells(self, domain: str, t0: pd.Timestamp, source: str) -> list[StormCell]:
        d = self._require(domain, t0, source)
        return [StormCell.model_validate(c) for c in json.loads((d / "cells.json").read_text())]

    def dataset(self, domain: str, t0: pd.Timestamp, source: str) -> xr.Dataset:
        d = self._require(domain, t0, source)
        with self._lock:
            if d in self._cache:
                self._cache.move_to_end(d)
                return self._cache[d]
        ds = xr.open_zarr(str(d / "forecast.zarr"), consolidated=True).load()
        with self._lock:
            self._cache[d] = ds
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return ds

    def times(self, domain: str, source: str) -> list[pd.Timestamp]:
        base = self._root / domain
        if not base.is_dir():
            return []
        return sorted(
            naive_utc(p.name) for p in base.iterdir() if (p / source / "meta.json").exists()
        )

    def prune(self, domain: str, keep: int) -> None:
        base = self._root / domain
        if not base.is_dir():
            return
        dirs = sorted(p for p in base.iterdir() if p.is_dir())
        for old in dirs[: max(0, len(dirs) - keep)]:
            with self._lock:
                for key in [k for k in self._cache if old in k.parents]:
                    del self._cache[key]
            shutil.rmtree(old, ignore_errors=True)
