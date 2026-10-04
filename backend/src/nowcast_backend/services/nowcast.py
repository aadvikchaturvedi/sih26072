"""The use cases of the backend: ingest observations, run forecasts, serve products.

Everything here talks to ports only, so it runs the same against the real model
and stores or against in-memory fakes.
"""

from __future__ import annotations

import logging
import re
import threading
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from nowcast_ml.data import channels as ch

from nowcast_backend.domain.errors import Conflict, InvalidObservations, InvalidRequest, NotFound
from nowcast_backend.domain.grid import Grid
from nowcast_backend.domain.models import (
    SOURCES,
    DistrictForecast,
    DistrictWarning,
    DomainStatus,
    Event,
    ForecastMeta,
    IngestResult,
    InputHealth,
    ModelDescription,
    Region,
    StormCell,
    Warning,
)
from nowcast_backend.domain.timeutil import aware_utc, naive_utc, now_utc
from nowcast_backend.ports import (
    EvaluationSource,
    ForecastEngine,
    ForecastStore,
    Notifier,
    ObservationStore,
    RegionProvider,
)
from nowcast_backend.services import cells as cell_service
from nowcast_backend.services import districts as district_service
from nowcast_backend.services import warnings as warning_rules
from nowcast_backend.services.warnings import WarningService
from nowcast_backend.settings import Settings

log = logging.getLogger(__name__)

_DOMAIN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def check_domain_id(domain: str) -> str:
    """Domain ids become directory names, so they are restricted to a safe alphabet."""
    if not _DOMAIN_ID.fullmatch(domain):
        raise InvalidRequest(
            f"invalid domain id {domain!r}: use lowercase letters, digits, '-' and '_' (max 64)"
        )
    return domain


class NowcastService:
    def __init__(
        self,
        settings: Settings,
        engine: ForecastEngine,
        observations: ObservationStore,
        forecasts: ForecastStore,
        warnings: WarningService,
        notifier: Notifier,
        regions: RegionProvider,
        evaluation: EvaluationSource,
    ):
        self._settings = settings
        self._engine = engine
        self._observations = observations
        self._forecasts = forecasts
        self._warnings = warnings
        self._notifier = notifier
        self._regions = regions
        self._evaluation = evaluation
        self._region_index: dict[str, tuple[Grid, district_service.RegionIndex]] = {}
        self._model = engine.describe()
        self._run_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)

    # ------------------------------------------------------------------ model / domains
    @property
    def model(self) -> ModelDescription:
        return self._model

    def domains(self) -> list[DomainStatus]:
        return [self._observations.status(d) for d in self._observations.domains()]

    def domain(self, domain: str) -> DomainStatus:
        return self._observations.status(check_domain_id(domain))

    def now(self, domain: str | None = None) -> pd.Timestamp:
        """Reference time for warning expiry (see ``Settings.clock``)."""
        if self._settings.clock == "data":
            names = [domain] if domain else self._observations.domains()
            latest = [self._observations.status(d).latest_time for d in names if d]
            if latest:
                return naive_utc(max(latest))
        return now_utc()

    # ------------------------------------------------------------------ ingest
    def ingest(self, domain: str, ds: xr.Dataset, run: bool | None = None) -> IngestResult:
        """Validate and store observation frames; forecast if they advance the domain's time."""
        check_domain_id(domain)
        if not isinstance(ds, xr.Dataset):
            raise InvalidObservations("observations must be an xarray Dataset")
        limit = self._settings.max_grid_px
        if ds.sizes.get("y", 0) > limit or ds.sizes.get("x", 0) > limit:
            raise InvalidObservations(f"grid larger than {limit} x {limit} pixels")
        if ds.sizes.get("time", 0) > self._settings.buffer_frames:
            ds = ds.isel(time=slice(-self._settings.buffer_frames, None))
        problems = self._engine.validate(ds)
        if problems:
            raise InvalidObservations("observations violate the input contract", problems)
        try:
            before = self._observations.status(domain).latest_time
        except NotFound:
            before = None
        accepted = self._observations.ingest(domain, ds)
        if not accepted:
            raise InvalidObservations("every frame is fully missing; nothing to ingest")
        status = self._observations.status(domain)
        result = IngestResult(
            domain=domain,
            accepted_times=[aware_utc(t) for t in accepted],
            latest_time=status.latest_time,
            n_frames=len(status.times),
        )
        self._notifier.publish(
            Event(
                type="observation.ingested",
                domain=domain,
                time=status.latest_time,
                data={"frames": len(accepted)},
            )
        )
        if self._settings.auto_forecast if run is None else run:
            # Every newly arrived time gets its forecast, oldest first, so a batch of
            # frames (a late delivery, a replay) leaves no holes in the forecast series.
            for t in sorted(t for t in accepted if before is None or aware_utc(t) > before):
                try:
                    result.forecast = self.run(domain, t)
                except Conflict as e:  # not enough history yet: the frames are stored regardless
                    log.info("domain %s: no forecast for %s: %s", domain, t, e.message)
        return result

    # ------------------------------------------------------------------ forecast
    def run(
        self,
        domain: str,
        t0: pd.Timestamp | None = None,
        source: str = "model",
        n_members: int = 0,
        force: bool = False,
    ) -> ForecastMeta:
        """Forecast from the frames ending at ``t0`` (default: latest) and store the products."""
        check_domain_id(domain)
        if source not in SOURCES:
            raise InvalidRequest(f"source must be one of {SOURCES}, got {source!r}")
        status = self._observations.status(domain)
        latest = naive_utc(status.latest_time)
        t0 = latest if t0 is None else naive_utc(t0)
        with self._run_locks[domain]:
            if not force and self._forecasts.exists(domain, t0, source):
                meta = self._forecasts.meta(domain, t0, source)
                if meta.n_members >= n_members:
                    return meta
            inputs, observed = self._observations.window(domain, t0, self._model.t_in)
            need = min(self._settings.min_observed_frames, self._model.t_in)
            if observed < need:
                raise Conflict(
                    f"domain {domain!r} has {observed} observed frame(s) in the "
                    f"{self._model.t_in} ending at {t0} UTC; {need} are needed"
                )
            forecast = self._engine.forecast(inputs, t0, source=source, n_members=n_members)
            grid = self._observations.grid(domain)
            step = pd.Timedelta(minutes=self._model.step_minutes)
            cells = cell_service.analyse(
                forecast,
                self._radar_frames(inputs),
                grid,
                self._settings.cells,
                self._model.step_minutes,
                previous=self._previous_cells(domain, t0, source),
                flashes=self._observations.flashes(domain, t0 - self._model.t_in * step, t0),
            )
            index = self._index(domain, grid)
            for cell in cells:
                cell.affected = district_service.affected_regions(cell, index, grid)
            districts = district_service.assess(forecast, index, cells, self._settings.districts)
            meta = self._describe(domain, source, forecast, status, observed, len(cells))
            meta.inputs = self._input_health(inputs)
            self._forecasts.save(meta, forecast, {"cells": cells, "districts": districts})
            self._forecasts.prune(domain, self._settings.forecast_retention)
        self._notifier.publish(
            Event(
                type="forecast.ready",
                domain=domain,
                time=meta.t0,
                data={"source": source, "mode": meta.mode, "cells": len(cells)},
            )
        )
        # Only the model's forecast for the newest observations drives public warnings.
        if source == "model" and t0 == latest:
            issued = self._warnings.replace(
                domain,
                warning_rules.derive(
                    forecast, grid, domain, self._settings.warnings, self.now(domain)
                ),
                self.now(domain),
            )
            self._notifier.publish(
                Event(
                    type="warnings.updated",
                    domain=domain,
                    time=meta.t0,
                    data={"warnings": [w.model_dump(mode="json") for w in issued]},
                )
            )
        return meta

    @staticmethod
    def _radar_frames(inputs: xr.Dataset) -> list[tuple[pd.Timestamp, np.ndarray]]:
        """MAX-Z per input frame, NaN where radar is missing; empty without a radar channel."""
        channels = [str(c) for c in inputs["channel"].values]
        if "maxz" not in channels:
            return []
        maxz = inputs["x"].values[:, channels.index("maxz")]
        groups = [str(g) for g in inputs["group"].values]
        gone = inputs["missing"].values[:, groups.index("radar")].astype(bool)
        maxz = np.where(gone, np.nan, maxz)
        return [(naive_utc(t), f) for t, f in zip(inputs["time"].values, maxz, strict=True)]

    def _previous_cells(self, domain: str, t0: pd.Timestamp, source: str) -> list[StormCell]:
        """Cells of the most recent earlier forecast, if it is recent enough to match against."""
        earlier = [t for t in self._forecasts.times(domain, source) if t < t0]
        if not earlier or t0 - earlier[-1] > pd.Timedelta(minutes=3 * self._model.step_minutes):
            return []
        return self._cells(domain, earlier[-1], source)

    def _cells(self, domain: str, t0: pd.Timestamp, source: str) -> list[StormCell]:
        items = self._forecasts.product(domain, t0, source, "cells")
        return [StormCell.model_validate(c) for c in items]

    def _index(self, domain: str, grid: Grid) -> district_service.RegionIndex:
        """Regions rasterised on the domain's grid (rebuilt only if the grid changes)."""
        cached = self._region_index.get(domain)
        if cached is None or cached[0] is not grid:
            cached = (grid, district_service.RegionIndex(self._regions.regions(), grid))
            self._region_index[domain] = cached
        return cached[1]

    def _input_health(self, inputs: xr.Dataset) -> list[InputHealth]:
        """Per input group: is it present at t0, and if not, when was it last seen."""
        times = [naive_utc(t) for t in inputs["time"].values]
        present = {ch.group_of(str(c)) for c in inputs["channel"].values}
        groups = [str(g) for g in inputs["group"].values]
        missing = inputs["missing"].values.astype(bool)
        floor = self._settings.min_group_coverage
        out = []
        for group in ch.GROUPS:
            seen = None
            if group in present:
                coverage = 1.0 - missing[:, groups.index(group)].mean(axis=(1, 2))
                frames = np.nonzero(coverage >= floor)[0]
                seen = times[int(frames[-1])] if len(frames) else None
            age = None if seen is None else int((times[-1] - seen).total_seconds())
            out.append(
                InputHealth(
                    name=group,
                    status="missing" if seen is None else "live" if age == 0 else "stale",
                    last_received=None if seen is None else aware_utc(seen),
                    stale_age_seconds=age or None,
                )
            )
        return out

    def _describe(
        self,
        domain: str,
        source: str,
        forecast: xr.Dataset,
        status: DomainStatus,
        observed: int,
        n_cells: int,
    ) -> ForecastMeta:
        attrs = forecast.attrs
        refl = forecast["reflectivity"].values
        ltg = [
            int(v.rsplit("_", 1)[1]) for v in forecast.data_vars if v.startswith("lightning_prob_")
        ]
        return ForecastMeta(
            domain=domain,
            t0=aware_utc(attrs["t0"]),
            source=source,
            model_name=str(attrs["model_name"]),
            model_version=str(attrs["model_version"]),
            mode=attrs["mode"],
            missing_channels=[str(c) for c in attrs.get("missing_channels", [])],
            lead_minutes=[int(v) for v in forecast["lead"].values],
            lightning_leads_min=sorted(ltg),
            n_members=int(forecast.sizes.get("member", 0)),
            inference_ms=float(attrs["inference_ms"]),
            created_at=aware_utc(now_utc()),
            shape=status.shape,
            grid_spacing_km=status.grid_spacing_km,
            bounds=status.bounds,
            observed_frames=observed,
            max_reflectivity_dbz=float(np.nanmax(refl)) if np.isfinite(refl).any() else 0.0,
            max_lightning_prob=max(
                (float(forecast[f"lightning_prob_{m}"].values.max()) for m in ltg), default=0.0
            ),
            n_cells=n_cells,
        )

    # ------------------------------------------------------------------ products
    def resolve_t0(self, domain: str, t0: pd.Timestamp | None, source: str) -> pd.Timestamp:
        """``None`` means the newest stored forecast of that source."""
        check_domain_id(domain)
        if t0 is not None:
            return naive_utc(t0)
        times = self._forecasts.times(domain, source)
        if not times:
            raise NotFound(f"no {source} forecast for domain {domain!r} yet")
        return times[-1]

    def forecast_times(self, domain: str, source: str, limit: int) -> list[ForecastMeta]:
        times = self._forecasts.times(check_domain_id(domain), source)[-limit:]
        return [self._forecasts.meta(domain, t, source) for t in reversed(times)]

    def forecast_meta(self, domain: str, t0: pd.Timestamp | None, source: str) -> ForecastMeta:
        return self._forecasts.meta(domain, self.resolve_t0(domain, t0, source), source)

    def forecast(self, domain: str, t0: pd.Timestamp | None, source: str) -> xr.Dataset:
        return self._forecasts.dataset(domain, self.resolve_t0(domain, t0, source), source)

    def cells(self, domain: str, t0: pd.Timestamp | None, source: str) -> list[StormCell]:
        return self._cells(domain, self.resolve_t0(domain, t0, source), source)

    def forecast_time_list(self, domain: str, source: str = "model") -> list[pd.Timestamp]:
        """Analysis times with a stored forecast, oldest first."""
        return self._forecasts.times(check_domain_id(domain), source)

    def districts(self, domain: str, t0: pd.Timestamp | None) -> list[DistrictForecast]:
        t0 = self.resolve_t0(domain, t0, "model")
        items = self._forecasts.product(domain, t0, "model", "districts")
        return [DistrictForecast.model_validate(d) for d in items]

    def district_warnings(self, domain: str, until: pd.Timestamp | None) -> list[DistrictWarning]:
        """District level changes in the stored forecasts up to ``until``, oldest first."""
        until = self.resolve_t0(domain, until, "model")
        times = [t for t in self._forecasts.times(domain, "model") if t <= until]
        return district_service.level_changes(
            ((t, self.districts(domain, t), self._cells(domain, t, "model")) for t in times),
            self._settings.districts.valid_minutes,
        )

    def regions(self) -> list[Region]:
        return self._regions.regions()

    def flashes(self, domain: str, t0: pd.Timestamp, minutes: int = 30):
        """Flash points on the domain during the ``minutes`` up to ``t0``: (time, lat, lon)."""
        check_domain_id(domain)
        t0 = naive_utc(t0)
        time, lat, lon = self._observations.flashes(domain, t0 - pd.Timedelta(minutes=minutes), t0)
        b = self._observations.grid(domain).bounds
        inside = (lat >= b.south) & (lat <= b.north) & (lon >= b.west) & (lon <= b.east)
        return time[inside], lat[inside], lon[inside]

    def evaluation(self) -> dict:
        metrics = self._evaluation.metrics()
        if metrics is None:
            raise NotFound("this model has no evaluation report yet (run nowcast-eval)")
        return metrics

    def layer(
        self,
        domain: str,
        t0: pd.Timestamp | None,
        source: str,
        layer: str,
        lead: int | None = None,
        threshold: float = 35.0,
    ) -> np.ndarray:
        """One (H, W) forecast field.

        ``reflectivity`` needs ``lead``; ``exceedance`` is the fraction of ensemble
        members at or above ``threshold`` dBZ at ``lead``.
        """
        ds = self.forecast(domain, t0, source)
        if layer in ("reflectivity", "exceedance"):
            leads = [int(v) for v in ds["lead"].values]
            if lead not in leads:
                raise InvalidRequest(f"lead must be one of {leads} (minutes), got {lead}")
            if layer == "reflectivity":
                return ds["reflectivity"].values[leads.index(lead)]
            if "reflectivity_members" not in ds:
                raise NotFound("this forecast has no ensemble members")
            members = ds["reflectivity_members"].values[:, leads.index(lead)]
            return (members >= threshold).mean(axis=0).astype(np.float32)
        if layer == "first_flash" or layer.startswith("lightning_prob_"):
            if layer in ds:
                return ds[layer].values.astype(np.float32)
        raise NotFound(f"unknown layer {layer!r}; available: {self.layer_names(ds)}")

    @staticmethod
    def layer_names(ds: xr.Dataset) -> list[str]:
        names = [
            "reflectivity",
            *sorted(v for v in ds.data_vars if v.startswith("lightning_prob_")),
        ]
        names.append("first_flash")
        if "reflectivity_members" in ds:
            names.append("exceedance")
        return names

    def point(self, domain: str, t0: pd.Timestamp | None, source: str, lat: float, lon: float):
        """Forecast values at the grid pixel containing (lat, lon)."""
        ds = self.forecast(domain, t0, source)
        grid = self._observations.grid(domain)
        iy, ix = grid.nearest_index(lat, lon)
        refl = ds["reflectivity"].values[:, iy, ix]
        out = {
            "t0": ds.attrs["t0"],
            "lat": float(grid.lat[iy, ix]),
            "lon": float(grid.lon[iy, ix]),
            "reflectivity": [
                {
                    "lead": int(lead),
                    "valid_time": aware_utc(valid),
                    "dbz": None if np.isnan(v) else round(float(v), 1),
                }
                for lead, valid, v in zip(
                    ds["lead"].values, ds["valid_time"].values, refl, strict=True
                )
            ],
            "lightning_prob": {
                v.rsplit("_", 1)[1]: round(float(ds[v].values[iy, ix]), 3)
                for v in sorted(ds.data_vars)
                if v.startswith("lightning_prob_")
            },
            "first_flash": bool(ds["first_flash"].values[iy, ix]),
        }
        if "reflectivity_members" in ds:
            members = ds["reflectivity_members"].values[:, :, iy, ix]
            for row, lo, hi in zip(
                out["reflectivity"],
                np.percentile(members, 10, axis=0),
                np.percentile(members, 90, axis=0),
                strict=True,
            ):
                row["dbz_p10"], row["dbz_p90"] = round(float(lo), 1), round(float(hi), 1)
        return out

    def observation(self, domain: str, time: pd.Timestamp | None, channel: str) -> np.ndarray:
        status = self.domain(domain)
        time = naive_utc(status.latest_time) if time is None else time
        return self._observations.frame(domain, time, channel)

    # ------------------------------------------------------------------ warnings
    def active_warnings(self, domain: str | None = None) -> list[Warning]:
        if domain is not None:
            check_domain_id(domain)
            return self._warnings.active(domain, self.now(domain))
        found = []
        for d in self._observations.domains():
            found += self._warnings.active(d, self.now(d))
        return found

    def warning(self, warning_id: str) -> tuple[Warning, Warning | None]:
        """A warning and the one it supersedes (if any)."""
        w = self._warnings.get(warning_id)
        if w is None:
            raise NotFound(f"no warning {warning_id!r}")
        return w, self._warnings.get(w.supersedes) if w.supersedes else None

    def warning_history(self, domain: str | None, status: str | None, limit: int) -> list[Warning]:
        if domain is not None:
            check_domain_id(domain)
        return self._warnings.history(domain, status, limit)
