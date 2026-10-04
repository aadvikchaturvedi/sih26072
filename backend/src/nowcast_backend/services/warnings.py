"""Warnings: derived from a forecast by threshold rules, then tracked across runs.

Each connected area above the yellow threshold becomes one warning whose level is
the highest threshold reached inside it. A new run's warnings replace the domain's
active ones: overlapping warnings of the same hazard are superseded (CAP ``Update``),
the rest are cancelled or expire.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from nowcast_ml.data.labels import dilate, radius_px
from scipy import ndimage

from nowcast_backend.domain.grid import Grid
from nowcast_backend.domain.models import LEVELS, Warning
from nowcast_backend.domain.timeutil import aware_utc, stamp
from nowcast_backend.ports import WarningRepository
from nowcast_backend.services.cap import INSTRUCTIONS
from nowcast_backend.services.geo import bboxes_intersect, mask_polygon
from nowcast_backend.settings import WarningSettings

_HAZARD_NAME = {"lightning": "Lightning", "thunderstorm": "Thunderstorm"}


def _level(peak: float, thresholds: tuple[float, float, float]) -> str:
    return LEVELS[max(0, int(np.searchsorted(thresholds, peak, side="right")) - 1)]


def derive(
    forecast: xr.Dataset,
    grid: Grid,
    domain: str,
    cfg: WarningSettings,
    issued_at: pd.Timestamp,
) -> list[Warning]:
    """Warnings implied by one forecast (not yet stored)."""
    t0 = pd.Timestamp(forecast.attrs["t0"]).tz_localize(None)
    leads = np.array([int(v) for v in forecast["lead"].values])
    mode = str(forecast.attrs["mode"])
    first_flash = forecast["first_flash"].values.astype(bool)
    grow = radius_px(cfg.buffer_km, grid.spacing_km)
    min_px = max(1, int(np.ceil(cfg.min_area_km2 / grid.pixel_area_km2)))
    degraded = (
        " Radar data were unavailable; this forecast is based on satellite data only and is "
        "less precise."
        if mode == "satellite_only"
        else ""
    )

    hazards = []
    ltg_leads = sorted(
        int(v.rsplit("_", 1)[1]) for v in forecast.data_vars if v.startswith("lightning_prob_")
    )
    if ltg_leads:
        lead = ltg_leads[0]
        prob = forecast[f"lightning_prob_{lead}"].values
        hazards.append(("lightning", prob, cfg.lightning_prob, "probability", lead, None))
    within = leads <= cfg.thunderstorm_horizon_min
    if within.any():
        refl = np.nan_to_num(forecast["reflectivity"].values[within], nan=-np.inf)
        hazards.append(
            (
                "thunderstorm",
                refl.max(axis=0),
                cfg.thunderstorm_dbz,
                "dBZ",
                int(leads[within][-1]),
                refl,
            )
        )

    out: list[Warning] = []
    for hazard, field, thresholds, unit, horizon, per_lead in hazards:
        labels, n = ndimage.label(dilate(field >= thresholds[0], grow))
        for i in range(1, n + 1):
            area = labels == i
            if area.sum() < min_px:
                continue
            polygon = mask_polygon(area, grid, simplify_px=1.0)
            if polygon is None:
                continue
            peak = float(field[area].max())
            level = _level(peak, thresholds)
            onset = t0
            if per_lead is not None:  # first lead at which the area reaches the threshold
                hit = (per_lead[:, area] >= thresholds[0]).any(axis=1)
                onset = t0 + pd.Timedelta(minutes=int(leads[within][int(np.argmax(hit))]))
            expires = t0 + pd.Timedelta(minutes=horizon)
            new_storm = hazard == "lightning" and bool(first_flash[area].any())
            iy, ix = ndimage.center_of_mass(area)
            lat, lon = grid.latlon_at(iy, ix)
            km2 = float(area.sum() * grid.pixel_area_km2)
            if hazard == "lightning":
                what = f"Lightning within 10 km is forecast in the next {horizon} minutes"
                detail = f"peak probability {peak:.0%}"
                if new_storm:
                    detail += "; includes areas with no lightning in the past 30 minutes"
            else:
                what = f"Intense thunderstorm cells are forecast until {expires:%H:%M} UTC"
                detail = f"peak reflectivity {peak:.0f} dBZ"
            out.append(
                Warning(
                    id=f"{domain}-{hazard}-{stamp(t0)}-{len(out) + 1:02d}",
                    domain=domain,
                    hazard=hazard,
                    level=level,
                    t0=aware_utc(t0),
                    issued_at=aware_utc(issued_at),
                    onset=aware_utc(onset),
                    expires=aware_utc(expires),
                    headline=f"{level.capitalize()} warning: {_HAZARD_NAME[hazard].lower()} "
                    f"until {expires:%H:%M} UTC",
                    description=f"{what} over about {km2:.0f} km² ({detail}).{degraded}",
                    instruction=INSTRUCTIONS[level],
                    peak_value=round(peak, 3),
                    peak_unit=unit,
                    first_flash=new_storm,
                    area_km2=km2,
                    lat=float(lat[0]),
                    lon=float(lon[0]),
                    polygon=polygon,
                    mode=mode,
                    model_version=str(forecast.attrs["model_version"]),
                )
            )
    return out


class WarningService:
    def __init__(self, repo: WarningRepository):
        self._repo = repo

    def replace(self, domain: str, new: list[Warning], now: pd.Timestamp) -> list[Warning]:
        """Make ``new`` the domain's active warnings; returns them with lineage filled in."""
        now = aware_utc(now)
        previous = self._repo.find(domain=domain, status="active")
        same_run = {p.id: p for p in previous if p.id in {w.id for w in new}}
        superseded: set[str] = set()
        issued = []
        for w in new:
            if w.id in same_run:  # a forced re-run of the same forecast keeps its lineage
                old = same_run[w.id]
                w = w.model_copy(update={"supersedes": old.supersedes, "msg_type": old.msg_type})
                issued.append(w)
                continue
            match = next(
                (
                    p
                    for p in previous
                    if p.id not in superseded
                    and p.id not in same_run
                    and p.hazard == w.hazard
                    and bboxes_intersect(p.polygon, w.polygon)
                ),
                None,
            )
            if match is not None:
                superseded.add(match.id)
                w = w.model_copy(update={"supersedes": match.id, "msg_type": "Update"})
            issued.append(w)
        for p in previous:
            if p.id in same_run:
                continue
            status = (
                "superseded"
                if p.id in superseded
                else "cancelled"
                if p.expires > now
                else "expired"
            )
            self._repo.set_status(p.id, status)
        self._repo.add(issued)
        return issued

    def active(self, domain: str | None, now: pd.Timestamp) -> list[Warning]:
        now = aware_utc(now)
        current = []
        for w in self._repo.find(domain=domain, status="active"):
            if w.expires > now:
                current.append(w)
            else:
                self._repo.set_status(w.id, "expired")
        return current

    def get(self, warning_id: str) -> Warning | None:
        return self._repo.get(warning_id)

    def history(self, domain: str | None, status: str | None, limit: int) -> list[Warning]:
        return self._repo.find(domain=domain, status=status, limit=limit)
