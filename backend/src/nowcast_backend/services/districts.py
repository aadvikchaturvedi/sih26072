"""District summaries of a forecast, and the level-change warnings that follow from them."""

from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np
import pandas as pd
import xarray as xr

from nowcast_backend.domain.grid import Grid
from nowcast_backend.domain.models import (
    AffectedRegion,
    DistrictForecast,
    DistrictWarning,
    Region,
    StormCell,
)
from nowcast_backend.domain.timeutil import aware_utc, stamp
from nowcast_backend.services.geo import polygons
from nowcast_backend.settings import DistrictSettings


class RegionIndex:
    """Regions rasterised onto a grid: ``labels[y, x]`` is 1 + the region's position, 0 = none."""

    def __init__(self, regions: list[Region], grid: Grid):
        self.regions = regions
        self.labels = np.zeros(grid.shape, dtype=np.int32)
        for n, region in enumerate(regions, start=1):
            for rings in polygons(region.geometry):
                exterior = np.asarray(rings[0], dtype=np.float64)
                rows, cols = grid.pixel_of(exterior[:, 1], exterior[:, 0])
                pts = np.round(np.column_stack([cols, rows])).astype(np.int32)
                cv2.fillPoly(self.labels, [pts], n)

    def region_at(self, grid: Grid, lat: float, lon: float) -> Region | None:
        row, col = (int(round(float(v[0]))) for v in grid.pixel_of(lat, lon))
        H, W = self.labels.shape
        if not (0 <= row < H and 0 <= col < W) or self.labels[row, col] == 0:
            return None
        return self.regions[self.labels[row, col] - 1]


def affected_regions(cell: StormCell, index: RegionIndex, grid: Grid) -> list[AffectedRegion]:
    """Regions under the cell now or along its extrapolated track, with arrival times."""
    out: dict[str, AffectedRegion] = {}
    points = [(0, cell.lat, cell.lon)] + [
        (int(round((p.time - cell.time).total_seconds() / 60)), p.lat, p.lon)
        for p in cell.forecast_track
    ]
    for eta, lat, lon in points:
        region = index.region_at(grid, lat, lon)
        if region is not None and region.id not in out:
            out[region.id] = AffectedRegion(region_id=region.id, name=region.name, eta_min=eta)
    return list(out.values())


def assess(
    forecast: xr.Dataset,
    index: RegionIndex,
    cells: Iterable[StormCell],
    cfg: DistrictSettings,
) -> list[DistrictForecast]:
    """One summary per region that lies (at least partly) on the grid."""
    leads = np.array([int(v) for v in forecast["lead"].values])
    refl = np.nan_to_num(forecast["reflectivity"].values[leads <= cfg.valid_minutes], nan=0.0)
    peak = refl.max(axis=0) if len(refl) else np.zeros(index.labels.shape)

    def prob(lead: int) -> np.ndarray:
        name = f"lightning_prob_{lead}"
        return forecast[name].values if name in forecast else np.zeros(index.labels.shape)

    p30, p60 = prob(30), prob(60)
    first_flash = forecast["first_flash"].values.astype(bool)
    # the strongest cell with a lightning jump heading for each region
    jumping = {
        a.region_id: cell.id
        for cell in sorted(cells, key=lambda c: c.max_dbz)
        if cell.lightning_jump
        for a in cell.affected
    }

    out = []
    for n, region in enumerate(index.regions, start=1):
        inside = index.labels == n
        if not inside.any():
            continue
        prob_30, prob_60 = float(p30[inside].max()), float(p60[inside].max())
        max_dbz = float(peak[inside].max())
        new_storm = bool(first_flash[inside].any())
        probability = max(prob_30, prob_60)
        jump_cell = jumping.get(region.id)

        # The trigger reported with every level is the lightning probability against the
        # probability threshold of that level; ``rule`` names the condition that decided it.
        orange = probability > cfg.prob_orange or jump_cell is not None
        threshold = cfg.prob_orange
        if orange and max_dbz >= cfg.dbz_red:
            level, rule = "red", f"orange + core >= {cfg.dbz_red:g} dBZ"
        elif orange and new_storm:
            level, rule = "red", "orange + first-flash flag"
        elif probability > cfg.prob_orange:
            level, rule = "orange", f"lightning probability > {cfg.prob_orange:.0%}"
        elif orange:
            level, rule = "orange", "lightning jump in approaching cell"
        elif probability >= cfg.prob_yellow:
            level, rule = "yellow", f"lightning probability >= {cfg.prob_yellow:.0%}"
            threshold = cfg.prob_yellow
        elif max_dbz >= cfg.dbz_yellow:
            level, rule = "yellow", f"forecast MAX-Z >= {cfg.dbz_yellow:g} dBZ"
            threshold = cfg.prob_yellow
        else:
            level, rule, threshold = "green", "no significant threat", cfg.prob_yellow
        value = probability
        out.append(
            DistrictForecast(
                district_id=region.id,
                name=region.name,
                lightning_prob_30=round(prob_30, 3),
                lightning_prob_60=round(prob_60, 3),
                max_dbz=round(max_dbz, 1),
                area_fraction_above_threshold=round(
                    float((peak[inside] >= cfg.area_threshold_dbz).mean()), 3
                ),
                first_flash=new_storm,
                level=level,
                rule=rule,
                trigger_value=round(value, 3),
                trigger_threshold=threshold,
                cause_cell=jump_cell or _nearest_cell(region.id, cells),
            )
        )
    return out


def _nearest_cell(region_id: str, cells: Iterable[StormCell]) -> str | None:
    """The cell that reaches the region first, if any is heading there."""
    arrivals = [(a.eta_min, c.id) for c in cells for a in c.affected if a.region_id == region_id]
    return min(arrivals)[1] if arrivals else None


_COMPASS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def compass(bearing_deg: float) -> str:
    return _COMPASS[int(((bearing_deg % 360) + 22.5) // 45) % 8]


def _cause(district: DistrictForecast, cells: dict[str, StormCell]) -> str:
    cell = cells.get(district.cause_cell or "")
    if cell is None:
        probability = max(district.lightning_prob_30, district.lightning_prob_60)
        return f"District-level lightning probability {probability:.0%}"
    text = f"Cell {cell.id}"
    if cell.speed_kmh is not None and cell.direction_deg is not None:
        text += f" moving {compass(cell.direction_deg)} at {cell.speed_kmh:.0f} km/h"
    if cell.lightning_jump and cell.lightning_jump_at:
        text += f", lightning jump {cell.lightning_jump_at:%H:%M} UTC"
    return text


def level_changes(
    runs: Iterable[tuple[pd.Timestamp, list[DistrictForecast], list[StormCell]]],
    valid_minutes: int,
) -> list[DistrictWarning]:
    """Walk forecasts oldest-first and turn each change of a district's level into a warning.

    A district that starts out green is not reported; a return to green is, since it
    lifts the earlier warning. Ids depend only on the time and the district, so the
    same forecasts always give the same warnings.
    """
    level: dict[str, str] = {}
    last_id: dict[str, str] = {}
    out = []
    for t0, districts, cells in runs:
        by_id = {c.id: c for c in cells}
        for d in districts:
            before = level.get(d.district_id)
            level[d.district_id] = d.level
            if before == d.level or (before is None and d.level == "green"):
                continue
            warning = DistrictWarning(
                id=f"W-{stamp(t0)}-{d.district_id}",
                level=d.level,
                previous_level=before,
                supersedes=last_id.get(d.district_id),
                district_id=d.district_id,
                district_name=d.name,
                cause_text=_cause(d, by_id),
                cause_cell=d.cause_cell,
                rule=d.rule,
                trigger_value=d.trigger_value,
                trigger_threshold=d.trigger_threshold,
                issued_at=aware_utc(t0),
                valid_until=aware_utc(t0 + pd.Timedelta(minutes=valid_minutes)),
            )
            last_id[d.district_id] = warning.id
            out.append(warning)
    return out
