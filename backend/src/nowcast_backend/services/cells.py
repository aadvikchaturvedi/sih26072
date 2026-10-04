"""Storm cells: identification, tracking and short extrapolated tracks.

A cell is a connected region of reflectivity at or above a threshold. Cells are
linked between consecutive frames by nearest centroid within the distance a
storm can travel in one step; the resulting displacement gives speed and
direction. Ids are carried over from the previous forecast's cells so a storm
keeps its id from run to run.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from nowcast_backend.domain.grid import Grid, haversine_km
from nowcast_backend.domain.models import StormCell, TrackPoint
from nowcast_backend.domain.timeutil import aware_utc, stamp
from nowcast_backend.services.geo import mask_polygon
from nowcast_backend.settings import CellSettings

_EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)


@dataclass
class Blob:
    iy: float
    ix: float
    area_px: int
    max_dbz: float
    mask: np.ndarray = field(repr=False)
    prev: Blob | None = field(default=None, repr=False)
    next: Blob | None = field(default=None, repr=False)


def find_blobs(field_dbz: np.ndarray, threshold: float, min_px: int) -> list[Blob]:
    values = np.nan_to_num(field_dbz, nan=-np.inf)
    labels, n = ndimage.label(values >= threshold, structure=_EIGHT_CONNECTED)
    blobs = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        mask = labels == i
        area = int(mask[sl].sum())
        if area < min_px:
            continue
        # intensity-weighted centroid, so the position follows the core
        iy, ix = ndimage.center_of_mass(np.where(mask, values - threshold + 1.0, 0.0))
        blobs.append(Blob(float(iy), float(ix), area, float(values[mask].max()), mask))
    return blobs


def link(before: list[Blob], after: list[Blob], max_px: float) -> None:
    """Greedy one-to-one nearest-centroid matching between two consecutive frames."""
    pairs = sorted(
        (np.hypot(a.iy - b.iy, a.ix - b.ix), i, j)
        for i, a in enumerate(before)
        for j, b in enumerate(after)
    )
    used_a, used_b = set(), set()
    for dist, i, j in pairs:
        if dist > max_px:
            break
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        before[i].next, after[j].prev = after[j], before[i]


def _chain(blob: Blob, frame: int) -> list[tuple[int, Blob]]:
    """All linked positions of a cell as (frame index, blob), oldest first."""
    back, b, i = [], blob.prev, frame - 1
    while b is not None:
        back.append((i, b))
        b, i = b.prev, i - 1
    fwd, b, i = [], blob.next, frame + 1
    while b is not None:
        fwd.append((i, b))
        b, i = b.next, i + 1
    return [*reversed(back), (frame, blob), *fwd]


def _window_max(field_2d: np.ndarray, iy: float, ix: float, radius: int) -> float | None:
    H, W = field_2d.shape
    y, x = int(round(iy)), int(round(ix))
    win = field_2d[
        max(0, y - radius) : min(H, y + radius + 1), max(0, x - radius) : min(W, x + radius + 1)
    ]
    return float(np.nanmax(win)) if win.size and np.isfinite(win).any() else None


def analyse(
    forecast: xr.Dataset,
    observed: list[tuple[pd.Timestamp, np.ndarray]],
    grid: Grid,
    cfg: CellSettings,
    step_minutes: int,
    previous: list[StormCell] | None = None,
) -> list[StormCell]:
    """Cells at the forecast's analysis time.

    ``observed`` holds the radar frames up to t0 (oldest first, NaN where missing).
    When the t0 frame has no radar the cells are taken from the +10 min forecast and
    tracked through the following leads instead.
    """
    t0 = pd.Timestamp(forecast.attrs["t0"]).tz_localize(None)
    leads = [int(v) for v in forecast["lead"].values]
    refl = forecast["reflectivity"].values
    min_px = max(1, int(np.ceil(cfg.min_area_km2 / grid.pixel_area_km2)))
    max_px = cfg.max_speed_kmh * step_minutes / 60.0 / grid.spacing_km + 1.0

    if observed and np.isfinite(observed[-1][1]).any():
        basis, frames, ref = "observed", observed, len(observed) - 1
    else:
        n = min(len(leads), 4)
        frames = [(t0 + pd.Timedelta(minutes=leads[i]), refl[i]) for i in range(n)]
        basis, ref = "forecast", 0
    blobs = [find_blobs(f, cfg.threshold_dbz, min_px) for _, f in frames]
    for before, after in zip(blobs, blobs[1:], strict=False):
        link(before, after, max_px)

    radius = max(1, int(round(15.0 / grid.spacing_km)))  # search radius around a track point
    cells = []
    for blob in sorted(blobs[ref], key=lambda b: -b.max_dbz):
        chain = _chain(blob, ref)
        (i_a, a), (i_b, b) = chain[0], chain[-1]
        hours = (frames[i_b][0] - frames[i_a][0]).total_seconds() / 3600.0
        vy, vx = ((b.iy - a.iy) / hours, (b.ix - a.ix) / hours) if hours > 0 else (0.0, 0.0)
        speed = direction = None
        if hours > 0:
            speed = float(np.hypot(vy, vx) * grid.spacing_km)
            direction = float(np.degrees(np.arctan2(vx, -vy)) % 360.0)  # rows grow southward

        time = frames[ref][0]
        track = []
        for k, lead in enumerate(leads):
            valid = t0 + pd.Timedelta(minutes=lead)
            if lead > cfg.track_minutes or valid <= time:
                continue
            dt = (valid - time).total_seconds() / 3600.0
            iy, ix = blob.iy + vy * dt, blob.ix + vx * dt
            if not grid.contains_px(iy, ix):
                break
            lat, lon = grid.latlon_at(iy, ix)
            track.append(
                TrackPoint(
                    time=aware_utc(valid),
                    lat=float(lat[0]),
                    lon=float(lon[0]),
                    max_dbz=_window_max(refl[k], iy, ix, radius),
                )
            )
        history = []
        for i, past in chain:
            if i < ref:
                lat, lon = grid.latlon_at(past.iy, past.ix)
                history.append(
                    TrackPoint(
                        time=aware_utc(frames[i][0]),
                        lat=float(lat[0]),
                        lon=float(lon[0]),
                        max_dbz=past.max_dbz,
                    )
                )
        trend = "steady"
        ahead = [p.max_dbz for p in track[:3] if p.max_dbz is not None]
        if ahead:
            change = ahead[-1] - blob.max_dbz
            trend = (
                "growing"
                if change >= cfg.trend_dbz
                else "decaying"
                if change <= -cfg.trend_dbz
                else "steady"
            )
        near = ndimage.binary_dilation(blob.mask, iterations=radius // 2 or 1)
        lightning = {
            name.rsplit("_", 1)[1]: round(float(forecast[name].values[near].max()), 3)
            for name in forecast.data_vars
            if name.startswith("lightning_prob_")
        }
        lat, lon = grid.latlon_at(blob.iy, blob.ix)
        cells.append(
            StormCell(
                id="",
                time=aware_utc(time),
                basis=basis,
                lat=float(lat[0]),
                lon=float(lon[0]),
                area_km2=blob.area_px * grid.pixel_area_km2,
                max_dbz=blob.max_dbz,
                speed_kmh=speed,
                direction_deg=direction,
                trend=trend,
                lightning_prob=lightning,
                polygon=mask_polygon(blob.mask, grid),
                history=history,
                forecast_track=track,
            )
        )
    _assign_ids(cells, previous or [], t0, cfg)
    return cells


def _assign_ids(cells: list[StormCell], previous: list[StormCell], t0, cfg: CellSettings) -> None:
    """Inherit ids from the previous run's cells (advected to now); number the rest."""
    candidates = []
    for i, cell in enumerate(cells):
        for j, old in enumerate(previous):
            hours = (cell.time - old.time).total_seconds() / 3600.0
            if hours <= 0:
                continue
            # where the old cell should be now: its track point at this time, else where it was
            lat, lon = next(
                ((p.lat, p.lon) for p in old.forecast_track if p.time == cell.time),
                (old.lat, old.lon),
            )
            dist = float(haversine_km(cell.lat, cell.lon, lat, lon))
            if dist <= cfg.max_speed_kmh * hours:
                candidates.append((dist, i, j))
    taken_new, taken_old = set(), set()
    for _, i, j in sorted(candidates):
        if i in taken_new or j in taken_old:
            continue
        taken_new.add(i)
        taken_old.add(j)
        cells[i].id = previous[j].id
    used = {c.id for c in cells}
    n = 0
    for cell in cells:
        while not cell.id:
            n += 1
            candidate = f"{stamp(t0)}-{n:02d}"
            if candidate not in used:
                cell.id = candidate
                used.add(candidate)
