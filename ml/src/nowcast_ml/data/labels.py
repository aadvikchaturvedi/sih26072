"""Lightning labels.

Definitions (all on the model grid, 10-min frames; frame ``t`` covers the
interval ``(t - 10 min, t]``):

* ``occurrence[t, y, x]`` - at least one flash in pixel (y, x) during frame t.
* Label for lead L at forecast time t0: 1 if any flash falls within
  ``radius_km`` (default 10 km) of the pixel during ``(t0, t0 + L]``, i.e. frames
  ``t0+1 .. t0+L/10``. Computed as ``any`` over time, then a disk max-pool
  (binary dilation) over space.
* ``past_lightning`` at t0: same, over ``(t0 - 30 min, t0]`` (frames t0-2..t0);
  used for the "first flash" product and the "lightning persists" baseline.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def radius_px(radius_km: float, grid_spacing_km: float) -> int:
    return max(0, int(np.floor(radius_km / grid_spacing_km + 1e-6)))


def disk(radius: int) -> np.ndarray:
    r = int(radius)
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (yy**2 + xx**2) <= r * r


def dilate(field: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation of the last two axes with a disk of ``radius`` pixels."""
    field = np.asarray(field, dtype=bool)
    if radius <= 0:
        return field.copy()
    fp = disk(radius)
    if field.ndim == 2:
        return binary_dilation(field, structure=fp)
    out = np.empty_like(field)
    for idx in np.ndindex(field.shape[:-2]):
        out[idx] = binary_dilation(field[idx], structure=fp)
    return out


def window_any(occurrence: np.ndarray, start: int, stop: int) -> np.ndarray:
    """``any`` over frames [start, stop) clipped to the valid range -> (H, W) bool."""
    start = max(start, 0)
    stop = min(stop, occurrence.shape[0])
    if stop <= start:
        return np.zeros(occurrence.shape[1:], dtype=bool)
    return occurrence[start:stop].any(axis=0)


def lightning_labels(
    occurrence: np.ndarray,
    t0: int,
    leads_min: list[int] | tuple[int, ...] = (30, 60),
    step_minutes: int = 10,
    radius: int = 5,
) -> np.ndarray:
    """(n_leads, H, W) float32 binary labels for forecast time index ``t0``."""
    out = []
    for lead in leads_min:
        n = lead // step_minutes
        out.append(dilate(window_any(occurrence, t0 + 1, t0 + 1 + n), radius))
    return np.stack(out).astype(np.float32)


def lightning_label_valid(
    lightning_missing: np.ndarray,
    t0: int,
    leads_min: list[int] | tuple[int, ...] = (30, 60),
    step_minutes: int = 10,
) -> np.ndarray:
    """(n_leads, H, W) bool: label is trustworthy (no missing lightning data in its window)."""
    T = lightning_missing.shape[0]
    out = []
    for lead in leads_min:
        n = lead // step_minutes
        if t0 + n >= T:
            out.append(np.zeros(lightning_missing.shape[1:], dtype=bool))
        else:
            out.append(~window_any(lightning_missing, t0 + 1, t0 + 1 + n))
    return np.stack(out)


def past_lightning(
    occurrence: np.ndarray, t0: int, window_min: int = 30, step_minutes: int = 10, radius: int = 5
) -> np.ndarray:
    """(H, W) bool: any flash within ``radius`` px during (t0 - window, t0]."""
    n = window_min // step_minutes
    return dilate(window_any(occurrence, t0 - n + 1, t0 + 1), radius)


def occurrence_from_points(
    flash_times: np.ndarray,
    flash_lat: np.ndarray,
    flash_lon: np.ndarray,
    times: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    max_dist_px: float = 1.0,
) -> np.ndarray:
    """Grid a flash point table onto (T, H, W) bool occurrence.

    Flash at time ``ft`` belongs to the frame ``t`` with ``times[t-1] < ft <= times[t]``.
    Points further than ``max_dist_px`` grid cells from any pixel centre are dropped
    (outside the domain).
    """
    from scipy.spatial import cKDTree

    T = len(times)
    H, W = lat.shape
    occ = np.zeros((T, H, W), dtype=bool)
    if len(flash_times) == 0:
        return occ
    times = np.asarray(times, dtype="datetime64[ns]")
    ft = np.asarray(flash_times, dtype="datetime64[ns]")
    ti = np.searchsorted(times, ft, side="left")
    keep = (ti < T) & (ft > times[0] - (times[1] - times[0] if T > 1 else np.timedelta64(600, "s")))
    coslat = np.cos(np.deg2rad(np.nanmean(lat)))
    tree = cKDTree(np.column_stack([lat.ravel(), lon.ravel() * coslat]))
    d, idx = tree.query(np.column_stack([flash_lat, np.asarray(flash_lon) * coslat]))
    # grid cell size in degrees (approx) for the domain check
    cell = np.median(np.abs(np.diff(lat[:, W // 2]))) if H > 1 else 1.0
    keep &= d <= max_dist_px * cell * 1.5
    yi, xi = np.unravel_index(idx[keep], (H, W))
    occ[ti[keep], yi, xi] = True
    return occ
