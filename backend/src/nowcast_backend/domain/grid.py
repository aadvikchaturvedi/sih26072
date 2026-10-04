"""The spatial grid of a domain (curvilinear lat/lon, row 0 = northern edge)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from nowcast_backend.domain.errors import InvalidRequest
from nowcast_backend.domain.models import Bounds

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    a = (
        np.sin((p2 - p1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.deg2rad(lon2 - lon1) / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


@dataclass(frozen=True)
class Grid:
    lat: np.ndarray  # (H, W) pixel-centre degrees
    lon: np.ndarray  # (H, W)
    spacing_km: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.lat.shape

    @property
    def pixel_area_km2(self) -> float:
        return self.spacing_km**2

    @property
    def bounds(self) -> Bounds:
        return Bounds(
            south=float(self.lat.min()),
            west=float(self.lon.min()),
            north=float(self.lat.max()),
            east=float(self.lon.max()),
        )

    def same_as(self, other: Grid) -> bool:
        return (
            self.shape == other.shape
            and abs(self.spacing_km - other.spacing_km) < 1e-6
            and np.allclose(self.lat, other.lat, atol=1e-5)
            and np.allclose(self.lon, other.lon, atol=1e-5)
        )

    def latlon_at(self, iy, ix) -> tuple[np.ndarray, np.ndarray]:
        """Lat/lon at fractional pixel coordinates (bilinear, clamped at the edges)."""
        coords = np.vstack([np.atleast_1d(iy), np.atleast_1d(ix)]).astype(np.float64)
        lat = map_coordinates(self.lat, coords, order=1, mode="nearest")
        lon = map_coordinates(self.lon, coords, order=1, mode="nearest")
        return lat, lon

    def contains_px(self, iy: float, ix: float) -> bool:
        H, W = self.shape
        return -0.5 <= iy <= H - 0.5 and -0.5 <= ix <= W - 0.5

    def nearest_index(self, lat: float, lon: float) -> tuple[int, int]:
        """Pixel containing a point; raises :class:`InvalidRequest` outside the domain."""
        d = haversine_km(self.lat, self.lon, lat, lon)
        iy, ix = np.unravel_index(int(np.argmin(d)), d.shape)
        if d[iy, ix] > self.spacing_km:
            b = self.bounds
            raise InvalidRequest(
                f"point ({lat}, {lon}) is outside the domain "
                f"(lat {b.south:.3f}..{b.north:.3f}, lon {b.west:.3f}..{b.east:.3f})"
            )
        return int(iy), int(ix)
