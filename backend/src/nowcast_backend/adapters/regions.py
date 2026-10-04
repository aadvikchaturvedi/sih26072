"""Regions (districts) from a GeoJSON file whose features carry ``id`` and ``name`` properties."""

from __future__ import annotations

import json
from pathlib import Path

from nowcast_backend.domain.models import Region


class GeoJsonRegions:
    def __init__(self, path: Path):
        features = json.loads(path.read_text())["features"]
        self._regions = [
            Region(
                id=str(f["properties"]["id"]),
                name=str(f["properties"].get("name", f["properties"]["id"])),
                geometry=f["geometry"],
            )
            for f in features
            if f.get("geometry") and f["geometry"]["type"] in ("Polygon", "MultiPolygon")
        ]

    def regions(self) -> list[Region]:
        return self._regions


class NoRegions:
    def regions(self) -> list[Region]:
        return []
