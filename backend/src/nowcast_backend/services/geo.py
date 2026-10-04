"""Raster masks -> GeoJSON geometry on a domain grid."""

from __future__ import annotations

import cv2
import numpy as np

from nowcast_backend.domain.grid import Grid


def mask_polygon(mask: np.ndarray, grid: Grid, simplify_px: float = 0.5) -> dict | None:
    """Outline of a boolean mask as a GeoJSON Polygon / MultiPolygon (holes are filled).

    The mask is traced at 2x resolution so outlines follow pixel edges rather than
    pixel centres, and a single pixel still gives a valid polygon.
    """
    big = np.repeat(np.repeat(mask.astype(np.uint8), 2, axis=0), 2, axis=1)
    contours, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        pts = cv2.approxPolyDP(contour, 2 * simplify_px, True)[:, 0, :].astype(np.float64)
        if len(pts) < 3:
            pts = contour[:, 0, :].astype(np.float64)
        if len(pts) < 3:
            continue
        # Contour points are boundary sub-pixels: an even index is the low half of a pixel
        # (its low edge is the outline), an odd index the high half. ceil(p / 2) - 0.5 maps
        # both onto that pixel edge in original pixel coordinates.
        lat, lon = grid.latlon_at(np.ceil(pts[:, 1] / 2) - 0.5, np.ceil(pts[:, 0] / 2) - 0.5)
        ring = [[round(float(x), 5), round(float(y), 5)] for x, y in zip(lon, lat, strict=True)]
        ring.append(ring[0])
        polygons.append([ring])
    if not polygons:
        return None
    if len(polygons) == 1:
        return {"type": "Polygon", "coordinates": polygons[0]}
    return {"type": "MultiPolygon", "coordinates": polygons}


def polygons(geometry: dict) -> list[list[list[list[float]]]]:
    """The polygons of a Polygon / MultiPolygon geometry, each a list of rings."""
    return [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]


def rings(geometry: dict) -> list[list[list[float]]]:
    """Exterior rings of a Polygon / MultiPolygon geometry."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"][0]]
    return [poly[0] for poly in geometry["coordinates"]]


def bbox(geometry: dict) -> tuple[float, float, float, float]:
    """(west, south, east, north)."""
    pts = np.array([p for ring in rings(geometry) for p in ring])
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


def bboxes_intersect(a: dict, b: dict) -> bool:
    aw, as_, ae, an = bbox(a)
    bw, bs, be, bn = bbox(b)
    return aw <= be and bw <= ae and as_ <= bn and bs <= an


def feature_collection(items, geometry_key: str = "polygon") -> dict:
    """Pydantic models with a geometry field -> GeoJSON FeatureCollection."""
    features = []
    for item in items:
        props = item.model_dump(mode="json")
        geometry = props.pop(geometry_key)
        features.append(
            {"type": "Feature", "id": props.get("id"), "geometry": geometry, "properties": props}
        )
    return {"type": "FeatureCollection", "features": features}
