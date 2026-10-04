"""Fields -> RGBA PNG map overlays, plus the legends that describe them.

Images are in grid space (row 0 = north) and are meant to be stretched over the
domain ``bounds``; that is exact for regular lat/lon grids and approximate otherwise.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

RGBA = tuple[int, int, int, int]


@dataclass(frozen=True)
class Style:
    name: str
    units: str
    kind: Literal["step", "linear", "mask"]
    #: (value, colour). ``step``: colour applies from the value up to the next stop.
    #: ``linear``: colours are interpolated between stops. Below the first stop is transparent.
    stops: tuple[tuple[float, RGBA], ...]

    def legend(self) -> dict:
        return {
            "name": self.name,
            "units": self.units,
            "kind": self.kind,
            "stops": [{"value": v, "rgba": list(c)} for v, c in self.stops],
        }


def _opaque(*rgb: int) -> RGBA:
    return (*rgb, 255)


REFLECTIVITY = Style(
    "reflectivity",
    "dBZ",
    "step",
    (
        (5, (4, 233, 231, 160)),
        (10, (1, 159, 244, 190)),
        (15, (3, 0, 244, 210)),
        (20, _opaque(2, 253, 2)),
        (25, _opaque(1, 197, 1)),
        (30, _opaque(0, 142, 0)),
        (35, _opaque(253, 248, 2)),
        (40, _opaque(229, 188, 0)),
        (45, _opaque(253, 149, 0)),
        (50, _opaque(253, 0, 0)),
        (55, _opaque(212, 0, 0)),
        (60, _opaque(188, 0, 0)),
        (65, _opaque(248, 0, 253)),
        (70, _opaque(152, 84, 198)),
    ),
)
PROBABILITY = Style(
    "probability",
    "1",
    "linear",
    (
        (0.05, (255, 255, 178, 90)),
        (0.3, (254, 204, 92, 190)),
        (0.5, (253, 141, 60, 215)),
        (0.7, (240, 59, 32, 230)),
        (1.0, (189, 0, 38, 240)),
    ),
)
FLAG = Style("flag", "bool", "mask", ((1, (214, 0, 214, 220)),))
BRIGHTNESS_TEMPERATURE = Style(
    "brightness_temperature",
    "K",
    "linear",
    ((180, _opaque(255, 255, 255)), (320, _opaque(0, 0, 0))),
)
FLASH_DENSITY = Style(
    "flash_density",
    "flashes/km2/10min",
    "linear",
    ((1e-6, (255, 237, 160, 200)), (0.05, (254, 178, 76, 230)), (0.5, (240, 59, 32, 255))),
)

STYLES: dict[str, Style] = {
    s.name: s for s in (REFLECTIVITY, PROBABILITY, FLAG, BRIGHTNESS_TEMPERATURE, FLASH_DENSITY)
}


def style_for_unit(unit: str) -> Style | None:
    """Style for an observation channel, chosen by its physical unit."""
    return {
        "dBZ": REFLECTIVITY,
        "K": BRIGHTNESS_TEMPERATURE,
        "flashes/km2/10min": FLASH_DENSITY,
    }.get(unit)


def greyscale_style(field: np.ndarray, units: str) -> Style:
    """Fallback for channels without a dedicated palette: stretch over the data range."""
    finite = field[np.isfinite(field)]
    lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)
    return Style(
        "greyscale",
        units,
        "linear",
        ((lo, _opaque(0, 0, 0)), (max(hi, lo + 1e-6), _opaque(255, 255, 255))),
    )


def colorize(field: np.ndarray, style: Style) -> np.ndarray:
    """(H, W) values -> (H, W, 4) uint8. NaN and values below the first stop are transparent."""
    values = np.asarray(field, dtype=np.float64)
    stops = np.array([v for v, _ in style.stops], dtype=np.float64)
    colours = np.array([c for _, c in style.stops], dtype=np.float64)
    safe = np.nan_to_num(values, nan=-np.inf)
    if style.kind == "linear":
        rgba = np.stack([np.interp(safe, stops, colours[:, k]) for k in range(4)], axis=-1)
    else:
        rgba = colours[np.clip(np.searchsorted(stops, safe, side="right") - 1, 0, len(stops) - 1)]
    rgba[safe < stops[0]] = 0
    return rgba.round().astype(np.uint8)


def render_png(field: np.ndarray, style: Style, scale: int = 1) -> bytes:
    image = Image.fromarray(colorize(field, style), mode="RGBA")
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.NEAREST)
    out = io.BytesIO()
    image.save(out, format="PNG", optimize=True)
    return out.getvalue()
