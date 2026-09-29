"""SEVIR -> the same tensor layout as the India events.

SEVIR (Veillette et al., 2020) stores 4-h events (49 frames, 5-min) over a
384 km x 384 km patch:

=========  ===================  ======================  ======================
img_type   array                units (raw)             our channel
=========  ===================  ======================  ======================
vil        (N, 384, 384, 49)    digital VIL, uint8      ``maxz`` (approx., below)
ir107      (N, 192, 192, 49)    deg C x 100, int16      ``tir1_bt`` (K)
ir069      (N, 192, 192, 49)    deg C x 100, int16      ``wv_bt`` (K)
lght       per-event (n, 5)     [t_s, lat, lon, x, y]   flash points / ``flash_density``
=========  ===================  ======================  ======================

Derived: ``tir1_cooling`` (10-min TIR1 change) and ``tir1_minus_wv``. All other
channels are absent and therefore marked unavailable, so a SEVIR-pretrained
checkpoint has the *same input layout* as the India fine-tune.

**VIL -> dBZ is an approximation.** Digital VIL is decoded to kg m^-2 with the
SEVIR piecewise formula and inverted with the Greene & Clark (1972) relation
``VIL = 3.44e-6 * Z^(4/7) * D`` for an assumed uniform column depth ``D``
(default 5 km). Pretraining uses it to learn motion/growth; India fine-tuning
corrects the intensity mapping.

Checked on the real files (September 2026): IR raw units are deg C x 100; the
lightning table columns are [seconds from catalog ``time_utc``, lat, lon, x, y] with
x/y on a 48 x 48 grid; image arrays are stored **south-up** (row 0 = south) and are
flipped here; ``minute_offsets`` repeats a value where a frame is missing (SEVIR
fills gaps by repeating a frame), and those frames are masked as missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from nowcast_ml.data import channels as ch
from nowcast_ml.data.schema import build_event_dataset

N_FRAMES = 49
FRAME_MINUTES = 5
IR_SCALE = 100.0  # raw int16 = deg C * 100
SEVIR_EXTENT_KM = 384.0
REQUIRED_TYPES = ("vil", "ir107", "ir069")
DEFAULT_PROJ = "+proj=laea +lat_0=38 +lon_0=-98 +units=m +a=6370997.0 +ellps=sphere"
LGHT_GRID = 48  # SEVIR lightning x/y columns are pixels of a 48 x 48 (8 km) grid
#: How flashes are placed on the grid. Checked on 30 real events with
#: ``scripts/sevir_to_zarr.py --check-lightning``: with north-up images, "xy" and "latlon"
#: put ~90% of flash pixels on >= 30 dBZ echoes; the wrong orientation gives ~37%. "latlon"
#: is the default: its positions are continuous, while x/y are quantized to 8 km pixels.
LIGHTNING_MODE = "latlon"


def vil_digital_to_kgm2(x: np.ndarray) -> np.ndarray:
    """SEVIR digital VIL (0-255) -> VIL in kg m^-2."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros_like(x)
    m1 = (x > 5) & (x <= 18)
    m2 = x > 18
    out[m1] = (x[m1] - 2.0) / 90.66
    out[m2] = np.exp((x[m2] - 83.9) / 38.9)
    return out


def vil_to_dbz(vil_kgm2: np.ndarray, depth_m: float = 5000.0, floor_dbz: float = 0.0) -> np.ndarray:
    """Invert Greene & Clark VIL for a uniform column: equivalent reflectivity (dBZ)."""
    vil = np.asarray(vil_kgm2, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (vil / (3.44e-6 * depth_m)) ** (7.0 / 4.0)
        dbz = 10.0 * np.log10(z)
    dbz = np.where(vil > 0, dbz, floor_dbz)
    return np.clip(dbz, floor_dbz, 80.0)


def ir_raw_to_kelvin(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    return raw / IR_SCALE + 273.15


def _resize(a: np.ndarray, size: int) -> np.ndarray:
    """(T, H, W) -> (T, size, size) with area averaging (bilinear if upsampling)."""
    t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[:, None]
    mode = "area" if a.shape[-1] >= size else "bilinear"
    kw = {} if mode == "area" else {"align_corners": False}
    return F.interpolate(t, size=(size, size), mode=mode, **kw)[:, 0].numpy()


@dataclass
class SevirEntry:
    event_id: str
    time_utc: pd.Timestamp
    files: dict[str, tuple[str, int]]  # img_type -> (file_name, file_index)
    corners: tuple[float, float, float, float]  # llcrnrlat, llcrnrlon, urcrnrlat, urcrnrlon
    offsets: dict[str, list[int]] = field(default_factory=dict)  # img_type -> minute offsets
    proj: str = DEFAULT_PROJ


def read_catalog(
    root: str | Path, catalog: str = "CATALOG.csv", require_lightning: bool = False
) -> list[SevirEntry]:
    root = Path(root)
    df = pd.read_csv(root / catalog, low_memory=False)
    df = df[df["event_id"].notna()]
    need = set(REQUIRED_TYPES) | ({"lght"} if require_lightning else set())
    entries = []
    for eid, g in df.groupby("id"):
        types = set(g["img_type"])
        if not need <= types:
            continue
        files = {r.img_type: (str(r.file_name), int(r.file_index)) for r in g.itertuples()}
        offsets = {
            r.img_type: [int(v) for v in r.minute_offsets.split(":")]
            for r in g.itertuples()
            if isinstance(getattr(r, "minute_offsets", None), str) and r.minute_offsets
        }
        r0 = g.iloc[0]
        entries.append(
            SevirEntry(
                event_id=str(eid),
                time_utc=pd.Timestamp(r0["time_utc"]).tz_localize(None),
                files=files,
                corners=(
                    float(r0["llcrnrlat"]),
                    float(r0["llcrnrlon"]),
                    float(r0["urcrnrlat"]),
                    float(r0["urcrnrlon"]),
                ),
                offsets=offsets,
                proj=str(r0["proj"]) if isinstance(r0.get("proj"), str) else DEFAULT_PROJ,
            )
        )
    return sorted(entries, key=lambda e: e.event_id)


def _read_array(root: Path, file_name: str, img_type: str, index: int) -> np.ndarray:
    import h5py

    with h5py.File(root / file_name, "r") as f:
        return f[img_type][index]  # (H, W, 49)


def _read_lightning(root: Path, file_name: str, event_id: str) -> np.ndarray:
    import h5py

    with h5py.File(root / file_name, "r") as f:
        if event_id not in f:
            return np.zeros((0, 5))
        return f[event_id][:]


def load_sevir_event(
    root: str | Path,
    entry: SevirEntry,
    size: int = 128,
    frame_stride: int = 2,
    vil_depth_m: float = 5000.0,
    lightning_mode: str = LIGHTNING_MODE,
):
    """Load one SEVIR event from local HDF5 files as a contract-format ``xarray.Dataset``."""
    root = Path(root)
    vil = _read_array(root, *_fi(entry, "vil"))
    ir107 = _read_array(root, *_fi(entry, "ir107"))
    ir069 = _read_array(root, *_fi(entry, "ir069"))
    lght = (
        _read_lightning(root, entry.files["lght"][0], entry.event_id)
        if "lght" in entry.files
        else None
    )
    return sevir_event_from_arrays(
        entry, vil, ir107, ir069, lght, size, frame_stride, vil_depth_m, lightning_mode
    )


def _duplicate_frames(offsets: list[int] | None) -> np.ndarray:
    """(49,) bool: frame repeats the previous frame's time (SEVIR fills gaps by repetition)."""
    dup = np.zeros(N_FRAMES, bool)
    if offsets is not None and len(offsets) == N_FRAMES:
        o = np.asarray(offsets)
        dup[1:] = o[1:] <= o[:-1]
    return dup


def _flash_pixels(tbl: np.ndarray, grid: LaeaGrid, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Fractional (row, col) of each flash on the output grid.

    ``xy``: SEVIR's own projected pixel columns (48 x 48 grid of 8 km). Their y follows the
    stored (south-up) image rows, so it is flipped to match the north-up output grid.
    ``xy_noflip``: without that flip (matches only unflipped images; kept for the check).
    ``latlon``: project the flash lat/lons with the event's LAEA grid (exact, continuous).
    """
    H = W = grid.size
    if mode in ("xy", "xy_noflip"):
        col = (tbl[:, 3] + 0.5) * W / LGHT_GRID - 0.5
        yy = tbl[:, 4] if mode == "xy_noflip" else (LGHT_GRID - 1 - tbl[:, 4])
        row = (yy + 0.5) * H / LGHT_GRID - 0.5
        return row, col
    if mode == "latlon":
        return grid.pixel(tbl[:, 1], tbl[:, 2])
    raise ValueError(f"unknown lightning_mode {mode!r}")


def sevir_event_from_arrays(
    entry: SevirEntry,
    vil: np.ndarray,
    ir107: np.ndarray,
    ir069: np.ndarray,
    lght: np.ndarray | None,
    size: int = 128,
    frame_stride: int = 2,
    vil_depth_m: float = 5000.0,
    lightning_mode: str = LIGHTNING_MODE,
):
    """Raw SEVIR arrays ((H, W, 49) images, (n, 5) flashes) -> contract-format Dataset."""
    from scipy.ndimage import map_coordinates

    # (H, W, 49) -> (49, H, W), flipped north-up: SEVIR arrays are stored with row 0 = SOUTH
    # (verified on real data: flash lat/lons land on >= 30 dBZ echoes only after this flip).
    vil, ir107, ir069 = (np.moveaxis(np.asarray(a), -1, 0)[:, ::-1, :] for a in (vil, ir107, ir069))
    maxz = vil_to_dbz(_resize(vil_digital_to_kgm2(vil), size), vil_depth_m)
    tir1 = _resize(ir_raw_to_kelvin(ir107), size)
    wv = _resize(ir_raw_to_kelvin(ir069), size)
    cooling = np.full_like(tir1, np.nan)
    cooling[frame_stride:] = tir1[frame_stride:] - tir1[:-frame_stride]

    dup_r = _duplicate_frames(entry.offsets.get("vil"))
    dup_s = _duplicate_frames(entry.offsets.get("ir107")) | _duplicate_frames(
        entry.offsets.get("ir069")
    )
    dup_cool = dup_s.copy()
    dup_cool[frame_stride:] |= dup_s[:-frame_stride]

    # Subsample to 10-min frames, starting where cooling is defined.
    sel = np.arange(frame_stride, N_FRAMES, frame_stride)
    offsets_min = -120 + FRAME_MINUTES * sel
    times = pd.DatetimeIndex([entry.time_utc + pd.Timedelta(minutes=int(m)) for m in offsets_min])
    T = len(sel)

    grid = LaeaGrid(entry.proj, entry.corners, size)
    lat, lon = grid.latlon()
    grid_km = round(grid.dx / 1000.0, 3)
    flashes = None
    density = np.zeros((T, size, size), np.float32)
    if lght is not None and len(lght):
        tbl = np.asarray(lght, dtype=np.float64)
        row, col = _flash_pixels(tbl, grid, lightning_mode)
        inside = (row > -0.5) & (row < size - 0.5) & (col > -0.5) & (col < size - 0.5)
        tbl, row, col = tbl[inside], row[inside], col[inside]
        ft = entry.time_utc.to_datetime64() + (tbl[:, 0] * 1e9).astype("timedelta64[ns]")
        tv = times.values.astype("datetime64[ns]")
        ti = np.searchsorted(tv, ft, side="left")
        ok = (ti < T) & (ft > tv[0] - (tv[1] - tv[0]))
        yi, xi = np.round(row).astype(int), np.round(col).astype(int)
        np.add.at(density, (ti[ok], yi[ok], xi[ok]), 1.0)
        density /= grid_km**2
        coords = np.stack([row, col])
        fla = map_coordinates(lat, coords, order=1, mode="nearest")
        flo = map_coordinates(lon, coords, order=1, mode="nearest")
        flashes = (ft, fla, flo)

    fields = {
        "maxz": maxz[sel],
        "tir1_bt": tir1[sel],
        "wv_bt": wv[sel],
        "tir1_cooling": cooling[sel],
        "tir1_minus_wv": (tir1 - wv)[sel],
        "flash_density": density,
    }
    names = list(ch.SEVIR_CHANNELS)
    x = np.stack([fields[n] for n in names], axis=1).astype(np.float32)
    ones = np.ones((T, size, size), bool)
    missing = {
        "radar": ones * dup_r[sel][:, None, None],
        "satellite": ones * (dup_s[sel] | dup_cool[sel])[:, None, None],
        "lightning": ones * (lght is None),
    }
    # masked values must not be required to be finite; finite everywhere else
    for j, n in enumerate(names):
        g = ch.group_of(n)
        x[:, j][missing[g]] = np.nan
    return build_event_dataset(
        x=x,
        channels=names,
        times=times,
        lat=lat,
        lon=lon,
        missing=missing,
        event_id=f"sevir_{entry.event_id}",
        grid_spacing_km=grid_km,
        flashes=flashes,
        attrs={
            "source": "sevir",
            "maxz_note": f"VIL->dBZ approx (Greene-Clark, D={vil_depth_m} m)",
            "lightning_mode": lightning_mode,
        },
    )


def _fi(entry: SevirEntry, t: str) -> tuple[str, str, int]:
    fname, idx = entry.files[t]
    return fname, t, idx


class LaeaGrid:
    """Spherical Lambert azimuthal equal-area grid of a SEVIR patch (Snyder 1987, eqs. 24-2..24-19).

    Built from the catalog ``proj`` string and the lower-left / upper-right corner lat/lons,
    which are the outer corners of the square domain. Row 0 is the northern edge.
    """

    def __init__(self, proj: str, corners: tuple[float, float, float, float], size: int):
        p = dict(kv.split("=", 1) for kv in proj.replace("+", " ").split() if "=" in kv)
        if p.get("proj", "laea") != "laea":
            raise ValueError(f"unsupported SEVIR projection {proj!r}")
        self.lat0 = np.deg2rad(float(p.get("lat_0", 38.0)))
        self.lon0 = np.deg2rad(float(p.get("lon_0", -98.0)))
        self.R = float(p.get("a", 6370997.0))
        lla, llo, ura, uro = corners
        self.x0, self.y0 = self.forward(np.array([lla]), np.array([llo]))
        self.x1, self.y1 = self.forward(np.array([ura]), np.array([uro]))
        self.x0, self.y0, self.x1, self.y1 = (
            float(v[0]) for v in (self.x0, self.y0, self.x1, self.y1)
        )
        self.size = size
        self.dx = (self.x1 - self.x0) / size
        self.dy = (self.y1 - self.y0) / size

    def forward(self, lat, lon):
        phi, lam = np.deg2rad(lat), np.deg2rad(lon) - self.lon0
        k = np.sqrt(
            2.0
            / (1 + np.sin(self.lat0) * np.sin(phi) + np.cos(self.lat0) * np.cos(phi) * np.cos(lam))
        )
        x = self.R * k * np.cos(phi) * np.sin(lam)
        y = (
            self.R
            * k
            * (np.cos(self.lat0) * np.sin(phi) - np.sin(self.lat0) * np.cos(phi) * np.cos(lam))
        )
        return x, y

    def inverse(self, x, y):
        rho = np.hypot(x, y)
        c = 2 * np.arcsin(np.clip(rho / (2 * self.R), -1, 1))
        with np.errstate(invalid="ignore", divide="ignore"):
            phi = np.arcsin(
                np.cos(c) * np.sin(self.lat0)
                + np.where(rho > 0, y * np.sin(c) * np.cos(self.lat0) / rho, 0)
            )
        lam = self.lon0 + np.arctan2(
            x * np.sin(c), rho * np.cos(self.lat0) * np.cos(c) - y * np.sin(self.lat0) * np.sin(c)
        )
        return np.rad2deg(phi), np.rad2deg(lam)

    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """(lat, lon) of pixel centres, each (size, size), row 0 = north."""
        i = np.arange(self.size) + 0.5
        xs = self.x0 + i * self.dx
        ys = self.y1 - i * self.dy
        X, Y = np.meshgrid(xs, ys)
        return self.inverse(X, Y)

    def pixel(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        """Fractional (row, col) of lat/lon points (pixel centres at integers)."""
        x, y = self.forward(np.asarray(lat, float), np.asarray(lon, float))
        return (self.y1 - y) / self.dy - 0.5, (x - self.x0) / self.dx - 0.5


def sevir_event_openers(cfg_data, entries: list[SevirEntry], channels: list[str]):
    """Zero-arg, picklable callables that load each event lazily (inside DataLoader workers)."""
    s = cfg_data.sevir
    return [partial(_open_sevir, s.root, e, s.size, s.frame_stride, channels) for e in entries]


def _open_sevir(root, entry, size, frame_stride, channels):
    from nowcast_ml.data.event import Event

    return Event(load_sevir_event(root, entry, size, frame_stride), channels)


def sevir_n_frames(frame_stride: int = 2) -> int:
    return len(range(frame_stride, N_FRAMES, frame_stride))
