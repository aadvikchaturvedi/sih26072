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

Assumptions to verify on first real download (``@pytest.mark.sevir`` tests):
row 0 of the arrays is the northern edge; IR raw units are deg C x 100; the
lightning table columns are [seconds from catalog ``time_utc``, lat, lon, x, y].
"""

from __future__ import annotations

from dataclasses import dataclass
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
):
    """Load one SEVIR event as a contract-format ``xarray.Dataset`` on a ``size``^2 grid."""
    root = Path(root)
    vil = _read_array(
        root,
        *_fi(entry, "vil"),
    )
    ir107 = _read_array(root, *_fi(entry, "ir107"))
    ir069 = _read_array(root, *_fi(entry, "ir069"))
    vil, ir107, ir069 = (np.moveaxis(a, -1, 0) for a in (vil, ir107, ir069))  # (49, H, W)

    maxz = vil_to_dbz(_resize(vil_digital_to_kgm2(vil), size), vil_depth_m)
    tir1 = _resize(ir_raw_to_kelvin(ir107), size)
    wv = _resize(ir_raw_to_kelvin(ir069), size)
    cooling = np.full_like(tir1, np.nan)
    cooling[frame_stride:] = tir1[frame_stride:] - tir1[:-frame_stride]

    # Subsample to 10-min frames, starting where cooling is defined.
    sel = np.arange(frame_stride, N_FRAMES, frame_stride)
    offsets_min = -120 + FRAME_MINUTES * sel
    times = pd.DatetimeIndex([entry.time_utc + pd.Timedelta(minutes=int(m)) for m in offsets_min])

    lat, lon = _approx_latlon(entry.corners, size)
    flashes, density = None, np.zeros((len(sel), size, size), np.float32)
    grid_km = SEVIR_EXTENT_KM / size
    if "lght" in entry.files:
        tbl = _read_lightning(root, entry.files["lght"][0], entry.event_id)
        if len(tbl):
            ft = entry.time_utc.to_datetime64() + (tbl[:, 0] * 1e9).astype("timedelta64[ns]")
            flashes = (ft, tbl[:, 1], tbl[:, 2])
            density = _flash_density(ft, tbl[:, 1], tbl[:, 2], times, lat, lon) / grid_km**2

    fields = {
        "maxz": maxz[sel],
        "tir1_bt": tir1[sel],
        "wv_bt": wv[sel],
        "tir1_cooling": cooling[sel],
        "tir1_minus_wv": (tir1 - wv)[sel],
        "flash_density": density,
    }
    names = [c for c in ch.SEVIR_CHANNELS]
    x = np.stack([fields[n] for n in names], axis=1).astype(np.float32)
    missing = {g: np.zeros((len(sel), size, size), bool) for g in ("radar", "satellite")}
    missing["lightning"] = np.full((len(sel), size, size), "lght" not in entry.files)
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
        },
    )


def _fi(entry: SevirEntry, t: str) -> tuple[str, str, int]:
    fname, idx = entry.files[t]
    return fname, t, idx


def _approx_latlon(corners, size: int):
    """Linear lat/lon grid from catalog corners (adequate at 384 km). Row 0 = north."""
    lla, llo, ura, uro = corners
    lat = np.linspace(ura, lla, size)
    lon = np.linspace(llo, uro, size)
    lon2, lat2 = np.meshgrid(lon, lat)
    return lat2, lon2


def _flash_density(ft, fla, flo, times, lat, lon) -> np.ndarray:
    """Counts per (frame, pixel); frame t covers (times[t-1], times[t]]."""
    T, (H, W) = len(times), lat.shape
    out = np.zeros((T, H, W), np.float32)
    tv = times.values.astype("datetime64[ns]")
    ti = np.searchsorted(tv, ft, side="left")
    ok = (ti < T) & (ft > tv[0] - (tv[1] - tv[0]))
    yi = np.round((lat[0, 0] - fla) / (lat[0, 0] - lat[-1, 0]) * (H - 1)).astype(int)
    xi = np.round((flo - lon[0, 0]) / (lon[0, -1] - lon[0, 0]) * (W - 1)).astype(int)
    ok &= (yi >= 0) & (yi < H) & (xi >= 0) & (xi < W)
    np.add.at(out, (ti[ok], yi[ok], xi[ok]), 1.0)
    return out


def sevir_event_openers(cfg_data, entries: list[SevirEntry], channels: list[str]):
    """Zero-arg callables that load each event lazily (inside DataLoader workers)."""
    from nowcast_ml.data.event import Event

    s = cfg_data.sevir
    return [
        (lambda e=e: Event(load_sevir_event(s.root, e, s.size, s.frame_stride), channels))
        for e in entries
    ]


def sevir_n_frames(frame_stride: int = 2) -> int:
    return len(range(frame_stride, N_FRAMES, frame_stride))
