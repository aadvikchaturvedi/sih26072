"""INPUT CONTRACT for event Zarr stores (and for ``Predictor.predict`` inputs).

An event store ``data/events/<event_id>.zarr`` is an ``xarray.Dataset`` with:

========================  ===============================  ==============================
name                      dims                             notes
========================  ===============================  ==============================
``x``                     (time, channel, y, x) float32    physical units (see channels.py)
``missing``               (time, group, y, x) uint8        1 = missing; groups from GROUPS
``time`` (coord)          (time,) datetime64[ns]           UTC, strictly 10-min spacing
``channel`` (coord)       (channel,) str                   names from channels.py
``group`` (coord)         (group,) str                     radar/satellite/lightning/nwp
``lat``, ``lon`` (coord)  (y, x) float64                   degrees; row 0 = northern edge
``flash_time``            (flash,) datetime64[ns]          optional lightning point table
``flash_lat/flash_lon``   (flash,) float64                 optional lightning point table
========================  ===============================  ==============================

Required attrs: ``event_id``, ``grid_spacing_km`` (2.0 for India), ``contract_version``.

Missing data is *masked*, never zero-filled: where ``missing == 1`` the value of
``x`` is ignored (NaN recommended); where ``missing == 0`` it must be finite.
Missing frames are included as frames with ``missing == 1`` so the time axis
stays regular. Channels may be absent altogether; models treat them as missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from nowcast_ml.data import channels as ch

CONTRACT_VERSION = "1.0"
X_VAR = "x"
MISSING_VAR = "missing"
X_DIMS = ("time", "channel", "y", "x")
MISSING_DIMS = ("time", "group", "y", "x")
FLASH_VARS = ("flash_time", "flash_lat", "flash_lon")
REQUIRED_ATTRS = ("event_id", "grid_spacing_km", "contract_version")
STEP = pd.Timedelta(minutes=10)
EARTH_RADIUS_KM = 6371.0


def _haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dphi = p2 - p1
    dlmb = np.deg2rad(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def validate_dataset(
    ds: xr.Dataset,
    *,
    expected_spacing_km: float | None = 2.0,
    spacing_tol: float = 0.1,
    check_values: bool = True,
    min_frames: int = 1,
) -> list[str]:
    """Return human-readable contract violations (empty list = valid)."""
    problems: list[str] = []

    for a in REQUIRED_ATTRS:
        if a not in ds.attrs:
            problems.append(f"missing global attribute {a!r}")
    if (
        "contract_version" in ds.attrs
        and str(ds.attrs["contract_version"]).split(".")[0] != CONTRACT_VERSION.split(".")[0]
    ):
        problems.append(
            f"contract_version {ds.attrs['contract_version']!r} incompatible with {CONTRACT_VERSION!r}"
        )

    if X_VAR not in ds:
        problems.append(f"missing data variable {X_VAR!r}")
        return problems
    x = ds[X_VAR]
    if tuple(x.dims) != X_DIMS:
        problems.append(f"{X_VAR!r} dims are {tuple(x.dims)}, expected {X_DIMS}")
        return problems
    if x.dtype != np.float32:
        problems.append(f"{X_VAR!r} dtype is {x.dtype}, expected float32")

    # --- channel coordinate
    if "channel" not in ds.coords:
        problems.append("missing coordinate 'channel'")
        chans: list[str] = []
    else:
        chans = [str(c) for c in ds["channel"].values]
        problems += [f"channel coordinate: {p}" for p in ch.validate_channel_list(chans)]

    # --- time coordinate
    if "time" not in ds.coords:
        problems.append("missing coordinate 'time'")
    else:
        t = ds["time"].values
        if not np.issubdtype(t.dtype, np.datetime64):
            problems.append(f"time coordinate dtype is {t.dtype}, expected datetime64")
        else:
            if len(t) < min_frames:
                problems.append(f"only {len(t)} frames, need at least {min_frames}")
            if pd.isnull(t).any():
                problems.append("time coordinate contains NaT")
            elif len(t) > 1:
                d = np.diff(t).astype("timedelta64[s]").astype(np.int64)
                bad = np.nonzero(d != STEP.total_seconds())[0]
                if len(bad):
                    i = int(bad[0])
                    problems.append(
                        f"time spacing must be 10 min; {len(bad)} irregular step(s), first between "
                        f"{pd.Timestamp(t[i])} and {pd.Timestamp(t[i + 1])} (include missing frames with missing=1)"
                    )
            tz = ds["time"].attrs.get("timezone") or ds["time"].encoding.get("units", "")
            if isinstance(tz, str) and ("+05:30" in tz or "IST" in tz):
                problems.append("time appears to be IST; the contract requires UTC")

    # --- lat / lon
    for c in ("lat", "lon"):
        if c not in ds.coords:
            problems.append(f"missing coordinate {c!r}")
        elif tuple(ds[c].dims) != ("y", "x"):
            problems.append(f"coordinate {c!r} dims are {tuple(ds[c].dims)}, expected ('y', 'x')")
    if "lat" in ds.coords and "lon" in ds.coords and ds["lat"].dims == ("y", "x"):
        lat, lon = ds["lat"].values, ds["lon"].values
        if lat.shape[0] > 1 and lat[0].mean() < lat[-1].mean():
            problems.append("row 0 must be the northern edge (lat should decrease with y)")
        if expected_spacing_km is not None and lat.shape[0] > 1 and lat.shape[1] > 1:
            cy, cx = lat.shape[0] // 2, lat.shape[1] // 2
            dy = _haversine_km(lat[cy, cx], lon[cy, cx], lat[cy - 1, cx], lon[cy - 1, cx])
            dx = _haversine_km(lat[cy, cx], lon[cy, cx], lat[cy, cx - 1], lon[cy, cx - 1])
            for name, d in (("y", dy), ("x", dx)):
                if abs(d - expected_spacing_km) > spacing_tol * expected_spacing_km:
                    problems.append(
                        f"grid spacing along {name} is {d:.2f} km, expected {expected_spacing_km} km"
                    )
        attr_sp = ds.attrs.get("grid_spacing_km")
        if (
            expected_spacing_km is not None
            and attr_sp is not None
            and abs(float(attr_sp) - expected_spacing_km) > 1e-6
        ):
            problems.append(f"attr grid_spacing_km={attr_sp}, expected {expected_spacing_km}")

    # --- missing mask
    if MISSING_VAR not in ds:
        problems.append(f"missing data variable {MISSING_VAR!r} (per-group missing mask)")
        miss = None
    else:
        miss = ds[MISSING_VAR]
        if tuple(miss.dims) != MISSING_DIMS:
            problems.append(f"{MISSING_VAR!r} dims are {tuple(miss.dims)}, expected {MISSING_DIMS}")
            miss = None
        else:
            groups = [str(g) for g in ds["group"].values] if "group" in ds.coords else []
            unknown = [g for g in groups if g not in ch.GROUPS]
            if unknown:
                problems.append(f"unknown group(s) in 'group' coordinate: {unknown}")
            need = sorted({ch.group_of(c) for c in chans if c in ch.CHANNELS})
            lacking = [g for g in need if g not in groups]
            if lacking:
                problems.append(f"'missing' lacks group(s) {lacking} for the channels present")
            if not (np.issubdtype(miss.dtype, np.integer) or miss.dtype == bool):
                problems.append(f"{MISSING_VAR!r} dtype is {miss.dtype}, expected uint8/bool")

    # --- flash table
    present = [v for v in FLASH_VARS if v in ds]
    if present and len(present) != len(FLASH_VARS):
        problems.append(
            f"lightning point table incomplete: have {present}, need {list(FLASH_VARS)}"
        )
    elif present:
        if any(ds[v].dims != ("flash",) for v in FLASH_VARS):
            problems.append("lightning point variables must have dims ('flash',)")
        elif not np.issubdtype(ds["flash_time"].dtype, np.datetime64):
            problems.append("flash_time must be datetime64 (UTC)")

    # --- values (loads data; can be skipped for very large stores)
    if check_values and chans and not [p for p in problems if "channel coordinate" in p]:
        groups = [str(g) for g in ds["group"].values] if miss is not None else []
        for i, name in enumerate(chans):
            c = ch.get(name)
            v = x.isel(channel=i).values
            if miss is not None and c.group in groups:
                m = ds[MISSING_VAR].sel(group=c.group).values.astype(bool)
            else:
                m = np.zeros(v.shape, dtype=bool)
            unmasked_nan = int((~np.isfinite(v) & ~m).sum())
            if unmasked_nan:
                problems.append(
                    f"channel {name!r}: {unmasked_nan} non-finite values not flagged in 'missing' "
                    f"(missing data must be masked, not left as NaN)"
                )
            valid = v[np.isfinite(v) & ~m]
            if valid.size:
                lo, hi = float(valid.min()), float(valid.max())
                if lo < c.valid_min or hi > c.valid_max:
                    problems.append(
                        f"channel {name!r}: values [{lo:.3g}, {hi:.3g}] outside plausible range "
                        f"[{c.valid_min}, {c.valid_max}] {c.unit} (wrong units?)"
                    )
    return problems


def open_event_dataset(path: str | Path) -> xr.Dataset:
    return xr.open_zarr(str(path), consolidated=None)


def validate_event(path: str | Path, **kwargs) -> list[str]:
    """Validate an event Zarr store on disk. Returns a list of problems (empty = valid)."""
    path = Path(path)
    if not path.exists():
        return [f"{path} does not exist"]
    try:
        ds = open_event_dataset(path)
    except Exception as e:  # noqa: BLE001 - report any open failure as a problem
        return [f"cannot open {path} as Zarr: {type(e).__name__}: {e}"]
    problems = validate_dataset(ds, **kwargs)
    eid = ds.attrs.get("event_id")
    if eid is not None and path.suffix == ".zarr" and path.stem != str(eid):
        problems.append(f"file name {path.name!r} does not match event_id {eid!r}")
    return problems


def build_event_dataset(
    x: np.ndarray,
    channels: list[str],
    times,
    lat: np.ndarray,
    lon: np.ndarray,
    missing: dict[str, np.ndarray] | np.ndarray,
    event_id: str,
    grid_spacing_km: float = 2.0,
    flashes: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    attrs: dict | None = None,
) -> xr.Dataset:
    """Construct a contract-compliant Dataset from arrays (helper for producers and tests).

    ``missing`` is either {group: (T,H,W) bool} or an array (T, n_groups, H, W) ordered as GROUPS.
    ``flashes`` is (times datetime64, lat, lon).
    """
    if isinstance(missing, dict):
        missing = np.stack(
            [missing.get(g, np.zeros(x[:, 0].shape, bool)) for g in ch.GROUPS], axis=1
        )
    ds = xr.Dataset(
        {
            X_VAR: (X_DIMS, x.astype(np.float32)),
            MISSING_VAR: (MISSING_DIMS, missing.astype(np.uint8)),
        },
        coords={
            "time": pd.DatetimeIndex(times).tz_localize(None).values.astype("datetime64[ns]"),
            "channel": np.array(channels, dtype=str),
            "group": np.array(ch.GROUPS, dtype=str),
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
        },
        attrs={
            "event_id": event_id,
            "grid_spacing_km": float(grid_spacing_km),
            "contract_version": CONTRACT_VERSION,
            **(attrs or {}),
        },
    )
    if flashes is not None:
        ft, fla, flo = flashes
        ds["flash_time"] = ("flash", np.asarray(ft, dtype="datetime64[ns]"))
        ds["flash_lat"] = ("flash", np.asarray(fla, dtype=np.float64))
        ds["flash_lon"] = ("flash", np.asarray(flo, dtype=np.float64))
    ds[X_VAR].attrs["units"] = "see channels.py"
    return ds
