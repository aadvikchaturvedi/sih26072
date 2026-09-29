"""OUTPUT CONTRACT for forecasts returned by ``Predictor.predict`` / ``predict_baseline``.

=============================  =========================  ==========================================
variable                       dims                       meaning
=============================  =========================  ==========================================
``reflectivity``               (lead, y, x) float32       forecast MAX-Z in dBZ (NaN = no forecast)
``lightning_prob_30``          (y, x) float32             P(>=1 flash within 10 km in (t0, t0+30])
``lightning_prob_60``          (y, x) float32             same for (t0, t0+60]; calibrated, in [0, 1]
``first_flash``                (y, x) bool                prob_30 >= threshold AND no flash within
                                                          10 km in (t0-30, t0]
``reflectivity_members``       (member, lead, y, x)       optional ensemble members (dBZ)
=============================  =========================  ==========================================

Coordinates: ``lead`` (int minutes, 10..120), ``valid_time`` (lead,), ``lat``/``lon`` (y, x).

Attributes: ``model_name``, ``model_version``, ``t0`` (ISO-8601 UTC, ``Z`` suffix),
``mode`` (``full`` | ``satellite_only``), ``missing_channels`` (list of str),
``inference_ms`` (float), ``output_contract_version``, ``first_flash_threshold``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr

OUTPUT_CONTRACT_VERSION = "1.0"
MODES = ("full", "satellite_only")
REQUIRED_ATTRS = (
    "model_name",
    "model_version",
    "t0",
    "mode",
    "missing_channels",
    "inference_ms",
    "output_contract_version",
)


def ltg_var(lead_min: int) -> str:
    return f"lightning_prob_{lead_min}"


@dataclass
class ForecastArrays:
    """Raw forecast arrays before they are wrapped into the output Dataset."""

    reflectivity: np.ndarray  # (n_leads, H, W) dBZ
    lightning: np.ndarray  # (n_ltg_leads, H, W) probabilities
    members: np.ndarray | None = None  # (M, n_leads, H, W) dBZ


def to_utc_timestamp(t0) -> pd.Timestamp:
    """Normalize a datetime / string to a tz-naive UTC ``pd.Timestamp``."""
    ts = pd.Timestamp(t0)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def iso_utc(t0) -> str:
    return to_utc_timestamp(t0).strftime("%Y-%m-%dT%H:%M:%SZ")


def first_flash(prob_30: np.ndarray, past_ltg: np.ndarray, threshold: float) -> np.ndarray:
    return (np.nan_to_num(prob_30, nan=0.0) >= threshold) & ~np.asarray(past_ltg, dtype=bool)


def build_forecast_dataset(
    fc: ForecastArrays,
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    t0: datetime | str | pd.Timestamp,
    lead_minutes: list[int],
    ltg_leads_min: list[int],
    past_ltg: np.ndarray,
    model_name: str,
    model_version: str,
    mode: str,
    missing_channels: list[str],
    inference_ms: float,
    first_flash_threshold: float = 0.5,
    extra_attrs: dict | None = None,
) -> xr.Dataset:
    t0_ts = to_utc_timestamp(t0)
    lead = np.asarray(lead_minutes, dtype=np.int32)
    valid = (t0_ts + pd.to_timedelta(lead, unit="min")).values
    ltg = np.clip(np.nan_to_num(fc.lightning.astype(np.float32), nan=0.0), 0.0, 1.0)
    data = {
        "reflectivity": (("lead", "y", "x"), fc.reflectivity.astype(np.float32)),
    }
    for i, L in enumerate(ltg_leads_min):
        data[ltg_var(L)] = (("y", "x"), ltg[i])
    p30 = ltg[ltg_leads_min.index(30)] if 30 in ltg_leads_min else ltg[0]
    data["first_flash"] = (("y", "x"), first_flash(p30, past_ltg, first_flash_threshold))
    if fc.members is not None:
        data["reflectivity_members"] = (("member", "lead", "y", "x"), fc.members.astype(np.float32))
    ds = xr.Dataset(
        data,
        coords={
            "lead": ("lead", lead, {"units": "minutes"}),
            "valid_time": ("lead", valid),
            "lat": (("y", "x"), np.asarray(lat)),
            "lon": (("y", "x"), np.asarray(lon)),
        },
        attrs={
            "model_name": model_name,
            "model_version": model_version,
            "t0": iso_utc(t0_ts),
            "mode": mode,
            "missing_channels": list(missing_channels),
            "inference_ms": float(inference_ms),
            "output_contract_version": OUTPUT_CONTRACT_VERSION,
            "first_flash_threshold": float(first_flash_threshold),
            **(extra_attrs or {}),
        },
    )
    ds["reflectivity"].attrs["units"] = "dBZ"
    for L in ltg_leads_min:
        ds[ltg_var(L)].attrs.update(units="1", long_name=f"P(flash within 10 km in next {L} min)")
    if "member" in ds.dims:
        ds = ds.assign_coords(member=np.arange(ds.sizes["member"]))
    return ds


def validate_forecast(
    ds: xr.Dataset, lead_minutes: list[int] | None = None, ltg_leads_min: list[int] = (30, 60)
) -> list[str]:
    """Return output-contract violations (empty list = valid)."""
    p: list[str] = []
    lead_minutes = list(lead_minutes or range(10, 121, 10))
    if "reflectivity" not in ds:
        p.append("missing 'reflectivity'")
    else:
        r = ds["reflectivity"]
        if r.dims != ("lead", "y", "x"):
            p.append(f"reflectivity dims {r.dims} != ('lead', 'y', 'x')")
        if r.dtype != np.float32:
            p.append(f"reflectivity dtype {r.dtype} != float32")
        vals = r.values[np.isfinite(r.values)]
        if vals.size and (vals.min() < -35 or vals.max() > 90):
            p.append(f"reflectivity outside [-35, 90] dBZ: [{vals.min():.1f}, {vals.max():.1f}]")
    if "lead" not in ds.coords or list(ds["lead"].values) != lead_minutes:
        p.append(f"lead coordinate must be {lead_minutes}")
    for L in ltg_leads_min:
        v = ltg_var(L)
        if v not in ds:
            p.append(f"missing {v!r}")
            continue
        a = ds[v]
        if a.dims != ("y", "x"):
            p.append(f"{v} dims {a.dims} != ('y', 'x')")
        if not np.isfinite(a.values).all() or a.values.min() < 0 or a.values.max() > 1:
            p.append(f"{v} must be finite and within [0, 1]")
    if "first_flash" not in ds:
        p.append("missing 'first_flash'")
    elif ds["first_flash"].dtype != bool:
        p.append("first_flash must be bool")
    if "reflectivity_members" in ds and ds["reflectivity_members"].dims != (
        "member",
        "lead",
        "y",
        "x",
    ):
        p.append("reflectivity_members dims must be ('member', 'lead', 'y', 'x')")
    for c in ("lat", "lon"):
        if c not in ds.coords or ds[c].dims != ("y", "x"):
            p.append(f"coordinate {c!r} (y, x) missing")
    for a in REQUIRED_ATTRS:
        if a not in ds.attrs:
            p.append(f"missing attribute {a!r}")
    if ds.attrs.get("mode") not in MODES:
        p.append(f"mode must be one of {MODES}")
    t0 = ds.attrs.get("t0")
    if isinstance(t0, str) and not t0.endswith("Z"):
        p.append("t0 attribute must be ISO-8601 UTC ending in 'Z'")
    return p
