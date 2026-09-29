# Input data contract (v1.0) for the data team

The nowcasting model reads **one Zarr store per event**. This document is the
complete specification. The code that enforces it is
`src/nowcast_ml/data/schema.py`, and the channel registry is
`src/nowcast_ml/data/channels.py`. If this document and the code disagree, the code
is right, so please open an issue.

Check your stores with:

```bash
nowcast-validate data/events/              # every *.zarr in the directory
nowcast-validate data/events/E123.zarr     # one store; exit code 1 if anything is wrong
```

The same checks run inside `Predictor.predict`. An invalid input raises
`InputContractError` with the list of problems.

## 1. Files

```
data/events/<event_id>.zarr      one store per event (training / evaluation)
```

- `<event_id>` must equal the store's `event_id` attribute, e.g. `20260512_kolkata_01`.
- An "event" is a contiguous period around convection (typically 3–12 h). Train/val/test
  splits are made **by event**, so do not cut one storm day into many tiny events.
- For **real-time inference**, the backend passes the same structure as an in-memory
  `xarray.Dataset`. It needs at least the last 7 frames up to `t0`, and no file is needed.

## 2. Structure

| Name | Kind | Dims | Dtype | Meaning |
|---|---|---|---|---|
| `x` | data var | `(time, channel, y, x)` | float32 | all gridded fields, physical units (table in §3) |
| `missing` | data var | `(time, group, y, x)` | uint8 | **1 = missing**, 0 = valid, per channel group |
| `time` | coord | `(time,)` | datetime64[ns] | **UTC**, strictly every **10 min** |
| `channel` | coord | `(channel,)` | str | channel names from §3, any subset, any order |
| `group` | coord | `(group,)` | str | `radar`, `satellite`, `lightning`, `nwp` |
| `lat`, `lon` | coord | `(y, x)` | float64 | pixel-centre degrees; **row 0 is the northern edge** |
| `flash_time` | data var (optional) | `(flash,)` | datetime64[ns] | lightning flash times, UTC |
| `flash_lat`, `flash_lon` | data var (optional) | `(flash,)` | float64 | flash locations |

Required global attributes:

| Attribute | Example | Notes |
|---|---|---|
| `event_id` | `"20260512_kolkata_01"` | equals the file name stem |
| `grid_spacing_km` | `2.0` | the model is trained for 2 km |
| `contract_version` | `"1.0"` | major version must match |

Recommended chunking: `x` chunks of `(1, n_channel, y, x)` (one frame per chunk), with
`consolidated=True` when writing.

## 3. Channels

| Name | Group | Unit | Plausible range | Description | In SEVIR |
|---|---|---|---|---|---|
| `maxz` | radar | dBZ | -35 – 85 | Column-maximum reflectivity (MAX-Z) | yes |
| `cappi3km` | radar | dBZ | -35 – 85 | Reflectivity CAPPI at 3 km |  |
| `tir1_bt` | satellite | K | 150 – 340 | INSAT TIR1 (10.8 µm) brightness temperature | yes |
| `tir2_bt` | satellite | K | 150 – 340 | INSAT TIR2 (12.0 µm) brightness temperature |  |
| `wv_bt` | satellite | K | 150 – 300 | INSAT WV (6.8 µm) brightness temperature | yes |
| `mir_bt` | satellite | K | 150 – 360 | INSAT MIR (3.9 µm) brightness temperature |  |
| `tir1_cooling` | satellite | K/10min | -60 – 60 | TIR1 BT change over the previous 10 min | yes |
| `tir1_minus_wv` | satellite | K | -40 – 100 | TIR1 minus WV BT difference | yes |
| `flash_density` | lightning | flashes/km2/10min | 0 – 100 | Lightning flash density | yes |
| `cape` | nwp | J/kg | 0 – 10000 | Convective available potential energy |  |
| `cin` | nwp | J/kg | -1500 – 0 | Convective inhibition (≤ 0) |  |
| `shear_0_6km` | nwp | m/s | 0 – 80 | 0–6 km bulk wind shear |  |
| `pwat` | nwp | mm | 0 – 100 | Precipitable water |  |
| `freezing_level` | nwp | m | 0 – 7000 | Freezing level height AGL |  |

Rules:

- **Units are physical, not normalized.** The model normalizes internally. BT is in
  Kelvin, not °C. If values fall outside the plausible range, the validator assumes the
  units are wrong.
- **No echo means 0 dBZ (or the measured low value), not missing.** Use `missing` only
  where there is no measurement (outside radar range, beam blockage, outages).
- **Any subset of channels is allowed.** An absent channel is treated as missing. A
  store with no radar channels, or radar entirely masked, runs in `satellite_only` mode.
- `tir1_cooling(t) = tir1_bt(t) − tir1_bt(t − 10 min)`, computed after regridding.
- `flash_density(t)` counts flashes in `(t − 10 min, t]` per km².
- NWP fields: use the latest available run, interpolated to each 10-min frame.
  Hourly fields may be held constant within the hour.

## 4. Time

- Timestamps are **UTC** (not IST), datetime64 without a timezone, e.g. `2026-05-12T10:30:00`.
- A frame's time is the **end** of its 10-min accumulation window (for lightning), or
  the nominal scan time (for radar/satellite), rounded to the 10-min grid.
- Spacing is exactly 10 min. **Include missing frames** with `missing = 1` instead of
  dropping them. The validator rejects gaps.

## 5. Missing data

- `missing[t, g, y, x] = 1` means every channel of group `g` is unobserved at that pixel/time.
- Where `missing = 1`, values in `x` are ignored. NaN is recommended there, but not required.
- Where `missing = 0`, values in `x` **must be finite**. A NaN not covered by `missing`
  is a contract violation. Never zero-fill missing data: to the model, 0 dBZ means
  "no rain", not "no radar".

## 6. Lightning labels

Training labels are *at least one flash within 10 km in (t0, t0 + L]*, for L = 30 and 60 min.
If you provide the point table (`flash_time`, `flash_lat`, `flash_lon`), labels are
built from the points; otherwise they come from `flash_density > 0`. Provide the point
table whenever possible, since it gives sharper labels. Flashes outside the grid are
ignored. Mark `missing[:, "lightning"] = 1` during lightning-network outages so those
frames produce no false "no lightning" labels.

## 7. Minimal producer example

<!-- example:producer -->
```python
import numpy as np
import pandas as pd

from nowcast_ml.data.schema import build_event_dataset, validate_event

T, H, W = 12, 64, 64
times = pd.date_range("2026-05-12T09:00", periods=T, freq="10min")  # UTC
lat0, lon0, d = 22.5, 88.3, 2.0 / 111.0                              # ~2 km in degrees
lat = (lat0 - np.arange(H) * d)[:, None].repeat(W, axis=1)            # row 0 = north
lon = (lon0 + np.arange(W) * d / np.cos(np.deg2rad(lat0)))[None, :].repeat(H, axis=0)

channels = ["maxz", "tir1_bt", "flash_density"]
x = np.zeros((T, len(channels), H, W), np.float32)
x[:, 1] = 290.0                                   # clear-sky TIR1 in K
missing = {"radar": np.zeros((T, H, W), bool),
           "satellite": np.zeros((T, H, W), bool),
           "lightning": np.zeros((T, H, W), bool)}
missing["radar"][5] = True                        # radar outage at 09:50
x[5, 0] = np.nan

ds = build_event_dataset(x, channels, times, lat, lon, missing,
                         event_id="20260512_example", grid_spacing_km=2.0)
ds.to_zarr("20260512_example.zarr", mode="w", consolidated=True)
assert validate_event("20260512_example.zarr") == []
```

## 8. Versioning

Additive changes (a new optional channel) bump the minor version. Anything that changes
the meaning of existing fields bumps the major version, and models refuse to load
inputs with a different major version.
