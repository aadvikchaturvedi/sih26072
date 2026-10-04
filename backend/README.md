# nowcast_backend

Backend service for the thunderstorm and lightning nowcasting system (SIH 2026, PS 26072).
It takes observation frames in the [input contract](../ml/docs/data_contract.md), runs the
`nowcast_ml` model, and serves forecasts, storm cells and warnings over HTTP.

```
frames (HTTP push / watched Zarr dir / replay)
   -> rolling observation buffer -> model forecast -> storm cells -> warnings -> CAP
                                         |                |             |
                                   PNG layers,      GeoJSON,      GeoJSON, CAP 1.2 XML,
                                   point series     tracks        Atom feed, webhook, SSE
```

Not included: parsers for raw radar / INSAT / lightning-network files. Those belong to the
data team, who write contract-format stores; `adapters/sources.py` is where a new source plugs in.

## Run

The backend shares the ML package's virtualenv.

```bash
cd backend
make install          # after `make install` in ../ml
make test             # 29 tests, ~4 s, no model needed
make test-all         # also trains a tiny model and runs the real Predictor

make demo             # everything on dummy data (see below)
NOWCAST_MODEL_PATH=<artifact>/latest nowcast-backend serve    # with a real model
```

Interactive API docs: <http://127.0.0.1:8000/api/v1/docs>.

Configuration is by `NOWCAST_*` environment variables or a `.env` file; see
[.env.example](.env.example) and `settings.py`. The server stops at startup with a clear
message if the model artifact is missing or inconsistent.

### Dummy data (`make demo`)

Real training data is not available yet, so `nowcast-backend demo` (`demo.py`) builds a
complete stand-in under `var/demo/` and serves it:

1. 24 synthetic storm events from `nowcast_ml.data.synthetic`,
2. a small model trained, calibrated and evaluated on them (about 90 s on a laptop; kept
   for later runs, `--retrain` redoes it),
3. one synthetic afternoon (09:00-12:30 UTC, radar lost from 11:30) on a 2 km grid over
   Odisha, replayed through the normal ingest path.

`--live-from N` preloads only the first N frames and delivers the rest one per poll, so
forecasts arrive while the console is open. Everything shown, including the skill page,
comes from generated storms and says nothing about real skill.

## API (`/api/v1`)

| Method and path | What it returns |
|---|---|
| `GET /health`, `GET /model`, `GET /styles` | status, model description, colour scales for legends |
| `GET /events` | server-sent events: `observation.ingested`, `forecast.ready`, `warnings.updated` |
| `GET /domains`, `GET /domains/{d}` | grid, bounds, channels and observed times per domain |
| `POST /domains/{d}/observations` 🔑 | ingest frames (zipped Zarr body); runs the forecast when time advances |
| `GET /domains/{d}/observations/{time}/{channel}.png` | an observed channel as a map overlay |
| `GET /domains/{d}/forecasts` | stored forecasts, newest first |
| `POST /domains/{d}/forecasts` 🔑 | run a forecast: `{"t0", "source", "n_members", "force"}` |
| `GET /domains/{d}/forecasts/{t0}` | forecast summary and available layers |
| `GET /domains/{d}/forecasts/{t0}/layers/{layer}.png?lead=30&scale=2` | transparent PNG to stretch over `bounds` |
| `GET /domains/{d}/forecasts/{t0}/layers/{layer}?lead=30&stride=2` | the same field as numbers |
| `GET /domains/{d}/forecasts/{t0}/point?lat=..&lon=..` | time series at a location |
| `GET /domains/{d}/forecasts/{t0}/cells` | storm cells (GeoJSON): motion, trend, extrapolated track |
| `GET /warnings?domain=..&status=active` | warnings (GeoJSON, or `format=json`) |
| `GET /warnings/{id}`, `GET /warnings/{id}/cap` | one warning; its CAP 1.2 alert |
| `GET /cap/feed.atom` | Atom feed of active warnings for CAP aggregators |

### Console API (`/api/v1/console`)

The web console (`../web`) defines the responses it needs in `web/src/data/types.ts`. These
endpoints return exactly those shapes for one domain (`NOWCAST_CONSOLE_DOMAIN`); the mapping
from the backend's own models is in `api/console/`.

| Path | What it returns |
|---|---|
| `GET /timeline` | analysis times with a forecast, step, leads, image extent |
| `GET /forecast/{t0}` | image URLs per lead (reflectivity, lightning probability, first flash) and model attrs |
| `GET /radar/{time}` | observed MAX-Z as PNG (transparent when there is no radar) |
| `GET /cells/{t0}` | storm cells: track, forecast path with risk, flash rate, lightning jump, districts on the path |
| `GET /lightning/{t0}` | flash points of the last 30 min, tagged with the nearest cell |
| `GET /districts/{t0}` | per-district probability, MAX-Z and level |
| `GET /warnings/{t0}` | district level changes up to `t0`, newest first |
| `GET /cap/{warning_id}` | `{"xml", "json"}` CAP 1.2 for a district warning |
| `GET /skill` | the `nowcast-eval` report (`NOWCAST_SKILL_REPORT_PATH`) |
| `GET /health`, `GET /health/{t0}` | model mode and per-input status at that run |
| `WS /live` | a tick (new warnings, cells, health) for every new forecast |

Districts come from a GeoJSON file (`NOWCAST_REGIONS_PATH`, features with `id` and `name`).
District levels follow the console's draft rules (`DistrictSettings`): yellow at probability
≥ 30 % or MAX-Z ≥ 40 dBZ, orange above 60 % or with a lightning jump in an approaching cell,
red when orange coincides with a ≥ 50 dBZ core or a first-flash flag. District warnings are
derived from the stored forecasts on request, so the same forecasts always give the same
warnings.

- `{t0}` and `{time}` are `latest` or a UTC time (`2026-05-12T10:50Z` or `20260512T1050Z`).
- `source` (query or body) is `model` (default), `persistence`, `extrapolation` or `steps`.
- Layers: `reflectivity` (needs `lead`), `lightning_prob_30`, `lightning_prob_60`, `first_flash`,
  and `exceedance` (share of ensemble members ≥ `threshold` dBZ, when members exist).
- 🔑 needs the `X-API-Key` header when `NOWCAST_API_KEY` is set. Without a key configured, write
  endpoints are open, which is why the default bind address is `127.0.0.1`.
- Errors are `{"error": {"code", "message", "problems": [...]}}`. Contract violations in pushed
  frames come back as 422 with every problem listed.

### Pushing frames

```python
from nowcast_backend.client import push_frames

result = push_frames("http://127.0.0.1:8000", "kolkata", frames, api_key="...")  # xr.Dataset
print(result["forecast"]["t0"], result["forecast"]["mode"])
```

or `nowcast-backend push event.zarr --domain kolkata --last 7`. A domain's grid and channel list
are fixed by its first upload. Frames may arrive late or out of order; missing 10-minute steps
are filled with fully-masked frames, and a forecast runs once at least
`NOWCAST_MIN_OBSERVED_FRAMES` (4) of the model's 7 input frames are real.

## What the backend adds to the model output

- **Storm cells** (`services/cells.py`): connected areas ≥ 35 dBZ in the radar frame at t0, tracked
  back through the input frames for speed and direction, extrapolated 120 min ahead with the
  forecast intensity and lightning probability along the track. Flash rates come from the
  flash points near the cell; a lightning jump is a rise above 2 sigma of its recent changes. Ids carry over between runs. Without radar
  (`satellite_only`), cells come from the +10 min forecast and are marked `basis: "forecast"`.
- **Warnings** (`services/warnings.py`): one per connected area above the yellow threshold, with
  the level set by the peak inside it. Lightning uses P(flash in 30 min) ≥ 0.3 / 0.5 / 0.7;
  thunderstorm uses forecast MAX-Z within 60 min ≥ 40 / 50 / 60 dBZ. Each run replaces the
  domain's active warnings: overlapping ones are superseded (CAP `Update` with `references`),
  the rest are cancelled or expire.
- **CAP 1.2** (`services/cap.py`): `status` is `Test` by default. The thresholds above are
  starting values, not verified ones; switch to `Actual` only after checking them on real events.

## Code layout

```
src/nowcast_backend/
  domain/      models (also the JSON schema), grid, typed errors, time helpers
  ports.py     interfaces: ForecastEngine, ObservationStore, ForecastStore,
               WarningRepository, Notifier, FrameSource
  services/    nowcast.py (use cases), cells.py, warnings.py, cap.py, render.py, geo.py
  adapters/    engine_ml.py (nowcast_ml Predictor), observations.py (rolling buffer + Zarr
               snapshot), forecast_store.py (Zarr), warning_repo.py (SQLite),
               notifiers.py (SSE bus, webhook), sources.py (watch / replay), zarr_zip.py
  api/         FastAPI app and routes; api/console/ maps domain models to the web console's shapes
  demo.py      dummy data and model for running without real data
  container.py the one place that picks adapters for the ports
  scheduler.py polls a FrameSource
```

Services import only `domain` and `ports`. To swap storage, the model runtime or the notification
channel, write another adapter and change `container.py`; the tests do exactly that with a fake
engine. State lives under `NOWCAST_DATA_DIR` and survives a restart.

## Limits

- One process: the observation buffer is in memory (snapshotted to disk) and inference is
  serialized, so run a single worker.
- PNG layers are in grid space. Stretching them over `bounds` is exact for regular lat/lon
  grids and approximate for projected ones (e.g. SEVIR's LAEA).
- Area warning polygons ignore holes. District boundaries are the console's community GeoJSON,
  not official Survey of India / IMD shapefiles.
- A lightning stroke carries time and position only; polarity and peak current are not in the
  input contract. Cell `confidence` is not supplied (no verified definition yet).
