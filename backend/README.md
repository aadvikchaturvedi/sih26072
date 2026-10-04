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
make test             # 19 tests, ~2 s, no model needed
make test-all         # also trains a tiny model and runs the real Predictor (~5 s)

# needs a model artifact; `make e2e` in ../ml builds a synthetic one
NOWCAST_MODEL_PATH=../ml/runs/e2e/artifacts/nowcast/latest nowcast-backend serve
make demo             # replays one synthetic event, a frame every 5 s
```

Interactive API docs: <http://127.0.0.1:8000/api/v1/docs>.

Configuration is by `NOWCAST_*` environment variables or a `.env` file; see
[.env.example](.env.example) and `settings.py`. The server stops at startup with a clear
message if the model artifact is missing or inconsistent.

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
  back through the input frames for speed and direction, extrapolated 60 min ahead with the
  forecast intensity along the track. Ids carry over between runs. Without radar
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
  api/         FastAPI app and routes
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
- Warning polygons ignore holes, and areas are not yet mapped to district names.
