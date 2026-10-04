# Thunderstorm and lightning nowcasting (SIH 2026, PS 26072)

| Directory | What it is |
|---|---|
| [`ml/`](ml/README.md) | `nowcast_ml`: data contract, model, training, calibration, evaluation, `Predictor` |
| [`backend/`](backend/README.md) | `nowcast_backend`: ingest, forecasts, storm cells, district and area warnings, CAP, HTTP API |
| [`web/`](web/README.md) | operator console (React): map, warnings, skill, status |

```
observation frames -> backend (rolling buffer -> nowcast_ml model -> cells, districts, warnings)
                          -> /api/v1/console -> web console
```

## Run the whole application

```bash
make install     # ml + backend into ml/.venv, npm install in web/
make demo        # backend on :8000 and console on http://localhost:5173
```

Real training data is not available yet, so `make demo` runs on **dummy data**: the backend
trains a small model on synthetic storms the first time (about 90 s), then replays a
synthetic afternoon over Odisha (19 forecast times, radar lost for the last hour to show the
satellite-only fallback). Storms, lightning, warnings and the skill scores in the console are
all generated and say nothing about real skill.

When real data arrives: produce input-contract stores (`ml/docs/data_contract.md`), train
with the stages in `ml/README.md`, point `NOWCAST_MODEL_PATH` at the artifact and push frames
to the backend (`backend/README.md`). The console needs no change.

`make test` runs the ML and backend test suites and type-checks the console.
