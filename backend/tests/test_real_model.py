"""The real chain: a tiny trained model behind ``PredictorEngine`` (slow: trains ~1 epoch)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nowcast_backend.adapters.zarr_zip import encode
from nowcast_backend.api.app import create_app

pytestmark = pytest.mark.slow

ML_CONFIGS = Path(__file__).resolve().parents[2] / "ml" / "configs"


@pytest.fixture(scope="module")
def artifact(tmp_path_factory) -> Path:
    from nowcast_ml.config import load_config
    from nowcast_ml.data.synthetic import write_events
    from nowcast_ml.training.train import run

    root = tmp_path_factory.mktemp("model")
    write_events(root / "events", 6, seed=1, size=32, n_frames=24, radar_missing_prob=0.0)
    cfg = load_config(
        ML_CONFIGS / "train" / "smoke.yaml",
        [
            f"data.events_dir={root / 'events'}",
            "data.sample_stride=3",
            "data.batch_size=4",
            "model.hid_s=8",
            "model.hid_t=16",
            "model.lightning_head.hidden=8",
            "train.max_epochs=1",
            f"train.output_dir={root / 'run'}",
            f"registry.root={root / 'models'}",
        ],
    )
    run(cfg)
    return root


def test_real_predictor_end_to_end(artifact, settings):
    import xarray as xr

    settings = settings.model_copy(
        update={"model_path": artifact / "models" / "nowcast" / "latest", "device": "cpu"}
    )
    event = xr.open_zarr(sorted((artifact / "events").glob("*.zarr"))[0]).load()
    with TestClient(create_app(settings)) as client:
        api = "/api/v1/domains/test"
        r = client.post(f"{api}/observations", content=encode(event.isel(time=slice(0, 10))))
        assert r.status_code == 200, r.text
        forecast = r.json()["forecast"]
        assert forecast["model_name"] == "nowcast" and forecast["mode"] == "full"
        assert (
            client.get(f"{api}/forecasts/latest/layers/reflectivity.png?lead=60").status_code == 200
        )

        blind = event.isel(time=slice(10, 11)).copy(deep=True)
        blind["missing"].loc[{"group": "radar"}] = 1
        r = client.post(f"{api}/observations", content=encode(blind))
        assert r.json()["forecast"]["mode"] == "satellite_only"  # no radar in the newest frame
        cells = client.get(f"{api}/forecasts/latest/cells", params={"format": "json"}).json()
        assert all(c["basis"] == "forecast" for c in cells)
        baseline = client.post(f"{api}/forecasts", json={"source": "persistence"})
        assert baseline.status_code == 200 and baseline.json()["model_version"] == "baseline"
        assert (
            client.post(f"{api}/forecasts", json={"n_members": 2, "force": True}).status_code == 400
        )
