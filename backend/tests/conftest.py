"""Fixtures. The API and services are tested against a fake engine (no torch, no
trained model); ``test_real_model.py`` covers the real ``Predictor`` adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastapi.testclient import TestClient
from nowcast_ml.data.schema import validate_dataset
from nowcast_ml.data.synthetic import generate_event, to_dataset
from nowcast_ml.inference.schema import ForecastArrays, build_forecast_dataset
from scipy.ndimage import maximum_filter

from nowcast_backend.api.app import create_app
from nowcast_backend.domain.errors import InvalidRequest
from nowcast_backend.domain.models import ModelDescription
from nowcast_backend.settings import Settings

LEADS = list(range(10, 121, 10))


class FakeEngine:
    """Persistence forecast with the real output schema: reflectivity = last radar frame,
    lightning probability = spread-out recent flash density."""

    def __init__(self, has_ensemble: bool = False):
        self.has_ensemble = has_ensemble
        self.calls = 0

    def describe(self) -> ModelDescription:
        return ModelDescription(
            name="fake",
            version="v0",
            channels=["maxz", "tir1_bt", "flash_density"],
            grid_spacing_km=2.0,
            t_in=7,
            step_minutes=10,
            lead_minutes=LEADS,
            lightning_leads_min=[30, 60],
            calibrated=False,
            has_ensemble=self.has_ensemble,
            backend="fake",
            device="cpu",
        )

    def validate(self, inputs: xr.Dataset) -> list[str]:
        return validate_dataset(inputs, expected_spacing_km=2.0)

    def forecast(self, inputs, t0, *, source="model", n_members=0) -> xr.Dataset:
        self.calls += 1
        if n_members and not self.has_ensemble:
            raise InvalidRequest("this model has no ensemble refiner")
        channels = [str(c) for c in inputs["channel"].values]
        groups = [str(g) for g in inputs["group"].values]
        last = inputs.isel(time=-1)
        x, missing = last["x"].values, last["missing"].values.astype(bool)
        radar_ok = "maxz" in channels and not missing[groups.index("radar")].all()
        maxz = (
            np.nan_to_num(x[channels.index("maxz")], nan=0.0) if radar_ok else np.zeros(x.shape[1:])
        )
        flashes = np.nan_to_num(inputs["x"].values[-3:, channels.index("flash_density")], nan=0.0)
        prob = np.clip(maximum_filter((flashes > 0).any(axis=0).astype(np.float32), size=9), 0, 1)
        refl = np.repeat(maxz[None], len(LEADS), axis=0).astype(np.float32)
        members = None
        if n_members:
            members = np.stack([refl + i for i in range(n_members)]).astype(np.float32)
        return build_forecast_dataset(
            ForecastArrays(refl, np.stack([prob * 0.9, prob]), members),
            lat=inputs["lat"].values,
            lon=inputs["lon"].values,
            t0=t0,
            lead_minutes=LEADS,
            ltg_leads_min=[30, 60],
            past_ltg=np.zeros(maxz.shape, dtype=bool),
            model_name="fake" if source == "model" else f"baseline_{source}",
            model_version="v0",
            mode="full" if radar_ok else "satellite_only",
            missing_channels=[],
            inference_ms=1.0,
        )


@pytest.fixture(scope="session")
def event() -> xr.Dataset:
    """A 4-hour synthetic storm event (24 frames, 48 x 48 px at 2 km) in the input contract."""
    return to_dataset(generate_event(seed=3, size=48, n_frames=24, n_storms=(3, 3))).load()


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        data_dir=tmp_path / "var",
        clock="data",
        cells={"threshold_dbz": 30.0, "min_area_km2": 8.0},
        warnings={"min_area_km2": 8.0},
    )


@pytest.fixture()
def engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture()
def client(settings, engine):
    with TestClient(create_app(settings, engine)) as c:
        yield c


@pytest.fixture()
def service(client):
    return client.app.state.container.service


def frames(event: xr.Dataset, start: int, stop: int) -> xr.Dataset:
    return event.isel(time=slice(start, stop))


def t(event: xr.Dataset, i: int) -> pd.Timestamp:
    return pd.Timestamp(event["time"].values[i])
