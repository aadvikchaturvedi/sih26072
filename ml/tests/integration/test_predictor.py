"""The backend-facing contract: Predictor.predict / predict_baseline."""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from nowcast_ml.inference import InputContractError, NowcastError, Predictor
from nowcast_ml.inference.schema import validate_forecast


@pytest.fixture(scope="module")
def predictor(artifact_root):
    return Predictor.load(artifact_root / "latest", device="cpu")


@pytest.fixture(scope="module")
def inputs(events_dir):
    p = sorted(events_dir.glob("*.zarr"))[0]
    return xr.open_zarr(p).isel(time=slice(0, 10)).load()


def test_predict_full(predictor, inputs):
    t0 = pd.Timestamp(inputs["time"].values[-1])
    fc = predictor.predict(inputs, t0.to_pydatetime())
    assert validate_forecast(fc) == []
    assert fc.attrs["mode"] == "full"
    assert fc["reflectivity"].shape == (12, 32, 32)
    assert (
        fc.attrs["model_name"] == "nowcast" and fc.attrs["model_version"] == predictor.info.version
    )
    assert fc.attrs["inference_ms"] > 0
    assert fc.attrs["t0"] == t0.strftime("%Y-%m-%dT%H:%M:%SZ")
    assert list(fc["lead"].values) == list(range(10, 121, 10))


def test_predict_satellite_only(predictor, inputs):
    ds = inputs.drop_sel(channel=["maxz", "cappi3km"])
    fc = predictor.predict(ds, ds["time"].values[-1])
    assert validate_forecast(fc) == []
    assert fc.attrs["mode"] == "satellite_only"
    assert {"maxz", "cappi3km"} <= set(fc.attrs["missing_channels"])
    ds2 = inputs.copy(deep=True)
    ds2["missing"].loc[{"group": "radar"}] = 1  # masked instead of dropped
    assert predictor.predict(ds2, ds2["time"].values[-1]).attrs["mode"] == "satellite_only"


def test_repeatable_and_stateless(predictor, inputs):
    t0 = inputs["time"].values[-1]
    a = predictor.predict(inputs, t0)
    predictor.predict(inputs.drop_sel(channel=["maxz", "cappi3km"]), t0)
    b = predictor.predict(inputs, t0)
    np.testing.assert_array_equal(a["reflectivity"].values, b["reflectivity"].values)
    np.testing.assert_array_equal(a["lightning_prob_60"].values, b["lightning_prob_60"].values)


def test_tz_aware_t0_and_earlier_t0(predictor, inputs):
    t = pd.Timestamp(inputs["time"].values[-2])
    aware = datetime(t.year, t.month, t.day, t.hour, t.minute, tzinfo=UTC)
    fc = predictor.predict(inputs, aware)
    assert fc.attrs["t0"] == t.strftime("%Y-%m-%dT%H:%M:%SZ")


def test_typed_errors(predictor, inputs):
    with pytest.raises(InputContractError, match="not one of the input times"):
        predictor.predict(inputs, "2001-01-01T00:00Z")
    with pytest.raises(InputContractError, match="need 7 frames"):
        predictor.predict(inputs, inputs["time"].values[3])
    bad = inputs.assign_coords(channel=["bogus"] + list(inputs["channel"].values[1:]))
    with pytest.raises(InputContractError) as e:
        predictor.predict(bad, inputs["time"].values[-1])
    assert any("bogus" in p for p in e.value.problems)
    with pytest.raises(InputContractError):
        predictor.predict("not a dataset", inputs["time"].values[-1])
    with pytest.raises(NowcastError, match="n_members"):
        predictor.predict(inputs, inputs["time"].values[-1], n_members=4)


@pytest.mark.parametrize("kind", ["persistence", "extrapolation", "steps"])
def test_predict_baseline_same_schema(predictor, inputs, kind):
    fc = predictor.predict_baseline(inputs, inputs["time"].values[-1], kind=kind, n_members=2)
    assert validate_forecast(fc) == []
    assert fc.attrs["model_name"] == f"baseline_{kind}"
    assert ("reflectivity_members" in fc) == (kind == "steps")


def test_info(predictor):
    info = predictor.info
    assert info.t_in == 7 and info.lead_minutes[-1] == 120 and len(info.channels) == 14
