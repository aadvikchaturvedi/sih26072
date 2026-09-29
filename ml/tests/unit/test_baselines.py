import numpy as np
import pytest

from nowcast_ml.baselines import ForecastInput, make_baseline
from nowcast_ml.data import channels as ch
from nowcast_ml.data.event import Event
from nowcast_ml.data.synthetic import generate_event, to_dataset
from nowcast_ml.data.zarr_events import WindowSpec, make_sample
from nowcast_ml.inference.schema import build_forecast_dataset, validate_forecast

CH = list(ch.ALL_CHANNELS)


def _blob_input(motion=(0.0, 2.0), size=64, drop_radar=False):
    """Single non-growing blob moving east at 2 px / 10 min."""
    ev = generate_event(21, size=size, n_frames=24, n_storms=(1, 1), motion=motion)
    ds = to_dataset(ev)
    if drop_radar:
        ds = ds.copy(deep=True)
        ds["missing"].loc[{"group": "radar"}] = 1
    event = Event(ds, CH)
    s = make_sample(event, 9, WindowSpec())
    return ForecastInput.from_sample(s, CH, 2.0), s, event


def _centroid_x(field):
    f = np.clip(np.nan_to_num(field), 0, None)
    return (f.sum(0) * np.arange(f.shape[1])).sum() / max(f.sum(), 1e-9)


def test_persistence():
    inp, s, _ = _blob_input()
    fc = make_baseline("persistence").forecast(inp)
    assert fc.reflectivity.shape == (12, 64, 64)
    np.testing.assert_array_equal(fc.reflectivity[5], s["x"][-1, 0])
    assert fc.lightning.shape == (2, 64, 64)


@pytest.mark.parametrize("kind", ["extrapolation", "steps"])
def test_blob_advected_east(kind):
    inp, s, _ = _blob_input()
    f = make_baseline(kind, n_members=3).forecast(inp)
    x_now = _centroid_x(s["x"][-1, 0])
    x_6 = _centroid_x(f.reflectivity[5])  # +60 min: expect ~ +12 px
    assert x_6 - x_now > 6, (x_now, x_6)
    if kind == "steps":
        assert f.members.shape == (3, 12, 64, 64)


def test_extrapolation_blob_advected_south():
    inp, s, _ = _blob_input(motion=(2.0, 0.0))
    f = make_baseline("extrapolation").forecast(inp)
    cy = lambda a: _centroid_x(np.nan_to_num(a).T)  # noqa: E731
    assert cy(f.reflectivity[5]) - cy(s["x"][-1, 0]) > 6


def test_satellite_only_extrapolation_uses_tir_motion():
    inp, _, _ = _blob_input(drop_radar=True)
    f = make_baseline("extrapolation").forecast(inp)
    assert np.isnan(f.reflectivity).all()  # no radar -> no reflectivity baseline
    assert f.lightning.shape == (2, 64, 64)


@pytest.mark.parametrize("kind", ["persistence", "extrapolation", "steps"])
def test_output_schema(kind):
    inp, s, ev = _blob_input(size=32)
    fc = make_baseline(kind, n_members=2).forecast(inp)
    ds = build_forecast_dataset(
        fc,
        lat=ev.lat,
        lon=ev.lon,
        t0=ev.times[9],
        lead_minutes=list(range(10, 121, 10)),
        ltg_leads_min=[30, 60],
        past_ltg=s["past_ltg"],
        model_name=kind,
        model_version="baseline",
        mode="full",
        missing_channels=[],
        inference_ms=1.0,
    )
    assert validate_forecast(ds) == []
    assert ds.attrs["t0"].endswith("Z")
    assert ("reflectivity_members" in ds) == (kind == "steps")
