import numpy as np
import pandas as pd

from nowcast_ml.data.schema import validate_dataset, validate_event


def test_good_dataset_passes(synth_ds):
    assert validate_dataset(synth_ds) == []


def test_good_store_passes(events_dir):
    for p in sorted(events_dir.glob("*.zarr")):
        assert validate_event(p) == [], p


def _has(problems, text):
    assert any(text in p for p in problems), problems


def test_missing_variable(synth_ds):
    _has(validate_dataset(synth_ds.drop_vars("missing")), "'missing'")
    _has(validate_dataset(synth_ds.drop_vars("x")), "missing data variable 'x'")


def test_wrong_dims(synth_ds):
    bad = synth_ds.assign(x=synth_ds["x"].transpose("time", "y", "x", "channel"))
    _has(validate_dataset(bad), "dims are")


def test_unknown_channel(synth_ds):
    names = list(synth_ds["channel"].values)
    names[3] = "tir9_bt"
    _has(validate_dataset(synth_ds.assign_coords(channel=names)), "unknown channel 'tir9_bt'")


def test_irregular_time(synth_ds):
    t = pd.DatetimeIndex(synth_ds["time"].values)
    t = t.insert(len(t), t[-1] + pd.Timedelta(minutes=25))[1:]
    _has(validate_dataset(synth_ds.assign_coords(time=t.values)), "time spacing must be 10 min")


def test_wrong_grid_spacing(synth_ds):
    _has(validate_dataset(synth_ds, expected_spacing_km=4.0), "grid spacing along")


def test_unmasked_nan_is_flagged(synth_ds):
    x = synth_ds["x"].values.copy()
    x[0, 2, 0, 0] = np.nan  # tir1_bt, satellite group not masked there
    _has(validate_dataset(synth_ds.assign(x=(synth_ds["x"].dims, x))), "not flagged in 'missing'")


def test_wrong_units_flagged(synth_ds):
    x = synth_ds["x"].values.copy()
    x[:, 2] -= 273.15  # tir1_bt given in Celsius
    _has(validate_dataset(synth_ds.assign(x=(synth_ds["x"].dims, x))), "outside plausible range")


def test_missing_attrs(synth_ds):
    ds = synth_ds.copy()
    ds.attrs = {}
    _has(validate_dataset(ds), "event_id")


def test_south_up_rejected(synth_ds):
    ds = synth_ds.isel(y=slice(None, None, -1))
    _has(validate_dataset(ds), "northern edge")


def test_nonexistent_path(tmp_path):
    assert "does not exist" in validate_event(tmp_path / "nope.zarr")[0]
