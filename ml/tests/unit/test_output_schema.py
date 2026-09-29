import numpy as np

from nowcast_ml.inference.schema import (
    ForecastArrays,
    build_forecast_dataset,
    first_flash,
    validate_forecast,
)


def _ds(**kw):
    H = W = 8
    fc = ForecastArrays(np.zeros((12, H, W)), np.full((2, H, W), 0.2))
    args = dict(
        lat=np.zeros((H, W)),
        lon=np.zeros((H, W)),
        t0="2026-05-12T10:30Z",
        lead_minutes=list(range(10, 121, 10)),
        ltg_leads_min=[30, 60],
        past_ltg=np.zeros((H, W), bool),
        model_name="m",
        model_version="v",
        mode="full",
        missing_channels=[],
        inference_ms=1.0,
    )
    args.update(kw)
    return build_forecast_dataset(fc, **args)


def test_valid():
    ds = _ds()
    assert validate_forecast(ds) == []
    assert ds.attrs["t0"] == "2026-05-12T10:30:00Z"
    assert str(ds["valid_time"].values[0]).startswith("2026-05-12T10:40")


def test_ist_input_converted_to_utc():
    assert _ds(t0="2026-05-12T16:00+05:30").attrs["t0"] == "2026-05-12T10:30:00Z"


def test_violations():
    ds = _ds()
    assert any("mode" in p for p in validate_forecast(ds.assign_attrs(mode="radar")))
    bad = ds.copy()
    bad["lightning_prob_30"] = bad["lightning_prob_30"] + 2
    assert any("lightning_prob_30" in p for p in validate_forecast(bad))
    assert any("lead" in p for p in validate_forecast(ds.isel(lead=slice(0, 6))))
    assert any("first_flash" in p for p in validate_forecast(ds.drop_vars("first_flash")))


def test_first_flash_logic():
    p = np.array([[0.9, 0.9], [0.1, 0.6]])
    past = np.array([[True, False], [False, False]])
    np.testing.assert_array_equal(first_flash(p, past, 0.5), [[False, True], [False, True]])
