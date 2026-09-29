import numpy as np

from nowcast_ml.data import channels as ch
from nowcast_ml.data.synthetic import generate_event


def test_deterministic():
    a = generate_event(3, size=32, n_frames=10)
    b = generate_event(3, size=32, n_frames=10)
    for n in ch.ALL_CHANNELS:
        np.testing.assert_array_equal(a.fields[n], b.fields[n])


def test_shapes_and_ranges(synth_event):
    e = synth_event
    for n, arr in e.fields.items():
        assert arr.shape == (24, 32, 32) and arr.dtype == np.float32
        c = ch.get(n)
        finite = arr[np.isfinite(arr)]
        assert finite.min() >= c.valid_min and finite.max() <= c.valid_max, n
    assert e.lat.shape == (32, 32) and e.lat[0, 0] > e.lat[-1, 0]  # row 0 = north


def test_storm_moves_with_steering_flow():
    e = generate_event(11, size=64, n_frames=12, n_storms=(1, 1), motion=(0.0, 2.0))
    z = np.nan_to_num(e.fields["maxz"])
    xs = np.arange(64)
    cx = [(z[t].sum(0) * xs).sum() / max(z[t].sum(), 1e-6) for t in (2, 8)]
    assert cx[1] - cx[0] > 6  # moved east ~12 px


def test_radar_missing_is_nan_not_zero():
    e = generate_event(5, size=32, n_frames=30, radar_missing_prob=0.5)
    m = e.missing["radar"]
    assert m.any()
    assert np.isnan(e.fields["maxz"][m]).all()
    assert np.isfinite(e.fields["tir1_bt"]).all()
