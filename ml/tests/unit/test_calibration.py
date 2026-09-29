import numpy as np
import pytest

from nowcast_ml.calibration.isotonic import IsotonicCalibrator


def _miscalibrated(n, seed):
    rng = np.random.default_rng(seed)
    true_p = rng.uniform(0, 1, (n, 2))
    y = (rng.uniform(0, 1, (n, 2)) < true_p).astype(float)
    raw = true_p**3  # systematically under-confident
    return raw, y


def test_fit_improves_heldout_brier_and_is_monotone(tmp_path):
    raw, y = _miscalibrated(20000, 0)
    cal = IsotonicCalibrator.fit(
        {"full": raw, "satellite_only": raw * 0.5}, {"full": y, "satellite_only": y}, [30, 60]
    )
    raw2, y2 = _miscalibrated(20000, 1)
    cal_p = cal.apply(raw2.T[:, :, None], "full")[:, :, 0].T
    assert np.mean((cal_p - y2) ** 2) < 0.9 * np.mean((raw2 - y2) ** 2)
    grid = np.linspace(0, 1, 101)
    for curves in cal.curves.values():
        for c in curves:
            out = c(grid)
            assert np.all(np.diff(out) >= -1e-9) and out.min() >= 0 and out.max() <= 1
    p = tmp_path / "calibrator.pkl"
    cal.save(p)
    cal2 = IsotonicCalibrator.load(p)
    x = np.random.default_rng(2).uniform(0, 1, (2, 4, 4))
    np.testing.assert_array_equal(cal.apply(x, "satellite_only"), cal2.apply(x, "satellite_only"))
    assert not np.allclose(cal.apply(x, "full"), cal.apply(x, "satellite_only"))  # per-mode curves


def test_bad_pickle_rejected(tmp_path):
    import pickle

    p = tmp_path / "c.pkl"
    p.write_bytes(pickle.dumps({"not": "a calibrator"}))
    with pytest.raises(ValueError):
        IsotonicCalibrator.load(p)


def test_lead_count_mismatch():
    raw, y = _miscalibrated(500, 0)
    cal = IsotonicCalibrator.fit({"full": raw}, {"full": y}, [30, 60])
    with pytest.raises(ValueError, match="leads"):
        cal.apply(np.zeros((3, 2, 2)))
    np.testing.assert_array_equal(
        cal.apply(np.zeros((2, 2, 2)), "satellite_only"), cal.apply(np.zeros((2, 2, 2)), "full")
    )
