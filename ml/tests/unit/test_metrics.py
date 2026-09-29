import numpy as np
import pytest

from nowcast_ml.evaluation import metrics as M


def test_contingency_hand_case():
    f = np.array([[40, 40, 10, 10], [40, 10, 10, 50]], float)
    o = np.array([[40, 10, 40, 10], [40, 10, 10, np.nan]], float)
    t = M.contingency(f, o, 35)
    # hits: (0,0),(1,0); miss: (0,2); FA: (0,1),(1,3)  [NaN obs = no event]; CN: 3
    np.testing.assert_array_equal(t, [2, 1, 2, 3])
    s = M.scores_from_table(t)
    assert (
        s["csi"] == pytest.approx(2 / 5)
        and s["pod"] == pytest.approx(2 / 3)
        and s["far"] == pytest.approx(0.5)
    )
    v = np.ones_like(f, bool)
    v[1, 3] = False
    np.testing.assert_array_equal(M.contingency(f, o, 35, v), [2, 1, 1, 3])


def test_contingency_nan_forecast_counts_as_no_event():
    f = np.array([np.nan, 40.0])
    o = np.array([40.0, 40.0])
    np.testing.assert_array_equal(M.contingency(f, o, 35, np.ones(2, bool)), [1, 1, 0, 0])


def test_scores_undefined_when_empty():
    assert np.isnan(M.scores_from_table([0, 0, 0, 5])["csi"])


def test_fss():
    o = np.zeros((20, 20))
    o[5, 5] = 50
    assert M.fss(o, o, 35, 1) == 1.0
    shifted = np.roll(o, 2, axis=1)
    assert M.fss(shifted, o, 35, 1) == 0.0  # no overlap at grid scale
    assert 0 < M.fss(shifted, o, 35, 5) < 1  # partial credit at 5 px
    assert M.fss(shifted, o, 35, 5) < M.fss(shifted, o, 35, 9)
    # hand value, scale 3 on 1-D-like case: fractions 1/9 at 9 cells each, 3 overlap columns
    a = np.zeros((9, 9))
    a[4, 3] = 50
    b = np.zeros((9, 9))
    b[4, 4] = 50
    # Pf, Po each 1/9 on a 3x3 block; blocks overlap on 6 cells
    num = 6 * 0 + 6 * (1 / 9) ** 2  # 3+3 non-overlapping cells
    den = 2 * 9 * (1 / 9) ** 2
    assert M.fss(a, b, 35, 3) == pytest.approx(1 - num / den)


def test_brier_bss_hand():
    p = np.array([0.9, 0.1, 0.8, 0.3])
    o = np.array([1, 0, 0, 0])
    assert M.brier(p, o) == pytest.approx((0.01 + 0.01 + 0.64 + 0.09) / 4)
    acc = M.ProbAccumulator(n_bins=10)
    acc.update(p, o, refs={"persistence": np.array([1.0, 0, 1, 0])})
    r = acc.result()
    bs = (0.01 + 0.01 + 0.64 + 0.09) / 4
    assert r["brier"] == pytest.approx(bs)
    assert r["base_rate"] == 0.25
    assert r["bss_climatology"] == pytest.approx(1 - bs / (0.25 * 0.75))
    assert r["bss_persistence"] == pytest.approx(1 - bs / 0.25)
    assert r["roc_auc"] == 1.0  # the positive has the highest probability


def test_reliability_bins():
    r = M.reliability(np.array([0.05, 0.05, 0.95, 0.95]), np.array([0, 1, 1, 1]), n_bins=10)
    assert r["count"][0] == 2 and r["count"][9] == 2
    assert r["observed_freq"][0] == 0.5 and r["observed_freq"][9] == 1.0


def test_roc_auc_hand():
    assert M.roc_auc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 1, 1])) == pytest.approx(0.75)
    assert np.isnan(M.roc_auc(np.array([0.1]), np.array([1])))


def test_first_flash_hand_case():
    T, H, W = 10, 1, 3
    occ = np.zeros((T, H, W), bool)
    occ[6, 0, 0] = True  # onset at frame 6, pixel 0
    occ[7, 0, 1] = True  # onset at frame 7, pixel 1 (never warned -> miss)
    warn = {t0: np.zeros((H, W), bool) for t0 in range(2, 9)}
    warn[4][0, 0] = True  # warned 20 min ahead
    warn[5][0, 0] = True
    warn[3][0, 2] = True  # nothing happens at pixel 2 -> false alarm
    r = M.first_flash_scores(warn, occ, step_minutes=10)
    assert r["hits"] == 1 and r["misses"] == 1 and r["lead_times"] == [20]
    assert r["issued"] == 3 and r["false_alarms"] == 1
    s = M.summarize_first_flash([r])
    assert (
        s["hit_rate"] == 0.5
        and s["median_lead_min"] == 20
        and s["false_alarm_ratio"] == pytest.approx(1 / 3)
    )


def test_sharpness_blur_detected():
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    obs = gaussian_filter(rng.normal(size=(64, 64)), 1) * 20 + 30
    assert M.high_freq_power_ratio(obs, obs, 2.0) == pytest.approx(1.0)
    assert M.high_freq_power_ratio(gaussian_filter(obs, 3), obs, 2.0) < 0.2
