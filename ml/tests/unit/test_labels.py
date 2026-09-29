import numpy as np
import pandas as pd

from nowcast_ml.data import labels as lb


def test_radius_px():
    assert lb.radius_px(10, 2.0) == 5
    assert lb.radius_px(10, 3.0) == 3


def test_hand_built_case():
    """One flash at frame 2, pixel (10, 10); 2 km grid, 10 km radius = 5 px."""
    T, H, W = 10, 21, 21
    occ = np.zeros((T, H, W), bool)
    occ[2, 10, 10] = True

    lab = lb.lightning_labels(occ, t0=0, leads_min=(30, 60), radius=5)
    assert lab.shape == (2, H, W)
    # frames 1..3 are in the +30 window -> flash counted
    assert lab[0, 10, 10] == 1 and lab[0, 10, 15] == 1 and lab[0, 10, 16] == 0
    assert lab[0, 14, 13] == 1  # 3-4-5 triangle: distance 5
    assert lab[0, 14, 14] == 0  # distance 5.66 > 5
    assert lab[1].sum() == lab[0].sum()  # also inside +60

    # t0 = 2: the flash is in the past, not the future
    assert lb.lightning_labels(occ, t0=2, radius=5).sum() == 0
    assert lb.past_lightning(occ, t0=2, radius=5)[10, 10]
    assert lb.past_lightning(occ, t0=4, radius=5)[10, 10]  # frames 2..4
    assert not lb.past_lightning(occ, t0=5, radius=5).any()  # frames 3..5

    # flash only at frame 5 -> in +60 (frames 1..6) but not +30 (frames 1..3)
    occ2 = np.zeros((T, H, W), bool)
    occ2[5, 3, 3] = True
    lab2 = lb.lightning_labels(occ2, t0=0, radius=0)
    assert lab2[0].sum() == 0 and lab2[1, 3, 3] == 1 and lab2[1].sum() == 1


def test_label_validity():
    miss = np.zeros((10, 4, 4), bool)
    miss[5, 1, 1] = True
    v = lb.lightning_label_valid(miss, t0=0)
    assert v[0].all() and not v[1, 1, 1] and v[1].sum() == 15
    assert not lb.lightning_label_valid(miss, t0=6)[1].any()  # horizon beyond event


def test_occurrence_from_points():
    times = pd.date_range("2026-05-12T10:00", periods=4, freq="10min").values
    lat = np.repeat(np.array([22.02, 22.0, 21.98])[:, None], 3, axis=1)
    lon = np.repeat(np.array([[78.0, 78.02, 78.04]]), 3, axis=0)
    ft = np.array(
        ["2026-05-12T10:15", "2026-05-12T10:20", "2026-05-12T12:00"], dtype="datetime64[ns]"
    )
    occ = lb.occurrence_from_points(
        ft, np.array([22.0, 21.98, 22.0]), np.array([78.02, 78.0, 78.02]), times, lat, lon
    )
    assert occ[2, 1, 1] and occ[2, 2, 0]  # 10:15 and 10:20 both belong to frame (10:10, 10:20]
    assert occ.sum() == 2  # 12:00 is after the event
