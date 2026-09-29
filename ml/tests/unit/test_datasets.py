import numpy as np

from nowcast_ml.data import channels as ch
from nowcast_ml.data.event import Event
from nowcast_ml.data.zarr_events import (
    WindowSpec,
    list_event_paths,
    make_sample,
    zarr_event_dataset,
)


def test_event_reorders_and_masks_missing_channels(synth_ds):
    chans = ["tir1_bt", "maxz", "cape"]
    sub = synth_ds.sel(channel=["maxz", "tir1_bt"])  # cape absent
    ev = Event(sub, chans)
    x, a = ev.window(0, 7)
    assert x.shape == (7, 3, 32, 32)
    np.testing.assert_array_equal(x[:, 0], synth_ds["x"].sel(channel="tir1_bt").values[:7])
    assert not a[:, 2].any() and np.isnan(x[:, 2]).all()
    assert ev.missing_channels == ["cape"]


def test_radar_missing_mask_propagates(synth_ds):
    ds = synth_ds.copy(deep=True)
    ds["missing"].loc[{"group": "radar"}] = 1
    x, a = Event(ds, list(ch.ALL_CHANNELS)).window(0, 7)
    assert not a[:, :2].any() and a[:, 2:].all()


def test_zarr_dataset_windows(events_dir):
    paths = list_event_paths(events_dir)
    spec = WindowSpec()
    dset = zarr_event_dataset(paths, list(ch.ALL_CHANNELS), spec, stride=1)
    assert len(dset) == 6 * (24 - 12 - 7 + 1)
    s = dset[0]
    assert s["x"].shape == (7, 14, 32, 32) and s["avail"].dtype == bool
    assert s["y_refl"].shape == (12, 32, 32) and s["y_ltg"].shape == (2, 32, 32)
    assert s["past_ltg"].shape == (32, 32)


def test_sample_targets_align_with_inputs(synth_ds):
    ev = Event(synth_ds, list(ch.ALL_CHANNELS))
    s = make_sample(ev, 10, WindowSpec())
    full = np.nan_to_num(synth_ds["x"].sel(channel="maxz").values)
    np.testing.assert_array_equal(np.nan_to_num(s["x"][-1, 0]), full[10])
    np.testing.assert_array_equal(np.nan_to_num(s["y_refl"][0]), full[11])


def test_flash_points_match_density(synth_ds):
    ev = Event(synth_ds, list(ch.ALL_CHANNELS))
    from_points = ev.lightning_occurrence
    from_density = Event(
        synth_ds.drop_vars(["flash_time", "flash_lat", "flash_lon"]), list(ch.ALL_CHANNELS)
    ).lightning_occurrence
    assert from_density.sum() > 0
    # jittered points land in the same or an adjacent pixel
    from nowcast_ml.data.labels import dilate

    assert (from_points <= dilate(from_density, 1)).all()
