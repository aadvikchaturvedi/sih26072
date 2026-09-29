import numpy as np
import pytest
import torch

from nowcast_ml.data import channels as ch
from nowcast_ml.data.transforms import (
    ModalityDropout,
    NormStats,
    NormStatsMismatchError,
    drop_radar,
    fit_norm_stats,
    model_input,
    random_crop,
    random_flip,
)


def _stats(synth_event):
    x = np.stack([synth_event.fields[c] for c in ch.ALL_CHANNELS], axis=1)
    return fit_norm_stats([x], list(ch.ALL_CHANNELS)), x


def test_fit_matches_numpy(synth_event):
    stats, x = _stats(synth_event)
    i = ch.ALL_CHANNELS.index("tir1_bt")
    v = x[:, i]
    assert stats.mean[i] == pytest.approx(float(np.nanmean(v)), rel=1e-5)
    assert stats.std[i] == pytest.approx(float(np.nanstd(v, ddof=1)), rel=1e-4)


def test_streaming_equals_batch(synth_event):
    _, x = _stats(synth_event)
    a = fit_norm_stats([x], list(ch.ALL_CHANNELS))
    b = fit_norm_stats([x[:5], x[5:13], x[13:]], list(ch.ALL_CHANNELS))
    np.testing.assert_allclose(a.mean, b.mean, rtol=1e-6)
    np.testing.assert_allclose(a.std, b.std, rtol=1e-5)


def test_roundtrip_and_save_load(tmp_path, synth_event):
    stats, x = _stats(synth_event)
    p = tmp_path / "norm.json"
    stats.save(p)
    s2 = NormStats.load(p)
    assert s2 == stats
    for name in ("maxz", "flash_density", "tir1_bt"):
        i = ch.ALL_CHANNELS.index(name)
        v = np.nan_to_num(x[:, i])
        back = s2.denormalize_channel(name, s2.normalize_channel(name, v))
        np.testing.assert_allclose(back, v, atol=1e-3)
    xn = s2.normalize(x)
    assert np.isnan(xn).sum() == np.isnan(x).sum()  # NaN stays NaN
    xt = s2.normalize(torch.from_numpy(x))
    np.testing.assert_allclose(xt.numpy(), xn, atol=1e-4, equal_nan=True)


def test_channel_mismatch_raises(synth_event):
    stats, _ = _stats(synth_event)
    with pytest.raises(NormStatsMismatchError):
        stats.check_channels(["maxz", "tir1_bt"])


def test_unobserved_channel_gets_background():
    x = np.full((2, 2, 4, 4), np.nan, np.float32)
    x[:, 0] = 10.0
    s = fit_norm_stats([x], ["maxz", "tir1_bt"])
    assert s.mean[1] == ch.get("tir1_bt").background and s.std[1] == 1.0


def test_model_input_zeroes_unavailable():
    x = torch.randn(2, 3, 4, 5, 5)
    x[0, :, 1] = float("nan")
    a = torch.ones_like(x, dtype=torch.bool)
    a[0, :, 1] = False
    out = model_input(x, a)
    assert out.shape == (2, 3, 8, 5, 5)
    assert torch.isfinite(out).all()
    assert (out[0, :, 1] == 0).all() and (out[0, :, 5] == 0).all() and (out[1, :, 5] == 1).all()


def test_modality_dropout_masks_radar_only():
    chans = list(ch.ALL_CHANNELS)
    avail = torch.ones(64, 7, len(chans), 4, 4, dtype=torch.bool)
    out = ModalityDropout(chans, p_radar=1.0)(avail)
    radar = [chans.index(c) for c in ch.channels_in_group("radar")]
    other = [i for i in range(len(chans)) if i not in radar]
    assert not out[:, :, radar].any() and out[:, :, other].all()
    assert torch.equal(ModalityDropout(chans, p_radar=0.0)(avail), avail)
    g = torch.Generator().manual_seed(0)
    out = ModalityDropout(chans, p_radar=0.3)(
        torch.ones(4000, 1, len(chans), 1, 1, dtype=torch.bool), g
    )
    rate = 1 - out[:, 0, radar[0]].float().mean().item()
    assert 0.26 < rate < 0.34
    # a radar pixel that was already unavailable stays unavailable
    avail[0, 0, radar[0], 0, 0] = False
    assert not ModalityDropout(chans, p_radar=0.0)(avail)[0, 0, radar[0], 0, 0]


def test_drop_radar():
    chans = list(ch.ALL_CHANNELS)
    a = drop_radar(torch.ones(1, 2, len(chans), 3, 3, dtype=torch.bool), chans)
    assert not a[:, :, :2].any() and a[:, :, 2:].all()


def test_crop_and_flip_consistent():
    rng = np.random.default_rng(0)
    base = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)
    s = {
        "x": np.broadcast_to(base, (2, 3, 16, 16)).copy(),
        "y_refl": np.broadcast_to(base, (4, 16, 16)).copy(),
    }
    c = random_crop(s, 8, rng)
    assert c["x"].shape == (2, 3, 8, 8)
    np.testing.assert_array_equal(c["x"][0, 0], c["y_refl"][0])
    f = random_flip(c, rng)
    np.testing.assert_array_equal(f["x"][1, 2], f["y_refl"][3])
