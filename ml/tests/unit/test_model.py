import pytest
import torch

from nowcast_ml.config import ModelConfig
from nowcast_ml.data import channels as ch
from nowcast_ml.data.transforms import drop_radar, model_input
from nowcast_ml.models import NowcastModel
from nowcast_ml.models.backbones import build_backbone
from nowcast_ml.models.simvp import sampling_generator

CH = list(ch.ALL_CHANNELS)


def _cfg(**kw):
    base = dict(hid_s=8, hid_t=16, n_s=4, n_t=2)
    base.update(kw)
    return ModelConfig(**base)


def test_sampling():
    assert sampling_generator(4) == [False, True, False, True]
    assert sampling_generator(2, reverse=True) == [True, False]


@pytest.mark.parametrize("n_s,factor", [(2, 2), (4, 4)])
def test_output_shapes_full_and_radar_missing(n_s, factor):
    torch.manual_seed(0)
    m = NowcastModel(_cfg(n_s=n_s), n_channels=len(CH), t_in=7, t_out=12, n_ltg=2).eval()
    assert m.spatial_factor == factor
    x = torch.randn(2, 7, len(CH), 32, 48)
    avail = torch.ones_like(x, dtype=torch.bool)
    with torch.no_grad():
        refl, ltg = m(model_input(x, avail))
        assert refl.shape == (2, 12, 32, 48) and ltg.shape == (2, 2, 32, 48)
        x_sat = x.clone()
        x_sat[:, :, :2] = float("nan")  # radar missing -> NaN, masked out
        refl2, ltg2 = m(model_input(x_sat, drop_radar(avail, CH)))
    assert refl2.shape == refl.shape and torch.isfinite(refl2).all() and torch.isfinite(ltg2).all()
    assert not torch.allclose(refl, refl2)  # the mask actually changes the input


def test_lightning_head_disabled():
    cfg = _cfg()
    cfg.lightning_head.enabled = False
    m = NowcastModel(cfg, n_channels=3).eval()
    _, ltg = m(torch.zeros(1, 7, 6, 16, 16))
    assert torch.sigmoid(ltg).max() < 1e-6


def test_unknown_backbone():
    with pytest.raises(ValueError, match="unknown backbone"):
        build_backbone(_cfg(backbone="convlstm"), 4, 7)
