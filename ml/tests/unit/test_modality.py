"""M5: modality dropout reaches training, and satellite-only mode is detected and applied."""

import numpy as np
import torch
from torch.utils.data import default_collate

from nowcast_ml.data import channels as ch
from nowcast_ml.inference.core import ModelRunner, detect_mode
from nowcast_ml.training.datamodule import NowcastDataModule
from nowcast_ml.training.lit_module import NowcastLitModule

CH = list(ch.ALL_CHANNELS)


def test_detect_mode():
    a = np.ones((7, len(CH), 10, 10), bool)
    assert detect_mode(a, CH) == "full"
    a[-1, :2] = False
    assert detect_mode(a, CH) == "satellite_only"
    a[-1, 0, :1, :1] = True  # 1% coverage is below the 5% default
    assert detect_mode(a, CH) == "satellite_only"
    a[-1, 0, :5, :5] = True  # 25%
    assert detect_mode(a, CH) == "full"
    assert detect_mode(a, ["tir1_bt"]) == "satellite_only"  # model without radar channels


def _lit(tiny_cfg, p_radar):
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.train.modality_dropout.p_radar = p_radar
    dm = NowcastDataModule(cfg)
    dm.setup()
    batch = default_collate([dm.sources.dataset("train")[i] for i in range(4)])
    return NowcastLitModule(cfg, dm.norm_stats), batch, dm


def test_training_step_applies_radar_dropout(tiny_cfg):
    lit, batch, _ = _lit(tiny_cfg, 1.0)
    C = len(CH)
    xin = lit.prepare_batch(batch, train=True)
    assert not xin[:, :, C : C + 2].any()  # radar availability channels all zero
    assert xin[:, :, C + 2 :].any()
    xin_val = lit.prepare_batch(batch, train=False)  # no dropout at validation
    assert xin_val[:, :, C].any()


def test_runner_satellite_only_ignores_radar(tiny_cfg):
    lit, batch, dm = _lit(tiny_cfg, 0.3)
    m = lit.model.eval()
    runner = ModelRunner(m, dm.norm_stats, CH, torch.device("cpu"), m.spatial_factor)
    x, a = batch["x"].numpy(), batch["avail"].numpy()
    d_sat, p_sat = runner.forecast_batch(x, a, satellite_only=True)
    x2 = x.copy()
    x2[:, :, :2] += 30.0  # change radar values: must not affect satellite-only output
    d_sat2, _ = runner.forecast_batch(x2, a, satellite_only=True)
    np.testing.assert_allclose(d_sat, d_sat2)
    d_full, _ = runner.forecast_batch(x, a, satellite_only=False)
    assert not np.allclose(d_full, d_sat)


def test_runner_pads_odd_domain(tiny_cfg):
    lit, batch, dm = _lit(tiny_cfg, 0.0)
    m = lit.model.eval()
    runner = ModelRunner(m, dm.norm_stats, CH, torch.device("cpu"), m.spatial_factor)
    x = batch["x"].numpy()[:1, :, :, :30, :29]
    a = batch["avail"].numpy()[:1, :, :, :30, :29]
    d, p = runner.forecast_batch(x, a)
    assert d.shape == (1, 12, 30, 29) and p.shape == (1, 2, 30, 29)
    assert ((p >= 0) & (p <= 1)).all() and (d >= 0).all()
