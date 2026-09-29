"""M6: lightning-head stage trains only the head; nowcast-calibrate writes a calibrator."""

import torch

from nowcast_ml.calibration.fit_cli import main as calibrate_main
from nowcast_ml.calibration.isotonic import IsotonicCalibrator
from nowcast_ml.training.lit_module import NowcastLitModule
from nowcast_ml.training.train import run


def test_lightning_head_stage_and_calibration(tiny_cfg, events_dir, tmp_path, capsys):
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.data.events_dir = str(events_dir)
    cfg.train.output_dir = str(tmp_path / "a")
    ck_a = run(cfg)["checkpoint"]

    cfg_c = cfg.model_copy(deep=True)
    cfg_c.train.stage = "lightning_head"
    cfg_c.train.trainable = "lightning_head"
    cfg_c.train.init_from = ck_a
    cfg_c.train.max_epochs = 2
    cfg_c.loss.refl_weight = 0.0
    cfg_c.train.output_dir = str(tmp_path / "c")
    ck_c = run(cfg_c)["checkpoint"]

    a = NowcastLitModule.from_checkpoint(ck_a).model.state_dict()
    c = NowcastLitModule.from_checkpoint(ck_c).model.state_dict()
    for k in a:
        if k.startswith("ltg_head."):
            continue
        assert torch.equal(a[k], c[k]), f"frozen parameter/buffer changed: {k}"
    assert any(not torch.equal(a[k], c[k]) for k in a if k.startswith("ltg_head."))
    # stage c reuses stage a normalization
    assert (tmp_path / "a" / "norm_stats.json").read_text() == (
        tmp_path / "c" / "norm_stats.json"
    ).read_text()

    calibrate_main(
        ["--model", ck_c, "--events", str(events_dir), "--split", "all", "--device", "cpu"]
    )
    out = capsys.readouterr().out
    assert "in-sample" in out
    cal = IsotonicCalibrator.load(tmp_path / "c" / "calibrator.pkl")
    assert set(cal.curves) == {"full", "satellite_only"} and len(cal.curves["full"]) == 2
