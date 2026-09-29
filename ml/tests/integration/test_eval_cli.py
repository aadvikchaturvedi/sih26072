import json

from nowcast_ml.evaluation.evaluate import main as eval_main
from nowcast_ml.training.train import run


def test_eval_model_and_baselines(tiny_cfg, events_dir, tmp_path):
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.train.output_dir = str(tmp_path / "run")
    cfg.data.events_dir = str(events_dir)
    ckpt = run(cfg)["checkpoint"]
    out = tmp_path / "report"
    eval_main(
        [
            "--model",
            ckpt,
            "--events",
            str(events_dir),
            "--split",
            "all",
            "--baselines",
            "all",
            "--out",
            str(out),
            "--device",
            "cpu",
            "eval.steps_members=2",
            "eval.sample_stride=4",
        ]
    )
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["forecasters"]) == {
        "persistence",
        "extrapolation",
        "steps",
        "model_full",
        "model_satellite_only",
    }
    assert m["meta"]["synthetic"] is True
    for f in ("skill.md", "csi_vs_lead.png", "reliability.png", "case_study.png"):
        assert (out / f).stat().st_size > 0, f
    md = (out / "skill.md").read_text()
    assert "SYNTHETIC DATA" in md and "Model (satellite-only)" in md
    csi = m["forecasters"]["persistence"]["reflectivity"]["csi"]["20"]
    assert len(csi) == 12


def test_eval_baselines_only_with_config(configs_dir, events_dir, tmp_path):
    out = tmp_path / "b"
    eval_main(
        [
            "--config",
            str(configs_dir / "train" / "smoke.yaml"),
            "--events",
            str(events_dir),
            "--split",
            "all",
            "--baselines",
            "persistence,extrapolation",
            "--out",
            str(out),
            "eval.sample_stride=6",
        ]
    )
    m = json.loads((out / "metrics.json").read_text())
    assert set(m["forecasters"]) == {"persistence", "extrapolation"}
