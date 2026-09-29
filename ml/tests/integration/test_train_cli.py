from nowcast_ml.training.train import main, run


def test_train_run_smoke(tiny_cfg, tmp_path):
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.train.output_dir = str(tmp_path / "run")
    res = run(cfg)
    assert res["checkpoint"].endswith(".ckpt")
    assert (tmp_path / "run" / "norm_stats.json").exists()
    assert (tmp_path / "run" / "splits.json").exists()
    assert "val/loss" in res["val_metrics"]


def test_train_cli_with_events_dir(configs_dir, events_dir, tmp_path, capsys):
    main(
        [
            "-c",
            str(configs_dir / "train" / "smoke.yaml"),
            f"data.events_dir={events_dir}",
            "model.hid_s=8",
            "model.hid_t=16",
            "train.max_epochs=1",
            f"train.output_dir={tmp_path}/run",
            "registry.save=false",
        ]
    )
    assert capsys.readouterr().out.strip().endswith(".ckpt")
