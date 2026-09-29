"""M9: refiner stage -> artifact with refiner.pt -> Predictor members -> eval ensemble row."""

import json
import shutil

import numpy as np
import pytest
import torch
import xarray as xr

from nowcast_ml.calibration.fit_cli import main as calibrate_main
from nowcast_ml.evaluation.evaluate import main as eval_main
from nowcast_ml.inference import ModelLoadError, Predictor
from nowcast_ml.inference.registry import load_artifact, resolve_version_dir
from nowcast_ml.inference.schema import validate_forecast
from nowcast_ml.training.train import run


@pytest.fixture(scope="module")
def refined(tiny_cfg, events_dir, artifact_root, tmp_path_factory):
    root = tmp_path_factory.mktemp("refined")
    base = root / "nowcast"
    shutil.copytree(artifact_root, base)
    calibrate_main(
        [
            "--model",
            str(base / "latest"),
            "--events",
            str(events_dir),
            "--split",
            "all",
            "--device",
            "cpu",
        ]
    )
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.data.events_dir = str(events_dir)
    cfg.train.stage = "refiner"
    cfg.train.init_from = str(base / "latest")
    cfg.train.output_dir = str(root / "run")
    cfg.train.max_epochs = 1
    cfg.model.refiner_params.base_channels = 8
    cfg.model.refiner_params.channel_mults = [1, 2]
    cfg.model.refiner_params.timesteps = 100
    cfg.model.refiner_params.sample_steps = 4
    cfg.registry.save = True
    cfg.registry.root = str(root)
    res = run(cfg)
    return base, resolve_version_dir(res["artifact"])


def test_refiner_artifact(refined):
    _, d = refined
    art = load_artifact(d)
    assert art.config.model.refiner == "diffusion" and art.refiner_state is not None
    assert art.calibrator is not None  # carried over: the base model is frozen
    assert "Ensemble refiner: `diffusion`" in (d / "model_card.md").read_text()


def test_base_weights_frozen(refined, artifact_root):
    _, d = refined
    base = load_artifact(artifact_root / "latest").state_dict
    new = load_artifact(d).state_dict
    for k in base:
        assert torch.equal(base[k], new[k]), k


def test_predictor_members(refined, events_dir):
    _, d = refined
    p = Predictor.load(d, device="cpu")
    assert p.info.has_ensemble
    ds = xr.open_zarr(sorted(events_dir.glob("*.zarr"))[0]).isel(time=slice(0, 10)).load()
    t0 = ds["time"].values[-1]
    fc = p.predict(ds, t0, n_members=3, seed=7)
    assert validate_forecast(fc) == []
    assert fc["reflectivity_members"].shape == (3, 12, 32, 32)
    assert fc.attrs["n_members"] == 3
    again = p.predict(ds, t0, n_members=3, seed=7)
    np.testing.assert_array_equal(
        fc["reflectivity_members"].values, again["reflectivity_members"].values
    )
    plain = p.predict(ds, t0)
    assert "reflectivity_members" not in plain
    np.testing.assert_allclose(plain["reflectivity"].values, fc["reflectivity"].values)
    # satellite-only inputs also produce members
    sat = p.predict(ds.drop_sel(channel=["maxz", "cappi3km"]), t0, n_members=2)
    assert sat.attrs["mode"] == "satellite_only" and sat["reflectivity_members"].shape[0] == 2


def test_eval_has_ensemble_row_with_crps(refined, events_dir, tmp_path):
    _, d = refined
    out = tmp_path / "rep"
    eval_main(
        [
            "--model",
            str(d),
            "--events",
            str(events_dir),
            "--split",
            "all",
            "--baselines",
            "steps",
            "--out",
            str(out),
            "--device",
            "cpu",
            "eval.sample_stride=8",
            "eval.ensemble_members=2",
            "eval.steps_members=2",
        ]
    )
    m = json.loads((out / "metrics.json").read_text())["forecasters"]
    assert "model_ensemble" in m
    for name in ("model_ensemble", "steps", "model_full"):
        assert len(m[name]["reflectivity"]["crps_dbz"]) == 12
    assert "spread_skill_ratio" in m["model_ensemble"]["reflectivity"]
    assert "CRPS" in (out / "skill.md").read_text()


def test_missing_refiner_weights_fail_loudly(refined, tmp_path):
    from nowcast_ml.inference.registry import refresh_manifest

    _, d = refined
    dst = tmp_path / d.name
    shutil.copytree(d, dst)
    (dst / "refiner.pt").unlink()
    refresh_manifest(dst)
    with pytest.raises(ModelLoadError, match="refiner.pt"):
        load_artifact(dst)
