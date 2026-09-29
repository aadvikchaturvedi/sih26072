"""CLI smoke: train -> calibrate -> eval -> export -> predict against a registry artifact."""

import json

import pytest
import xarray as xr

from nowcast_ml.calibration.fit_cli import main as calibrate_main
from nowcast_ml.evaluation.evaluate import main as eval_main
from nowcast_ml.inference.predict_cli import main as predict_main
from nowcast_ml.inference.registry import load_artifact, resolve_version_dir
from nowcast_ml.inference.schema import validate_forecast


def test_cli_chain(artifact_copy, events_dir, tmp_path, capsys):
    latest = str(artifact_copy / "latest")
    calibrate_main(
        ["--model", latest, "--events", str(events_dir), "--split", "all", "--device", "cpu"]
    )
    eval_main(
        [
            "--model",
            latest,
            "--events",
            str(events_dir),
            "--split",
            "all",
            "--baselines",
            "persistence",
            "--out",
            str(tmp_path / "rep"),
            "--device",
            "cpu",
            "eval.sample_stride=6",
        ]
    )
    art = load_artifact(latest)  # hashes still valid after calibrator + metrics were added
    assert art.calibrator is not None and "eval_all" in art.metrics
    card = (resolve_version_dir(latest) / "model_card.md").read_text()
    assert "eval_all" in card and "Calibration" in card

    ev = sorted(events_dir.glob("*.zarr"))[0]
    out = tmp_path / "fc.zarr"
    predict_main(
        [
            "--model",
            latest,
            "--event",
            str(ev),
            "--t0",
            "auto",
            "--out",
            str(out),
            "--device",
            "cpu",
        ]
    )
    fc = xr.open_zarr(out).load()
    fc.attrs["missing_channels"] = json.loads(fc.attrs["missing_channels"])
    assert validate_forecast(fc) == []
    assert "mode=full" in capsys.readouterr().out

    t0 = str(xr.open_zarr(ev)["time"].values[9])[:16] + "Z"
    predict_main(
        [
            "--model",
            latest,
            "--event",
            str(ev),
            "--t0",
            t0,
            "--out",
            str(tmp_path / "b.zarr"),
            "--baseline",
            "extrapolation",
            "--device",
            "cpu",
        ]
    )


def test_predict_cli_exit_codes(artifact_root, events_dir, tmp_path):
    ev = sorted(events_dir.glob("*.zarr"))[0]
    with pytest.raises(SystemExit) as e:
        predict_main(
            [
                "--model",
                str(artifact_root / "latest"),
                "--event",
                str(ev),
                "--t0",
                "1999-01-01T00:00Z",
                "--out",
                str(tmp_path / "x.zarr"),
                "--device",
                "cpu",
            ]
        )
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        predict_main(
            [
                "--model",
                str(tmp_path / "nope"),
                "--event",
                str(ev),
                "--t0",
                "auto",
                "--out",
                str(tmp_path / "x.zarr"),
            ]
        )
    assert e.value.code == 3
