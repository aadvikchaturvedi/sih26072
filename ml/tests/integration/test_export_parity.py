import json

import pytest

from nowcast_ml.inference.export import main as export_main
from nowcast_ml.inference.registry import load_artifact, resolve_version_dir


def test_torchscript_and_onnx_parity(artifact_copy, capsys):
    export_main(["--model", str(artifact_copy / "latest"), "--size", "32"])
    d = resolve_version_dir(artifact_copy)
    rep = json.loads((d / "export_report.json").read_text())
    assert rep["torchscript"]["ok"] and rep["torchscript"]["max_abs_diff"] <= 1e-4
    assert rep["onnx"]["ok"], rep["onnx"]
    assert rep["onnx"]["max_abs_diff"] <= 1e-4
    assert (d / "model.ts").exists() and (d / "model.onnx").exists()
    load_artifact(d)  # manifest refreshed: still loads with hash verification

    from nowcast_ml.inference import Predictor

    p_ts = Predictor.load(d, device="cpu", backend="torchscript")
    assert p_ts.info.backend == "torchscript"


def test_torchscript_backend_requires_export(artifact_copy):
    from nowcast_ml.inference import ModelLoadError, Predictor

    with pytest.raises(ModelLoadError, match="nowcast-export"):
        Predictor.load(artifact_copy, device="cpu", backend="torchscript")
