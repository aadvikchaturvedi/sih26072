"""``nowcast-export``: TorchScript + ONNX export with a parity check against eager.

    nowcast-export --model artifacts/models/nowcast/latest [--size 64] [--atol 1e-4]

Writes ``model.ts`` and (if export succeeds) ``model.onnx`` into the artifact,
plus ``export_report.json`` with max |difference| vs eager on the export size
and on a second, different spatial size (both exports use dynamic H/W).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

from nowcast_ml.inference.registry import load_artifact, refresh_manifest, write_model_card
from nowcast_ml.models.nowcast_model import NowcastModel
from nowcast_ml.utils.io import write_json
from nowcast_ml.utils.logging import get_logger

log = get_logger(__name__)
OPSET = 17


def _example(model: NowcastModel, size: int, batch: int = 1, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, model.t_in, model.in_channels, size, size, generator=g)
    C = model.n_channels
    x[:, :, C:] = (
        torch.rand(batch, model.t_in, C, size, size, generator=g) > 0.2
    ).float()  # masks in {0,1}
    return x


def _maxdiff(a: tuple, b: tuple) -> float:
    return max(
        float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(a, b, strict=True)
    )


def export_torchscript(model: NowcastModel, path: Path, size: int) -> torch.jit.ScriptModule:
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts = torch.jit.trace(model, _example(model, size), check_trace=False)
        ts = torch.jit.freeze(ts.eval())
    ts.save(str(path))
    return torch.jit.load(str(path))


def export_onnx(model: NowcastModel, path: Path, size: int) -> None:
    x = _example(model, size)
    kw = dict(
        input_names=["x"],
        output_names=["reflectivity_norm", "lightning_logits"],
        opset_version=OPSET,
    )
    axes = {"x": {0: "batch", 3: "height", 4: "width"}}
    axes_out = {
        "reflectivity_norm": {0: "batch", 2: "height", 3: "width"},
        "lightning_logits": {0: "batch", 2: "height", 3: "width"},
    }
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            torch.onnx.export(
                model, (x,), str(path), dynamo=False, dynamic_axes={**axes, **axes_out}, **kw
            )
        except Exception as legacy_err:  # noqa: BLE001
            try:
                import onnxscript  # noqa: F401
            except ImportError:
                raise legacy_err from None
            h, w = torch.export.Dim("height", min=8), torch.export.Dim("width", min=8)
            torch.onnx.export(
                model, (x,), str(path), dynamo=True, dynamic_shapes=({3: h, 4: w},), **kw
            )


def run_export(
    model_path: str | Path, size: int | None = None, atol: float = 1e-4, onnx: bool = True
) -> dict:
    art = load_artifact(model_path)
    cfg = art.config
    model = NowcastModel(
        cfg.model,
        len(art.channels),
        cfg.data.t_in,
        cfg.data.t_out,
        len(cfg.data.lightning_leads_min),
    )
    model.load_state_dict(art.state_dict, strict=True)
    model.eval()
    f = model.spatial_factor
    size = size or max(32, (cfg.data.crop or 64))
    size = int(np.ceil(size / f) * f)
    size2 = size + 2 * f  # a second, different size to prove dynamic H/W works
    report: dict = {"export_size": size, "check_sizes": [size, size2], "atol": atol}

    with torch.no_grad():
        refs = {
            s: tuple(t.numpy() for t in model(_example(model, s, seed=s))) for s in (size, size2)
        }

    ts = export_torchscript(model, art.file("model.ts"), size)
    with torch.no_grad():
        d = max(
            _maxdiff(refs[s], tuple(t.numpy() for t in ts(_example(model, s, seed=s))))
            for s in refs
        )
    report["torchscript"] = {"ok": d <= atol, "max_abs_diff": d}
    if d > atol:
        raise RuntimeError(f"TorchScript parity failed: max |diff| {d:.2e} > {atol}")

    if onnx:
        onnx_path = art.file("model.onnx")
        try:
            import onnxruntime as ort

            export_onnx(model, onnx_path, size)
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            d = max(
                _maxdiff(refs[s], tuple(sess.run(None, {"x": _example(model, s, seed=s).numpy()})))
                for s in refs
            )
            report["onnx"] = {"ok": d <= atol, "max_abs_diff": d, "opset": OPSET}
            if d > atol:
                onnx_path.unlink(missing_ok=True)
                report["onnx"]["note"] = "parity failed; model.onnx removed"
        except Exception as e:  # noqa: BLE001 - ONNX is optional; record why it failed
            onnx_path.unlink(missing_ok=True)
            report["onnx"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:500]}
            log.warning("ONNX export skipped: %s", report["onnx"]["error"])

    write_json(art.file("export_report.json"), report)
    write_model_card(art.path)
    refresh_manifest(art.path)
    report["artifact"] = str(art.path)
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="nowcast-export",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="artifact dir or .../latest")
    ap.add_argument(
        "--size", type=int, help="spatial size used for tracing (default: training crop or 64)"
    )
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args(argv)
    r = run_export(args.model, args.size, args.atol, onnx=not args.no_onnx)
    print(f"torchscript: max|diff|={r['torchscript']['max_abs_diff']:.2e}")
    if "onnx" in r:
        o = r["onnx"]
        print(
            "onnx: "
            + (
                f"max|diff|={o['max_abs_diff']:.2e}"
                if "max_abs_diff" in o
                else f"FAILED ({o['error']})"
            )
        )
    print(r["artifact"])


if __name__ == "__main__":
    main()
