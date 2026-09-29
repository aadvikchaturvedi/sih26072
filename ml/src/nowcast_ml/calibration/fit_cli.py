"""``nowcast-calibrate``: training stage (d), isotonic calibration on the validation split.

    nowcast-calibrate --model artifacts/models/nowcast/latest --split val

Collects raw lightning probabilities for full and satellite-only inputs on the
split's events, fits one isotonic curve per (mode, lead), reports Brier before and
after (in-sample on the fitting split; use ``nowcast-eval`` on test for skill), and writes ``calibrator.pkl`` into the artifact (or next to a .ckpt run).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from nowcast_ml.calibration.isotonic import MODES, IsotonicCalibrator
from nowcast_ml.data.zarr_events import WindowSpec, make_sample, valid_t0_indices
from nowcast_ml.evaluation.evaluate import resolve_events
from nowcast_ml.evaluation.forecasters import load_runner
from nowcast_ml.utils.io import write_json
from nowcast_ml.utils.logging import get_logger

log = get_logger(__name__)


def collect(runner, cfg, events, stride: int, max_pixels_per_sample: int = 20_000, seed: int = 0):
    spec = WindowSpec.from_config(cfg.data)
    rng = np.random.default_rng(seed)
    probs = {m: [] for m in MODES}
    labels = {m: [] for m in MODES}
    for _, ev in events:
        for t0 in valid_t0_indices(ev.n_times, spec, stride):
            s = make_sample(ev, t0, spec)
            valid = s["y_ltg_valid"].all(axis=0)
            idx = np.flatnonzero(valid)
            if idx.size == 0:
                continue
            if idx.size > max_pixels_per_sample:
                idx = rng.choice(idx, max_pixels_per_sample, replace=False)
            y = s["y_ltg"].reshape(len(spec.leads_min), -1)[:, idx].T
            for mode in MODES:
                _, p = runner.forecast_batch(
                    s["x"][None], s["avail"][None], satellite_only=mode == "satellite_only"
                )
                probs[mode].append(p[0].reshape(len(spec.leads_min), -1)[:, idx].T)
                labels[mode].append(y)
    return (
        {m: np.concatenate(v) for m, v in probs.items() if v},
        {m: np.concatenate(v) for m, v in labels.items() if v},
    )


def calibrate(
    model: str, events_dir: str | None, split: str, device: str = "cpu", out: str | None = None
) -> dict:
    runner, cfg, splits = load_runner(model, device)
    runner.calibrator = None  # always fit on raw probabilities
    events = resolve_events(cfg, events_dir, split, splits)
    probs, labels = collect(runner, cfg, events, cfg.eval.sample_stride)
    if not probs:
        raise SystemExit("no valid lightning labels in the calibration split")
    leads = list(cfg.data.lightning_leads_min)
    cal = IsotonicCalibrator.fit(probs, labels, leads)
    cal.meta.update(split=split, events=[e for e, _ in events])
    report = {}
    for mode in probs:
        p, y = probs[mode], labels[mode]
        pc = np.stack([c(p[:, i]) for i, c in enumerate(cal.curves[mode])], axis=1)
        report[mode] = {
            str(L): {
                "brier_raw": float(np.mean((p[:, i] - y[:, i]) ** 2)),
                "brier_calibrated_in_sample": float(np.mean((pc[:, i] - y[:, i]) ** 2)),
                "base_rate": float(y[:, i].mean()),
                "n": int(len(y)),
            }
            for i, L in enumerate(leads)
        }
    cal.meta["fit_report"] = report
    target = _target_path(model, out)
    cal.save(target)
    write_json(target.with_name("calibration_report.json"), report)
    if Path(model).suffix != ".ckpt" and out is None:
        from nowcast_ml.inference.registry import refresh_manifest, write_model_card

        write_model_card(target.parent)
        refresh_manifest(target.parent)
    log.info("wrote %s", target)
    return {"path": str(target), "report": report}


def _target_path(model: str, out: str | None) -> Path:
    if out:
        return Path(out)
    p = Path(model)
    if p.suffix == ".ckpt":
        return p.parent.parent / "calibrator.pkl"
    from nowcast_ml.inference.registry import resolve_version_dir

    return resolve_version_dir(p) / "calibrator.pkl"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="nowcast-calibrate",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", required=True, help="artifact dir (or .../latest) or .ckpt")
    ap.add_argument("--events", help="event Zarr directory (default: from the model config)")
    ap.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", help="output path (default: inside the artifact)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    res = calibrate(args.model, args.events, args.split, args.device, args.out)
    for mode, r in res["report"].items():
        for L, v in r.items():
            print(
                f"{mode:15s} +{L}min  Brier raw {v['brier_raw']:.4f} -> calibrated {v['brier_calibrated_in_sample']:.4f} (in-sample)  (n={v['n']})"
            )
    print(res["path"])


if __name__ == "__main__":
    main()
