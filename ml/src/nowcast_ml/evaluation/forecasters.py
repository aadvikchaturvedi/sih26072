"""Build the set of forecasters that evaluation scores with identical code."""

from __future__ import annotations

from pathlib import Path

from nowcast_ml.baselines import BASELINES, make_baseline
from nowcast_ml.config import Config
from nowcast_ml.inference.core import ModelForecaster, ModelRunner
from nowcast_ml.utils.device import resolve_device


def load_runner(path: str | Path, device: str = "auto") -> tuple[ModelRunner, Config, dict | None]:
    """Model runner + its config (+ splits if known) from a registry artifact or a Lightning .ckpt."""
    p = Path(path)
    dev = resolve_device(device)
    if p.suffix == ".ckpt":
        from nowcast_ml.training.lit_module import NowcastLitModule
        from nowcast_ml.utils.io import read_json

        lit = NowcastLitModule.from_checkpoint(str(p))
        model = lit.model.eval().to(dev)
        runner = ModelRunner(
            model, lit.norm_stats, lit.channels, dev, model.spatial_factor, name=p.stem
        )
        splits_file = p.parent.parent / "splits.json"
        splits = read_json(splits_file) if splits_file.exists() else None
        return runner, lit.cfg, splits
    from nowcast_ml.inference.predictor import Predictor

    pred = Predictor.load(p, device=device)
    return pred.runner, pred.config, pred.artifact.splits


def parse_baselines(spec: str | None) -> list[str]:
    if spec in (None, "", "none"):
        return []
    if spec == "all":
        return list(BASELINES)
    kinds = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [k for k in kinds if k not in BASELINES]
    if bad:
        raise ValueError(f"unknown baselines {bad}; choose from {BASELINES} or 'all'")
    return kinds


def build_forecasters(
    runner: ModelRunner | None,
    baselines: list[str],
    steps_members: int = 20,
    satellite_only_row: bool = True,
) -> list:
    fs = [make_baseline(k, n_members=steps_members) for k in baselines]
    if runner is not None:
        fs.append(ModelForecaster(runner, satellite_only=False))
        if satellite_only_row:
            fs.append(ModelForecaster(runner, satellite_only=True))
    return fs
