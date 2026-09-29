"""Shared fixtures. Everything is built from nowcast_ml.data.synthetic; no downloads."""

from __future__ import annotations

from pathlib import Path

import pytest

ML_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ML_ROOT / "configs"


@pytest.fixture(scope="session")
def configs_dir() -> Path:
    return CONFIGS


@pytest.fixture(scope="session")
def synth_event():
    from nowcast_ml.data.synthetic import generate_event

    return generate_event(seed=7, size=32, n_frames=24, n_storms=(2, 2))


@pytest.fixture(scope="session")
def synth_ds(synth_event):
    from nowcast_ml.data.synthetic import to_dataset

    return to_dataset(synth_event)


@pytest.fixture(scope="session")
def events_dir(tmp_path_factory) -> Path:
    """Six small contract-valid synthetic event stores on disk."""
    from nowcast_ml.data.synthetic import write_events

    d = tmp_path_factory.mktemp("events")
    write_events(d, 6, seed=1, size=32, n_frames=24, radar_missing_prob=0.1)
    return d


@pytest.fixture(scope="session")
def tiny_cfg(configs_dir):
    """Smoke config shrunk further for fast unit tests (32x32, tiny widths)."""
    from nowcast_ml.config import load_config

    return load_config(
        configs_dir / "train" / "smoke.yaml",
        [
            "data.synthetic.n_events=6",
            "data.synthetic.size=32",
            "data.synthetic.n_frames=22",
            "data.sample_stride=3",
            "data.batch_size=4",
            "model.hid_s=8",
            "model.hid_t=16",
            "model.lightning_head.hidden=8",
            "train.max_epochs=1",
            "registry.save=false",
        ],
    )


@pytest.fixture(scope="session")
def artifact_root(tiny_cfg, events_dir, tmp_path_factory) -> Path:
    """Train a tiny model once and save it to a temporary registry; returns <root>/nowcast."""
    from nowcast_ml.training.train import run

    root = tmp_path_factory.mktemp("registry")
    cfg = tiny_cfg.model_copy(deep=True)
    cfg.data.events_dir = str(events_dir)
    cfg.train.output_dir = str(root / "run")
    cfg.registry.save = True
    cfg.registry.root = str(root / "models")
    run(cfg)
    return root / "models" / "nowcast"


@pytest.fixture()
def artifact_copy(artifact_root, tmp_path) -> Path:
    """A private copy of the artifact that a test may modify."""
    import shutil

    dst = tmp_path / "nowcast"
    shutil.copytree(artifact_root, dst)
    return dst
