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
