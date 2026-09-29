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
