"""Reference forecasters sharing the :class:`~nowcast_ml.baselines.base.Forecaster` protocol."""

from nowcast_ml.baselines.base import Forecaster, ForecastInput
from nowcast_ml.baselines.extrapolation import ExtrapolationForecaster
from nowcast_ml.baselines.persistence import PersistenceForecaster
from nowcast_ml.baselines.steps_ensemble import StepsForecaster

BASELINES = ("persistence", "extrapolation", "steps")


def make_baseline(kind: str, n_members: int = 20, seed: int = 0) -> Forecaster:
    if kind == "persistence":
        return PersistenceForecaster()
    if kind == "extrapolation":
        return ExtrapolationForecaster()
    if kind == "steps":
        return StepsForecaster(n_members=n_members, seed=seed)
    raise ValueError(f"unknown baseline {kind!r}; choose from {BASELINES}")


__all__ = [
    "BASELINES",
    "ExtrapolationForecaster",
    "ForecastInput",
    "Forecaster",
    "PersistenceForecaster",
    "StepsForecaster",
    "make_baseline",
]
