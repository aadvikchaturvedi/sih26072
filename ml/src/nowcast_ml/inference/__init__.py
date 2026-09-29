"""Backend-facing API.

``from nowcast_ml.inference import Predictor`` loads torch lazily; the typed
exceptions can be imported without torch via ``nowcast_ml.inference.errors``.
"""

from nowcast_ml.inference.errors import (
    InputContractError,
    ModelLoadError,
    NowcastError,
    OutputContractError,
)

__all__ = [
    "InputContractError",
    "ModelInfo",
    "ModelLoadError",
    "NowcastError",
    "OutputContractError",
    "Predictor",
]


def __getattr__(name):
    if name in ("Predictor", "ModelInfo"):
        from nowcast_ml.inference import predictor

        return getattr(predictor, name)
    raise AttributeError(name)
