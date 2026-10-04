"""ForecastEngine backed by ``nowcast_ml.inference.Predictor``."""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd
import xarray as xr
from nowcast_ml.data.schema import validate_dataset
from nowcast_ml.inference.errors import InputContractError, NowcastError

from nowcast_backend.domain.errors import EngineError, InvalidObservations, InvalidRequest
from nowcast_backend.domain.models import SOURCES, ModelDescription


class PredictorEngine:
    """Serializes inference: one forecast at a time keeps memory use predictable."""

    def __init__(self, predictor):
        self._predictor = predictor
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls, path: str | Path, device: str = "auto", backend: str = "eager"
    ) -> PredictorEngine:
        """Raises ``nowcast_ml.inference.ModelLoadError`` if the artifact is unusable."""
        from nowcast_ml.inference import Predictor  # imports torch

        return cls(Predictor.load(path, device=device, backend=backend))

    def describe(self) -> ModelDescription:
        info = self._predictor.info
        return ModelDescription(
            name=info.name,
            version=info.version,
            channels=list(info.channels),
            grid_spacing_km=info.grid_spacing_km,
            t_in=info.t_in,
            step_minutes=self._predictor.config.data.step_minutes,
            lead_minutes=list(info.lead_minutes),
            lightning_leads_min=list(info.lightning_leads_min),
            calibrated=info.calibrated,
            has_ensemble=info.has_ensemble,
            backend=info.backend,
            device=info.device,
        )

    def validate(self, inputs: xr.Dataset) -> list[str]:
        return validate_dataset(
            inputs, expected_spacing_km=self._predictor.info.grid_spacing_km, min_frames=1
        )

    def forecast(
        self, inputs: xr.Dataset, t0: pd.Timestamp, *, source: str = "model", n_members: int = 0
    ) -> xr.Dataset:
        if source not in SOURCES:
            raise InvalidRequest(f"source must be one of {SOURCES}, got {source!r}")
        try:
            with self._lock:
                if source == "model":
                    return self._predictor.predict(inputs, t0, n_members=n_members)
                kwargs = {"n_members": n_members} if source == "steps" and n_members else {}
                return self._predictor.predict_baseline(inputs, t0, kind=source, **kwargs)
        except InputContractError as e:
            raise InvalidObservations(str(e).splitlines()[0], e.problems) from e
        except NowcastError as e:
            # e.g. ensemble members requested from a model without the refiner
            raise InvalidRequest(str(e)) from e
        except Exception as e:  # noqa: BLE001 - surface any engine failure as a typed error
            raise EngineError(f"{source} forecast failed: {type(e).__name__}: {e}") from e
