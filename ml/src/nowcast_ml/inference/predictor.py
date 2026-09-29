"""Backend-facing predictor.

Load once, call many times::

    from nowcast_ml.inference import Predictor

    predictor = Predictor.load("artifacts/models/nowcast/latest", device="auto")
    forecast = predictor.predict(inputs, t0)          # xr.Dataset (see inference/schema.py)
    baseline = predictor.predict_baseline(inputs, t0, kind="extrapolation")

``Predictor`` holds only read-only state after ``load`` (weights, normalization,
calibrator); ``predict`` has no side effects and is safe to call repeatedly.
Errors are typed: :class:`InputContractError` for bad inputs and
:class:`ModelLoadError` for bad artifacts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import xarray as xr

from nowcast_ml.baselines import BASELINES, ForecastInput, make_baseline
from nowcast_ml.config import Config
from nowcast_ml.data import labels as lb
from nowcast_ml.data.event import Event
from nowcast_ml.data.schema import validate_dataset
from nowcast_ml.inference.core import ModelRunner
from nowcast_ml.inference.errors import (
    InputContractError,
    ModelLoadError,
    NowcastError,
    OutputContractError,
)
from nowcast_ml.inference.registry import Artifact, load_artifact
from nowcast_ml.inference.schema import (
    ForecastArrays,
    build_forecast_dataset,
    to_utc_timestamp,
    validate_forecast,
)
from nowcast_ml.models.nowcast_model import NowcastModel
from nowcast_ml.models.refiner import build_refiner
from nowcast_ml.utils.device import resolve_device

BaselineKind = Literal["persistence", "extrapolation", "steps"]


@dataclass(frozen=True)
class ModelInfo:
    name: str
    version: str
    channels: tuple[str, ...]
    grid_spacing_km: float
    t_in: int
    lead_minutes: tuple[int, ...]
    lightning_leads_min: tuple[int, ...]
    calibrated: bool
    has_ensemble: bool
    backend: str
    device: str


class Predictor:
    """Stateless nowcast predictor. Construct with :meth:`load`."""

    def __init__(self, artifact: Artifact, runner: ModelRunner, backend: str):
        self.artifact = artifact
        self.runner = runner
        self.config: Config = artifact.config
        self._backend = backend

    # ------------------------------------------------------------------ load
    @classmethod
    def load(
        cls,
        path: str | Path,
        device: str = "auto",
        backend: Literal["eager", "torchscript"] = "eager",
        verify_hashes: bool = True,
    ) -> Predictor:
        """Load an artifact (``.../<name>/latest``, ``.../<name>`` or a version dir).

        Raises :class:`ModelLoadError` if files are missing or modified, or if channels,
        normalization, calibrator and weights disagree.
        """
        art = load_artifact(path, verify_hashes=verify_hashes)
        cfg = art.config
        dev = resolve_device(device)
        model = NowcastModel(
            cfg.model,
            len(art.channels),
            cfg.data.t_in,
            cfg.data.t_out,
            len(cfg.data.lightning_leads_min),
        )
        try:
            model.load_state_dict(art.state_dict, strict=True)
        except RuntimeError as e:
            raise ModelLoadError(
                f"artifact {art.path}: weights do not match the configured model: {e}"
            ) from e
        spatial_factor = model.spatial_factor
        if backend == "torchscript":
            ts = art.file("model.ts")
            if not ts.exists():
                raise ModelLoadError(
                    f"artifact {art.path} has no model.ts; run nowcast-export first"
                )
            model = torch.jit.load(str(ts), map_location=dev)
        elif backend != "eager":
            raise ValueError(f"backend must be 'eager' or 'torchscript', got {backend!r}")
        model = model.to(dev).eval()
        refiner = None
        if art.refiner_state is not None:
            refiner = build_refiner(
                cfg.model.refiner, cfg.data.t_out, 2 * len(art.channels), cfg.model.refiner_params
            )
            try:
                refiner.load_state_dict(art.refiner_state, strict=True)
            except RuntimeError as e:
                raise ModelLoadError(
                    f"artifact {art.path}: refiner weights do not match config: {e}"
                ) from e
            refiner = refiner.to(dev).eval()
        runner = ModelRunner(
            model=model,
            norm_stats=art.norm_stats,
            channels=art.channels,
            device=dev,
            spatial_factor=spatial_factor,
            calibrator=art.calibrator,
            name=art.name,
            min_radar_coverage=cfg.inference.satellite_only_radar_coverage,
            refiner=refiner,
        )
        return cls(art, runner, backend)

    @property
    def info(self) -> ModelInfo:
        d = self.config.data
        return ModelInfo(
            name=self.artifact.name,
            version=self.artifact.version,
            channels=tuple(self.artifact.channels),
            grid_spacing_km=d.grid_spacing_km,
            t_in=d.t_in,
            lead_minutes=tuple(d.lead_minutes),
            lightning_leads_min=tuple(d.lightning_leads_min),
            calibrated=self.artifact.calibrator is not None,
            has_ensemble=self.runner.refiner is not None,
            backend=self._backend,
            device=str(self.runner.device),
        )

    # ------------------------------------------------------------------ inputs
    def _prepare(self, inputs: xr.Dataset, t0: datetime | str | pd.Timestamp):
        if not isinstance(inputs, xr.Dataset):
            raise InputContractError(
                f"inputs must be an xarray.Dataset, got {type(inputs).__name__}"
            )
        d = self.config.data
        problems = validate_dataset(
            inputs, expected_spacing_km=d.grid_spacing_km, min_frames=d.t_in
        )
        if problems:
            raise InputContractError("inputs violate the input contract", problems)
        ev = Event(inputs, self.artifact.channels)
        try:
            t0_ts = to_utc_timestamp(t0)
            i0 = ev.index_of(t0_ts)
        except (KeyError, ValueError) as e:
            raise InputContractError(
                f"t0={t0} is not one of the input times ({ev.times[0]} .. {ev.times[-1]} UTC)"
            ) from e
        if i0 < d.t_in - 1:
            raise InputContractError(f"need {d.t_in} frames ending at t0; only {i0 + 1} available")
        x, avail = ev.window(i0 - d.t_in + 1, i0 + 1)
        radius = lb.radius_px(d.lightning_radius_km, ev.grid_spacing_km)
        past = lb.past_lightning(ev.lightning_occurrence, i0, 30, d.step_minutes, radius)
        inp = ForecastInput(
            x, avail, list(self.artifact.channels), past, ev.grid_spacing_km, d.step_minutes
        )
        return ev, inp, t0_ts

    def _wrap(
        self,
        fc: ForecastArrays,
        ev: Event,
        inp: ForecastInput,
        t0,
        name,
        version,
        mode,
        ms,
        extra=None,
    ):
        d = self.config.data
        ds = build_forecast_dataset(
            fc,
            lat=ev.lat,
            lon=ev.lon,
            t0=t0,
            lead_minutes=d.lead_minutes,
            ltg_leads_min=list(d.lightning_leads_min),
            past_ltg=inp.past_ltg,
            model_name=name,
            model_version=version,
            mode=mode,
            missing_channels=[
                c for j, c in enumerate(self.artifact.channels) if not inp.avail[:, j].any()
            ],
            inference_ms=ms,
            first_flash_threshold=self.config.inference.first_flash_threshold,
            extra_attrs=extra,
        )
        problems = validate_forecast(ds, d.lead_minutes, list(d.lightning_leads_min))
        if problems:
            raise OutputContractError(f"forecast failed output validation: {problems}")
        return ds

    # ------------------------------------------------------------------ predict
    def predict(
        self,
        inputs: xr.Dataset,
        t0: datetime | str | pd.Timestamp,
        *,
        n_members: int = 0,
        seed: int | None = 0,
    ) -> xr.Dataset:
        """Nowcast from the ``t_in`` frames ending at ``t0``.

        Args:
            inputs: input-contract Dataset with at least ``t_in`` (7) frames up to ``t0``.
                Missing channels / masked regions are allowed; if radar is absent the
                forecast runs in ``satellite_only`` mode.
            t0: analysis time (UTC; tz-aware datetimes are converted). Must match a frame.
            n_members: ensemble members to add as ``reflectivity_members`` (needs a model
                trained with the diffusion refiner, see ``info.has_ensemble``).
            seed: seed for the members' initial noise (same seed -> same members; None = random).

        Returns:
            Dataset following the output contract (``inference/schema.py``).

        Raises:
            InputContractError: inputs or ``t0`` violate the input contract.
            NowcastError: ``n_members > 0`` but the model has no ensemble refiner.
        """
        if n_members > 0 and self.runner.refiner is None:
            raise NowcastError(
                f"n_members={n_members} requested but this model has no ensemble refiner "
                "(train one with configs/train/refiner.yaml)"
            )
        start = time.perf_counter()
        ev, inp, t0_ts = self._prepare(inputs, t0)
        mode = self.runner.mode_for(inp.avail, satellite_only=False)
        fc = self.runner.forecast(inp, n_members=n_members, seed=seed)
        ms = 1000 * (time.perf_counter() - start)
        return self._wrap(
            fc,
            ev,
            inp,
            t0_ts,
            self.artifact.name,
            self.artifact.version,
            mode,
            ms,
            {
                "calibrated": int(self.artifact.calibrator is not None),
                "backend": self._backend,
                "n_members": int(n_members),
            },
        )

    def predict_baseline(
        self,
        inputs: xr.Dataset,
        t0: datetime | str | pd.Timestamp,
        kind: BaselineKind = "extrapolation",
        *,
        n_members: int = 20,
        seed: int = 0,
    ) -> xr.Dataset:
        """Same output schema as :meth:`predict`, from a reference forecaster.

        ``kind`` is ``persistence``, ``extrapolation`` (pysteps Lucas-Kanade + semi-Lagrangian)
        or ``steps`` (pysteps STEPS with ``n_members`` members in ``reflectivity_members``).
        """
        if kind not in BASELINES:
            raise ValueError(f"kind must be one of {BASELINES}, got {kind!r}")
        start = time.perf_counter()
        ev, inp, t0_ts = self._prepare(inputs, t0)
        mode = self.runner.mode_for(inp.avail, satellite_only=False)
        d = self.config.data
        fc = make_baseline(kind, n_members=n_members, seed=seed).forecast(
            inp, d.t_out, tuple(d.lightning_leads_min)
        )
        fc.reflectivity = np.asarray(fc.reflectivity, dtype=np.float32)
        ms = 1000 * (time.perf_counter() - start)
        return self._wrap(fc, ev, inp, t0_ts, f"baseline_{kind}", "baseline", mode, ms)
