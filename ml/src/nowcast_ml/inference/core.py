"""Model execution shared by :class:`Predictor` and evaluation.

``ModelRunner`` turns a :class:`~nowcast_ml.baselines.base.ForecastInput` into
:class:`~nowcast_ml.inference.schema.ForecastArrays`: normalize -> mask -> pad ->
network -> crop -> denormalize -> sigmoid -> calibrate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from nowcast_ml.baselines.base import ForecastInput
from nowcast_ml.data import channels as ch
from nowcast_ml.data.transforms import NormStats, drop_radar, model_input, pad_to_multiple
from nowcast_ml.inference.schema import ForecastArrays


@dataclass
class ModelRunner:
    model: torch.nn.Module  # eager NowcastModel or a TorchScript module with the same signature
    norm_stats: NormStats
    channels: list[str]
    device: torch.device
    spatial_factor: int = 4
    calibrator: object | None = None  # IsotonicCalibrator: .apply(probs (n_leads, H, W), mode)
    name: str = "model"
    min_radar_coverage: float = 0.05

    refiner: torch.nn.Module | None = None  # DiffusionRefiner (eager) for ensemble members

    def _pad_factor(self) -> int:
        f = self.spatial_factor
        if self.refiner is not None:
            f = math.lcm(f, int(self.refiner.spatial_factor))
        return f

    def _run(self, x: np.ndarray, avail: np.ndarray, satellite_only: bool):
        """-> padded model input, padded normalized reflectivity, padded logits, original (H, W)."""
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        at = torch.from_numpy(np.ascontiguousarray(avail, dtype=bool))
        if satellite_only:
            at = drop_radar(at, self.channels)
        xin = model_input(self.norm_stats.normalize(xt), at)
        H, W = xin.shape[-2:]
        xin = pad_to_multiple(xin, self._pad_factor()).to(self.device)
        with torch.inference_mode():
            refl, ltg = self.model(xin)
        return xin, refl, ltg, (H, W)

    def _to_dbz(self, refl_norm: torch.Tensor) -> np.ndarray:
        dbz = self.norm_stats.denormalize_channel(ch.TARGET_CHANNEL, refl_norm.float().cpu())
        return dbz.clamp(0.0, 80.0).numpy()

    def forecast_batch(self, x: np.ndarray, avail: np.ndarray, satellite_only: bool = False):
        """(B, T, C, H, W) physical + avail -> (refl dBZ (B, T_out, H, W), raw probs (B, n_ltg, H, W))."""
        _, refl, ltg, (H, W) = self._run(x, avail, satellite_only)
        dbz = self._to_dbz(refl[..., :H, :W])
        return dbz, torch.sigmoid(ltg[..., :H, :W].float().cpu()).numpy()

    def forecast_members(
        self,
        x: np.ndarray,
        avail: np.ndarray,
        n_members: int,
        seed: int | None = 0,
        satellite_only: bool = False,
    ):
        """Deterministic forecast plus ``n_members`` refiner members.

        Returns (dbz (B, T_out, H, W), raw probs (B, n_ltg, H, W), members dBZ (B, M, T_out, H, W)).
        """
        if self.refiner is None:
            raise ValueError("this model has no ensemble refiner")
        xin, refl, ltg, (H, W) = self._run(x, avail, satellite_only)
        with torch.inference_mode():
            members = self.refiner.sample(xin, refl.float(), n_members, seed=seed)
        return (
            self._to_dbz(refl[..., :H, :W]),
            torch.sigmoid(ltg[..., :H, :W].float().cpu()).numpy(),
            self._to_dbz(members[..., :H, :W]),
        )

    def calibrate(self, probs: np.ndarray, mode: str) -> np.ndarray:
        return probs if self.calibrator is None else self.calibrator.apply(probs, mode)

    def mode_for(self, avail: np.ndarray, satellite_only: bool) -> str:
        if satellite_only:
            return "satellite_only"
        return detect_mode(avail, self.channels, self.min_radar_coverage)

    def forecast(
        self,
        inp: ForecastInput,
        satellite_only: bool = False,
        n_members: int = 0,
        seed: int | None = 0,
    ) -> ForecastArrays:
        mode = self.mode_for(inp.avail, satellite_only)
        sat = mode == "satellite_only"
        if n_members > 0:
            dbz, p, members = self.forecast_members(
                inp.x[None], inp.avail[None], n_members, seed, sat
            )
            return ForecastArrays(dbz[0], self.calibrate(p[0], mode), members[0])
        dbz, p = self.forecast_batch(inp.x[None], inp.avail[None], sat)
        return ForecastArrays(reflectivity=dbz[0], lightning=self.calibrate(p[0], mode))


class ModelForecaster:
    """Adapter so the model is scored through the same Forecaster protocol as baselines."""

    def __init__(
        self,
        runner: ModelRunner,
        satellite_only: bool = False,
        name: str | None = None,
        n_members: int = 0,
    ):
        self.runner = runner
        self.satellite_only = satellite_only
        self.n_members = n_members
        default = "model_satellite_only" if satellite_only else "model_full"
        self.name = name or (default if n_members == 0 else "model_ensemble")

    def forecast(
        self, inp: ForecastInput, n_leads: int = 12, ltg_leads_min=(30, 60)
    ) -> ForecastArrays:
        return self.runner.forecast(
            inp, satellite_only=self.satellite_only, n_members=self.n_members
        )


def detect_mode(avail: np.ndarray, channels: list[str], min_radar_coverage: float = 0.05) -> str:
    """``full`` if radar covers >= ``min_radar_coverage`` of the domain in the last input frame,
    otherwise ``satellite_only``. ``avail`` is (T, C, H, W)."""
    from nowcast_ml.data.event import radar_coverage

    return "full" if radar_coverage(avail, channels) >= min_radar_coverage else "satellite_only"
