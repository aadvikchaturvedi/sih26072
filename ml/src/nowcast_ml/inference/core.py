"""Model execution shared by :class:`Predictor` and evaluation.

``ModelRunner`` turns a :class:`~nowcast_ml.baselines.base.ForecastInput` into
:class:`~nowcast_ml.inference.schema.ForecastArrays`: normalize -> mask -> pad ->
network -> crop -> denormalize -> sigmoid -> calibrate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from nowcast_ml.baselines.base import ForecastInput
from nowcast_ml.data import channels as ch
from nowcast_ml.data.transforms import NormStats, drop_radar, model_input
from nowcast_ml.inference.schema import ForecastArrays


@dataclass
class ModelRunner:
    model: torch.nn.Module  # eager NowcastModel or a TorchScript module with the same signature
    norm_stats: NormStats
    channels: list[str]
    device: torch.device
    spatial_factor: int = 4
    calibrator: object | None = None  # has .apply(probs (n_leads, H, W)) -> probs
    name: str = "model"

    def forecast_batch(self, x: np.ndarray, avail: np.ndarray, satellite_only: bool = False):
        """(B, T, C, H, W) physical + avail -> (refl dBZ (B, T_out, H, W), raw probs (B, n_ltg, H, W))."""
        xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        at = torch.from_numpy(np.ascontiguousarray(avail, dtype=bool))
        if satellite_only:
            at = drop_radar(at, self.channels)
        xin = model_input(self.norm_stats.normalize(xt), at)
        H, W = xin.shape[-2:]
        f = self.spatial_factor
        ph, pw = (-H) % f, (-W) % f
        if ph or pw:
            B, T, C = xin.shape[:3]
            xin = F.pad(xin.reshape(B, T * C, H, W), (0, pw, 0, ph), mode="replicate").reshape(
                B, T, C, H + ph, W + pw
            )
        with torch.inference_mode():
            refl, ltg = self.model(xin.to(self.device))
        refl = refl[..., :H, :W].float().cpu()
        ltg = ltg[..., :H, :W].float().cpu()
        dbz = self.norm_stats.denormalize_channel(ch.TARGET_CHANNEL, refl).clamp(0.0, 80.0).numpy()
        return dbz, torch.sigmoid(ltg).numpy()

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        return probs if self.calibrator is None else self.calibrator.apply(probs)

    def forecast(self, inp: ForecastInput, satellite_only: bool = False) -> ForecastArrays:
        dbz, p = self.forecast_batch(inp.x[None], inp.avail[None], satellite_only)
        return ForecastArrays(reflectivity=dbz[0], lightning=self.calibrate(p[0]))


class ModelForecaster:
    """Adapter so the model is scored through the same Forecaster protocol as baselines."""

    def __init__(self, runner: ModelRunner, satellite_only: bool = False, name: str | None = None):
        self.runner = runner
        self.satellite_only = satellite_only
        self.name = name or ("model_satellite_only" if satellite_only else "model_full")

    def forecast(
        self, inp: ForecastInput, n_leads: int = 12, ltg_leads_min=(30, 60)
    ) -> ForecastArrays:
        return self.runner.forecast(inp, satellite_only=self.satellite_only)


def detect_mode(avail: np.ndarray, channels: list[str], min_radar_coverage: float = 0.05) -> str:
    """``full`` if radar covers >= ``min_radar_coverage`` of the domain in the last input frame,
    otherwise ``satellite_only``. ``avail`` is (T, C, H, W)."""
    from nowcast_ml.data.event import radar_coverage

    return "full" if radar_coverage(avail, channels) >= min_radar_coverage else "satellite_only"
