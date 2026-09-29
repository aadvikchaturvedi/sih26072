"""Isotonic calibration of lightning probabilities.

One monotone mapping per (mode, lead): ``full`` and ``satellite_only`` inputs
produce differently calibrated raw probabilities, so each gets its own curve.
Curves are fitted with scikit-learn and stored as plain threshold arrays
(applied with ``np.interp``), so ``calibrator.pkl`` does not depend on the
scikit-learn version at load time.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

CALIBRATOR_VERSION = 1
MODES = ("full", "satellite_only")


@dataclass
class IsotonicCurve:
    x: np.ndarray
    y: np.ndarray

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(p, 0, 1), self.x, self.y).astype(np.float32)


@dataclass
class IsotonicCalibrator:
    leads_min: list[int]
    curves: dict[str, list[IsotonicCurve]] = field(default_factory=dict)  # mode -> per-lead curve
    meta: dict = field(default_factory=dict)
    version: int = CALIBRATOR_VERSION

    @classmethod
    def fit(
        cls,
        probs: dict[str, np.ndarray],
        labels: dict[str, np.ndarray],
        leads_min: list[int],
        max_points: int = 2_000_000,
        seed: int = 0,
    ) -> IsotonicCalibrator:
        """``probs[mode]``/``labels[mode]`` are (N, n_leads) arrays of valid pixels."""
        from sklearn.isotonic import IsotonicRegression

        rng = np.random.default_rng(seed)
        cal = cls(list(leads_min), meta={"fitted_at": datetime.now(UTC).isoformat(), "n": {}})
        for mode, p in probs.items():
            y = labels[mode]
            curves = []
            for i in range(len(leads_min)):
                pi, yi = p[:, i], y[:, i]
                if len(pi) > max_points:
                    sel = rng.choice(len(pi), max_points, replace=False)
                    pi, yi = pi[sel], yi[sel]
                ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(pi, yi)
                curves.append(
                    IsotonicCurve(np.asarray(ir.X_thresholds_), np.asarray(ir.y_thresholds_))
                )
            cal.curves[mode] = curves
            cal.meta["n"][mode] = int(len(p))
        return cal

    def apply(self, probs: np.ndarray, mode: str = "full") -> np.ndarray:
        """(n_leads, H, W) raw probabilities -> calibrated."""
        curves = self.curves.get(mode) or self.curves.get("full")
        if curves is None:
            return probs
        if probs.shape[0] != len(curves):
            raise ValueError(f"calibrator has {len(curves)} leads, got {probs.shape[0]}")
        return np.stack([c(p) for c, p in zip(curves, probs, strict=True)])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> IsotonicCalibrator:
        with open(path, "rb") as f:
            obj = pickle.load(f)  # noqa: S301 - artifacts are produced by this package
        if (
            not isinstance(obj, IsotonicCalibrator)
            or getattr(obj, "version", None) != CALIBRATOR_VERSION
        ):
            raise ValueError(f"{path} is not a v{CALIBRATOR_VERSION} IsotonicCalibrator")
        return obj
