"""Normalization, modality dropout and spatial augmentation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from nowcast_ml.data import channels as ch

NORM_STATS_VERSION = 1
#: Channels normalized after log1p (heavy-tailed, non-negative).
LOG1P_CHANNELS = frozenset({"flash_density"})


class NormStatsMismatchError(ValueError):
    """Normalization statistics do not match the requested channel list."""


@dataclass
class NormStats:
    channels: list[str]
    mean: list[float]
    std: list[float]
    log1p: list[bool] = field(default_factory=list)

    def __post_init__(self):
        if not self.log1p:
            self.log1p = [c in LOG1P_CHANNELS for c in self.channels]
        n = len(self.channels)
        if not (len(self.mean) == len(self.std) == len(self.log1p) == n):
            raise NormStatsMismatchError("norm stats arrays have inconsistent lengths")
        if any(s <= 0 for s in self.std):
            raise NormStatsMismatchError("norm stats std must be > 0")

    # --------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {
            "version": NORM_STATS_VERSION,
            "channels": self.channels,
            "mean": self.mean,
            "std": self.std,
            "log1p": self.log1p,
        }

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> NormStats:
        if d.get("version") != NORM_STATS_VERSION:
            raise NormStatsMismatchError(
                f"norm stats version {d.get('version')} != {NORM_STATS_VERSION}"
            )
        return cls(d["channels"], d["mean"], d["std"], d["log1p"])

    @classmethod
    def load(cls, path: str | Path) -> NormStats:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def check_channels(self, channels: list[str]) -> None:
        if list(channels) != list(self.channels):
            raise NormStatsMismatchError(
                f"normalization stats are for channels {self.channels}, model/config uses {list(channels)}"
            )

    # --------------------------------------------------------------- math
    def _arrays(self, like):
        if isinstance(like, torch.Tensor):
            kw = {"dtype": like.dtype, "device": like.device}
            return (
                torch.tensor(self.mean, **kw),
                torch.tensor(self.std, **kw),
                torch.tensor(self.log1p, device=like.device),
            )
        return (
            np.asarray(self.mean, np.float32),
            np.asarray(self.std, np.float32),
            np.asarray(self.log1p, bool),
        )

    def normalize(self, x, channel_axis: int = -3):
        """Normalize (..., C, H, W) data (numpy or torch). NaN stays NaN."""
        mean, std, lg = self._arrays(x)
        shape = [1] * x.ndim
        shape[channel_axis] = -1
        mean, std, lg = mean.reshape(shape), std.reshape(shape), lg.reshape(shape)
        if isinstance(x, torch.Tensor):
            v = torch.where(lg, torch.log1p(torch.clamp(x, min=0)), x)
        else:
            with np.errstate(invalid="ignore"):
                v = np.where(lg, np.log1p(np.clip(x, 0, None)), x)
        return (v - mean) / std

    def channel_index(self, name: str) -> int:
        return self.channels.index(name)

    def normalize_channel(self, name: str, v):
        i = self.channel_index(name)
        out = (
            v
            if not self.log1p[i]
            else (torch.log1p(v) if isinstance(v, torch.Tensor) else np.log1p(v))
        )
        return (out - self.mean[i]) / self.std[i]

    def denormalize_channel(self, name: str, v):
        i = self.channel_index(name)
        out = v * self.std[i] + self.mean[i]
        if self.log1p[i]:
            out = torch.expm1(out) if isinstance(out, torch.Tensor) else np.expm1(out)
        return out


def fit_norm_stats(arrays, channels: list[str], min_std: float = 1e-3) -> NormStats:
    """Fit per-channel mean/std over finite values of (…, C, H, W) arrays (NaN = unavailable).

    Uses a numerically stable parallel (Chan) update so it can stream many windows.
    Channels never observed get mean = registry background and std = 1.
    """
    C = len(channels)
    lg = np.array([c in LOG1P_CHANNELS for c in channels])
    n = np.zeros(C)
    mean = np.zeros(C)
    m2 = np.zeros(C)
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        a = np.moveaxis(a, -3, 0).reshape(C, -1)
        for i in range(C):
            v = a[i][np.isfinite(a[i])]
            if lg[i]:
                v = np.log1p(np.clip(v, 0, None))
            if v.size == 0:
                continue
            nb, mb = v.size, v.mean()
            m2b = ((v - mb) ** 2).sum()
            delta = mb - mean[i]
            tot = n[i] + nb
            mean[i] += delta * nb / tot
            m2[i] += m2b + delta**2 * n[i] * nb / tot
            n[i] = tot
    std = np.where(n > 1, np.sqrt(m2 / np.maximum(n - 1, 1)), 1.0)
    std = np.maximum(std, min_std)
    for i, c in enumerate(channels):
        if n[i] == 0:
            bg = ch.get(c).background
            mean[i] = np.log1p(bg) if lg[i] else bg
            std[i] = 1.0
    return NormStats(list(channels), mean.tolist(), std.tolist(), lg.tolist())


def model_input(x_norm: torch.Tensor, avail: torch.Tensor) -> torch.Tensor:
    """(B, T, C, H, W) normalized data + availability -> (B, T, 2C, H, W) model input.

    Unavailable values become 0 *after* normalization (i.e. the channel mean) and the
    availability mask is appended so the network can tell "missing" from "average".
    """
    a = avail.to(x_norm.dtype)
    x = torch.nan_to_num(x_norm, nan=0.0) * a
    return torch.cat([x, a], dim=2)


class ModalityDropout:
    """Randomly hide the radar group (per sample) and other channels (per sample-channel).

    Operates on ``avail`` (B, T, C, H, W) bool and returns a new mask; data are untouched
    because :func:`model_input` zeroes anything unavailable.
    """

    def __init__(self, channels: list[str], p_radar: float = 0.3, p_channel: float = 0.0):
        self.p_radar = p_radar
        self.p_channel = p_channel
        self.radar_idx = [i for i, c in enumerate(channels) if ch.group_of(c) == "radar"]
        self.other_idx = [i for i, c in enumerate(channels) if ch.group_of(c) != "radar"]

    def __call__(
        self, avail: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        B = avail.shape[0]
        C = avail.shape[2]
        keep = torch.ones(B, C, dtype=torch.bool, device=avail.device)
        if self.p_radar > 0 and self.radar_idx:
            drop = torch.rand(B, generator=generator).to(avail.device) < self.p_radar
            keep[:, self.radar_idx] &= ~drop[:, None]
        if self.p_channel > 0 and self.other_idx:
            drop = (
                torch.rand(B, len(self.other_idx), generator=generator).to(avail.device)
                < self.p_channel
            )
            keep[:, self.other_idx] &= ~drop
        return avail & keep[:, None, :, None, None]


def drop_radar(avail: torch.Tensor, channels: list[str]) -> torch.Tensor:
    """Deterministically hide all radar channels (satellite-only mode)."""
    idx = [i for i, c in enumerate(channels) if ch.group_of(c) == "radar"]
    out = avail.clone()
    out[:, :, idx] = False
    return out


# ------------------------------------------------------------------ spatial aug (numpy)
SPATIAL_KEYS = ("x", "avail", "y_refl", "y_refl_valid", "y_ltg", "y_ltg_valid", "past_ltg")


def random_crop(sample: dict, size: int, rng: np.random.Generator) -> dict:
    H, W = sample["x"].shape[-2:]
    if size > min(H, W):
        raise ValueError(f"crop {size} larger than domain {H}x{W}")
    y0 = int(rng.integers(0, H - size + 1))
    x0 = int(rng.integers(0, W - size + 1))
    out = dict(sample)
    for k in SPATIAL_KEYS:
        if k in out:
            out[k] = out[k][..., y0 : y0 + size, x0 : x0 + size]
    return out


def random_flip(sample: dict, rng: np.random.Generator) -> dict:
    out = dict(sample)
    for axis in (-1, -2):
        if rng.random() < 0.5:
            for k in SPATIAL_KEYS:
                if k in out:
                    out[k] = np.flip(out[k], axis=axis)
    for k in SPATIAL_KEYS:
        if k in out:
            out[k] = np.ascontiguousarray(out[k])
    return out


def pad_to_multiple(xin: torch.Tensor, factor: int) -> torch.Tensor:
    """Replicate-pad the last two dims of (B, T, C, H, W) up to a multiple of ``factor``."""
    B, T, C, H, W = xin.shape
    ph, pw = (-H) % factor, (-W) % factor
    if not (ph or pw):
        return xin
    return F.pad(xin.reshape(B, T * C, H, W), (0, pw, 0, ph), mode="replicate").reshape(
        B, T, C, H + ph, W + pw
    )
