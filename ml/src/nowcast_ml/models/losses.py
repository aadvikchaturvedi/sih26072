"""Losses: intensity-weighted MSE for reflectivity, binary focal loss for lightning."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IntensityWeightedMSE(nn.Module):
    """Masked MSE with step weights on the *target* intensity.

    ``w = 1`` below the first threshold, ``weights[i]`` at or above ``thresholds[i]``
    (default: x2 at >= 20 dBZ, x5 at >= 35 dBZ), so convective cores dominate the loss.
    The weighted squared error is averaged over valid pixels.
    """

    def __init__(self, thresholds_dbz=(20.0, 35.0), weights=(2.0, 5.0)):
        super().__init__()
        self.thresholds = list(thresholds_dbz)
        self.weights = list(weights)

    def weight_map(self, target_dbz: torch.Tensor) -> torch.Tensor:
        w = torch.ones_like(target_dbz)
        t = torch.nan_to_num(target_dbz, nan=-100.0)
        for thr, wt in zip(self.thresholds, self.weights, strict=True):
            w = torch.where(t >= thr, torch.full_like(w, wt), w)
        return w

    def forward(self, pred, target, target_dbz, valid) -> torch.Tensor:
        v = valid.to(pred.dtype)
        se = (pred - torch.nan_to_num(target, nan=0.0)) ** 2
        return (self.weight_map(target_dbz) * se * v).sum() / v.sum().clamp_min(1.0)


class BinaryFocalLoss(nn.Module):
    """Sigmoid focal loss (Lin et al. 2017) on logits, averaged over valid pixels."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target, valid=None) -> torch.Tensor:
        target = target.to(logits.dtype)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * target + (1 - p) * (1 - target)
        loss = ce * (1 - p_t) ** self.gamma
        if self.alpha >= 0:
            loss = (self.alpha * target + (1 - self.alpha) * (1 - target)) * loss
        if valid is None:
            return loss.mean()
        v = valid.to(logits.dtype)
        return (loss * v).sum() / v.sum().clamp_min(1.0)
