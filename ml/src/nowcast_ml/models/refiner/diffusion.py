"""Diffusion residual refiner (M9).

The deterministic backbone forecast ``det`` (normalized MAX-Z, B x T_out x H x W)
is sharp at short leads and blurry later. The refiner learns the distribution of
the residual ``r = (y - det) / residual_scale`` with a DDPM (cosine schedule,
v-prediction, Salimans & Ho 2022), conditioned on ``det`` and the last input frame (data +
availability masks). Members are ``det + residual_scale * r_sample``, sampled with
deterministic DDIM from independent initial noise, so each seed gives a
reproducible member (cf. residual diffusion in CorrDiff, Mardani et al. 2023).

The backbone stays frozen while the refiner trains; the refiner is not exported to
TorchScript/ONNX and runs eagerly in :class:`~nowcast_ml.inference.Predictor`.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nowcast_ml.config import RefinerConfig
from nowcast_ml.models.refiner.unet import ConditionalUNet


def cosine_alpha_bar(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cumulative signal fraction alpha_bar_t for t = 0..T-1 (Nichol & Dhariwal 2021)."""
    steps = torch.arange(timesteps + 1, dtype=torch.float64)
    f = torch.cos(((steps / timesteps) + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class DiffusionRefiner(nn.Module):
    name = "diffusion"

    def __init__(self, t_out: int, cond_channels: int, cfg: RefinerConfig | None = None):
        super().__init__()
        self.cfg = cfg or RefinerConfig()
        self.t_out = t_out
        self.cond_channels = cond_channels
        self.net = ConditionalUNet(
            t_out, t_out + cond_channels, self.cfg.base_channels, tuple(self.cfg.channel_mults)
        )
        self.register_buffer("alpha_bar", cosine_alpha_bar(self.cfg.timesteps), persistent=False)

    @property
    def spatial_factor(self) -> int:
        return self.net.factor

    def conditioning(self, xin: torch.Tensor, det: torch.Tensor) -> torch.Tensor:
        """Model input (B, T_in, 2C, H, W) + deterministic forecast -> (B, T_out + 2C, H, W)."""
        return torch.cat([det, xin[:, -1]], dim=1)

    # ------------------------------------------------------------------ training
    def loss(
        self,
        xin: torch.Tensor,
        det: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Masked v-prediction MSE. ``target`` is normalized MAX-Z (NaN allowed where invalid).

        v-prediction (v = sqrt(ab) * eps - sqrt(1 - ab) * r) keeps the implied x0 estimate
        well-conditioned at high noise, where epsilon-prediction divides by sqrt(ab) ~ 0.
        """
        B = det.shape[0]
        r = (torch.nan_to_num(target, nan=0.0) - det) / self.cfg.residual_scale
        v = valid.to(det.dtype)
        r = r * v
        t = torch.randint(0, self.cfg.timesteps, (B,), device=det.device, generator=generator)
        noise = torch.randn(r.shape, device=det.device, generator=generator)
        ab = self.alpha_bar[t][:, None, None, None]
        x_t = ab.sqrt() * r + (1 - ab).sqrt() * noise
        v_target = ab.sqrt() * noise - (1 - ab).sqrt() * r
        v_pred = self.net(x_t, t, self.conditioning(xin, det))
        return (((v_pred - v_target) ** 2) * v).sum() / v.sum().clamp_min(1.0)

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        deterministic: torch.Tensor,
        n_members: int,
        seed: int | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        """(B, n_members, T_out, H, W) normalized members (see :class:`Refiner`)."""
        B, T, H, W = deterministic.shape
        steps = steps or self.cfg.sample_steps
        dev = deterministic.device
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        else:
            g.seed()
        det = deterministic.repeat_interleave(n_members, dim=0)
        c = self.conditioning(cond, deterministic).repeat_interleave(n_members, dim=0)
        x = torch.randn((B * n_members, T, H, W), generator=g).to(dev)
        ts = torch.linspace(self.cfg.timesteps - 1, 0, steps).round().long().tolist()
        for i, t in enumerate(ts):
            ab = self.alpha_bar[t]
            ab_prev = (
                self.alpha_bar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=dev)
            )
            tt = torch.full((B * n_members,), t, device=dev, dtype=torch.long)
            v = self.net(x, tt, c)
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * v
            eps = (1 - ab).sqrt() * x + ab.sqrt() * v
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps  # DDIM, eta = 0
        members = det + self.cfg.residual_scale * x
        return members.reshape(B, n_members, T, H, W)
