"""Small conditional UNet used as the denoiser of the diffusion refiner."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of integer diffusion steps (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=1)


def _groups(c: int) -> int:
    for g in (8, 4, 2, 1):
        if c % g == 0:
            return g
    return 1


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(c_in), c_in)
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.temb = nn.Linear(t_dim, c_out)
        self.norm2 = nn.GroupNorm(_groups(c_out), c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    """Denoiser: predicts v for ``x_t`` (B, c_out, H, W) given conditioning (B, c_cond, H, W) and step t.

    H and W must be divisible by ``2 ** (len(channel_mults) - 1)``.
    """

    def __init__(self, c_out: int, c_cond: int, base: int = 32, channel_mults=(1, 2, 2)):
        super().__init__()
        self.t_dim = base * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base, self.t_dim), nn.SiLU(), nn.Linear(self.t_dim, self.t_dim)
        )
        self.base = base
        self.inp = nn.Conv2d(c_out + c_cond, base, 3, padding=1)
        chans = [base * m for m in channel_mults]
        self.down = nn.ModuleList()
        self.downsample = nn.ModuleList()
        c = base
        skips = []
        for i, ch in enumerate(chans):
            self.down.append(ResBlock(c, ch, self.t_dim))
            c = ch
            skips.append(c)
            last = i == len(chans) - 1
            self.downsample.append(
                nn.Identity() if last else nn.Conv2d(c, c, 3, stride=2, padding=1)
            )
        self.mid = ResBlock(c, c, self.t_dim)
        self.up = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i, ch in enumerate(reversed(chans)):
            self.up.append(ResBlock(c + skips[-1 - i], ch, self.t_dim))
            c = ch
            last = i == len(chans) - 1
            self.upsample.append(
                nn.Identity() if last else nn.Upsample(scale_factor=2, mode="nearest")
            )
        self.out = nn.Sequential(
            nn.GroupNorm(_groups(c), c), nn.SiLU(), nn.Conv2d(c, c_out, 3, padding=1)
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.factor = 2 ** (len(chans) - 1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        temb = self.time_mlp(timestep_embedding(t, self.base))
        h = self.inp(torch.cat([x_t, cond], dim=1))
        hs = []
        for block, ds in zip(self.down, self.downsample, strict=True):
            h = block(h, temb)
            hs.append(h)
            h = ds(h)
        h = self.mid(h, temb)
        for block, us in zip(self.up, self.upsample, strict=True):
            h = block(torch.cat([h, hs.pop()], dim=1), temb)
            h = us(h)
        return self.out(h)
