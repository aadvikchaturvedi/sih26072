"""Task heads on top of the shared backbone."""

from __future__ import annotations

import torch
from torch import nn


class ReflectivityHead(nn.Module):
    """Decoder features of all input frames (B, T_in, F, H, W) -> T_out normalized MAX-Z frames."""

    def __init__(self, t_in: int, feat: int, t_out: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(32, 2 * feat)
        self.net = nn.Sequential(
            nn.Conv2d(t_in * feat, hidden, 3, padding=1),
            nn.GroupNorm(2, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, t_out, 1),
        )

    def forward(self, dec: torch.Tensor) -> torch.Tensor:
        B, T, F, H, W = dec.shape
        return self.net(dec.reshape(B, T * F, H, W))


class LightningHead(nn.Module):
    """Shared-encoder latent (B, L, H/f, W/f) + last input frame (B, C_in, H, W) -> logits (B, n_leads, H, W).

    The latent carries storm dynamics; the last input frame (incl. current flash density
    and availability masks) is concatenated at full resolution after upsampling.
    """

    def __init__(
        self,
        latent_channels: int,
        in_channels: int,
        n_leads: int,
        spatial_factor: int,
        hidden: int = 32,
    ):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(latent_channels, hidden, 3, padding=1),
            nn.GroupNorm(2, hidden),
            nn.SiLU(inplace=True),
        )
        ups = []
        f = spatial_factor
        while f > 1:
            ups += [
                nn.Conv2d(hidden, hidden * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.SiLU(inplace=True),
            ]
            f //= 2
        self.up = nn.Sequential(*ups)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden + in_channels, hidden, 3, padding=1),
            nn.GroupNorm(2, hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, n_leads, 1),
        )
        # Rare-event prior: start at p ~= 1% so early training is stable (Lin et al. 2017).
        nn.init.constant_(self.fuse[-1].bias, -4.6)

    def forward(self, latent: torch.Tensor, last_frame: torch.Tensor) -> torch.Tensor:
        h = self.up(self.reduce(latent))
        return self.fuse(torch.cat([h, last_frame], dim=1))
