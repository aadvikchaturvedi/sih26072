"""Combined nowcast model: backbone + reflectivity head + lightning head.

Input is the normalized data concatenated with the availability mask
(:func:`nowcast_ml.data.transforms.model_input`): (B, T_in, 2C, H, W). Output is a
tuple ``(refl_norm (B, T_out, H, W), ltg_logits (B, n_ltg, H, W))``. Normalization,
calibration and the output schema live outside the network so the exported
TorchScript/ONNX graph is just this module.
"""

from __future__ import annotations

import torch
from torch import nn

from nowcast_ml.config import ModelConfig
from nowcast_ml.models.backbones import build_backbone
from nowcast_ml.models.heads import LightningHead, ReflectivityHead

DISABLED_LOGIT = -20.0


class NowcastModel(nn.Module):
    def __init__(
        self, cfg: ModelConfig, n_channels: int, t_in: int = 7, t_out: int = 12, n_ltg: int = 2
    ):
        super().__init__()
        self.n_channels = n_channels
        self.in_channels = 2 * n_channels  # data + availability mask
        self.t_in, self.t_out, self.n_ltg = t_in, t_out, n_ltg
        self.backbone = build_backbone(cfg, self.in_channels, t_in)
        self.spatial_factor = int(self.backbone.spatial_factor)
        self.refl_head = ReflectivityHead(t_in, self.backbone.out_channels, t_out)
        self.ltg_enabled = bool(cfg.lightning_head.enabled)
        self.ltg_head = (
            LightningHead(
                self.backbone.latent_channels,
                self.in_channels,
                n_ltg,
                self.spatial_factor,
                cfg.lightning_head.hidden,
            )
            if self.ltg_enabled
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dec, latent = self.backbone(x)
        refl = self.refl_head(dec)
        if self.ltg_enabled:
            ltg = self.ltg_head(latent, x[:, -1])
        else:
            B, _, _, H, W = x.shape
            ltg = torch.full((B, self.n_ltg, H, W), DISABLED_LOGIT, dtype=x.dtype, device=x.device)
        return refl, ltg

    # parameter groups used by the training stages
    def backbone_parameters(self):
        return list(self.backbone.parameters()) + list(self.refl_head.parameters())

    def lightning_head_parameters(self):
        return list(self.ltg_head.parameters())
