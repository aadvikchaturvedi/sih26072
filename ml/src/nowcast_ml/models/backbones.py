"""Backbone registry: ``model.backbone`` in the config selects the class.

A backbone maps (B, T, C_in, H, W) to ``(dec, latent)`` where ``dec`` is
(B, T, out_channels, H, W) and ``latent`` is (B, latent_channels, H/f, W/f), and
exposes ``out_channels``, ``latent_channels`` and ``spatial_factor``.
"""

from __future__ import annotations

from torch import nn

from nowcast_ml.config import ModelConfig
from nowcast_ml.models.simvp import SimVPBackbone


def _simvp(cfg: ModelConfig, in_channels: int, t_in: int) -> nn.Module:
    return SimVPBackbone(
        in_channels=in_channels,
        t_in=t_in,
        hid_s=cfg.hid_s,
        hid_t=cfg.hid_t,
        n_s=cfg.n_s,
        n_t=cfg.n_t,
        spatio_kernel_enc=cfg.spatio_kernel_enc,
        spatio_kernel_dec=cfg.spatio_kernel_dec,
        mlp_ratio=cfg.mlp_ratio,
        drop=cfg.drop,
        drop_path=cfg.drop_path,
    )


BACKBONES = {"simvp": _simvp}


def build_backbone(cfg: ModelConfig, in_channels: int, t_in: int) -> nn.Module:
    try:
        factory = BACKBONES[cfg.backbone]
    except KeyError as e:
        raise ValueError(
            f"unknown backbone {cfg.backbone!r}; available: {sorted(BACKBONES)}"
        ) from e
    return factory(cfg, in_channels, t_in)
