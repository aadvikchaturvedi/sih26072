"""SimVP v2 backbone (Gao et al. 2022; Tan et al. 2022, gSTA translator).

Encoder: per-frame strided ConvSC stack -> latent (B*T, hid_s, H/f, W/f)
Translator: frames stacked on channels (B, T*hid_s, H/f, W/f) -> N_t gSTA blocks
Decoder: per-frame PixelShuffle ConvSC stack with a skip from the first encoder layer.

Implemented from the paper/OpenSTL design with only static shapes and standard ops,
so it scripts with TorchScript and exports to ONNX (opset 17).
"""

from __future__ import annotations

import torch
from torch import nn


def sampling_generator(n: int, reverse: bool = False) -> list[bool]:
    samplings = ([False, True] * (n // 2 + 1))[:n]
    return list(reversed(samplings)) if reverse else samplings


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class ConvSC(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 3,
        downsampling: bool = False,
        upsampling: bool = False,
    ):
        super().__init__()
        pad = (k - 1) // 2
        if upsampling:
            self.conv = nn.Sequential(nn.Conv2d(c_in, c_out * 4, k, 1, pad), nn.PixelShuffle(2))
        else:
            self.conv = nn.Conv2d(c_in, c_out, k, 2 if downsampling else 1, pad)
        self.norm = nn.GroupNorm(2, c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, c_in: int, hid: int, n_s: int, k: int):
        super().__init__()
        s = sampling_generator(n_s)
        self.enc = nn.ModuleList(
            [ConvSC(c_in, hid, k, downsampling=s[0])]
            + [ConvSC(hid, hid, k, downsampling=d) for d in s[1:]]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for layer in self.enc[1:]:
            latent = layer(latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, hid: int, c_out: int, n_s: int, k: int):
        super().__init__()
        s = sampling_generator(n_s, reverse=True)
        self.dec = nn.ModuleList([ConvSC(hid, hid, k, upsampling=u) for u in s])
        self.readout = nn.Conv2d(hid, c_out, 1)

    def forward(self, hid, enc1):
        for layer in self.dec[:-1]:
            hid = layer(hid)
        y = self.dec[-1](hid + enc1)
        return self.readout(y)


# ------------------------------------------------------------------ gSTA translator
class AttentionModule(nn.Module):
    """Large-kernel attention decomposed into depthwise, dilated depthwise and 1x1 convs."""

    def __init__(self, dim: int, kernel_size: int = 21, dilation: int = 3):
        super().__init__()
        d_k = 2 * dilation - 1
        d_p = (d_k - 1) // 2
        dd_k = kernel_size // dilation + ((kernel_size // dilation) % 2 - 1)
        dd_p = dilation * (dd_k - 1) // 2
        self.conv0 = nn.Conv2d(dim, dim, d_k, padding=d_p, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, dd_k, padding=dd_p, groups=dim, dilation=dilation)
        self.conv1 = nn.Conv2d(dim, 2 * dim, 1)
        self.dim = dim

    def forward(self, x):
        attn = self.conv1(self.conv_spatial(self.conv0(x)))
        f, g = torch.split(attn, self.dim, dim=1)
        return torch.sigmoid(g) * f


class SpatialAttention(nn.Module):
    def __init__(self, dim: int, kernel_size: int = 21):
        super().__init__()
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.act = nn.GELU()
        self.gate = AttentionModule(dim, kernel_size)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        return self.proj_2(self.gate(self.act(self.proj_1(x)))) + x


class MixMlp(nn.Module):
    def __init__(self, dim: int, hidden: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.dw(self.fc1(x))))))


class GASubBlock(nn.Module):
    def __init__(
        self, dim, kernel_size=21, mlp_ratio=4.0, drop=0.0, drop_path=0.0, init_value=1e-2
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.attn = SpatialAttention(dim, kernel_size)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mlp = MixMlp(dim, int(dim * mlp_ratio), drop)
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones(dim))
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1[None, :, None, None] * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2[None, :, None, None] * self.mlp(self.norm2(x)))
        return x


class MetaBlock(nn.Module):
    def __init__(self, c_in, c_out, mlp_ratio=4.0, drop=0.0, drop_path=0.0):
        super().__init__()
        self.block = GASubBlock(c_in, 21, mlp_ratio, drop, drop_path)
        self.reduction = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.reduction(self.block(x))


class MidMetaNet(nn.Module):
    def __init__(self, c_in, c_hid, n_t, mlp_ratio=4.0, drop=0.0, drop_path=0.1):
        super().__init__()
        assert n_t >= 2, "n_t must be >= 2"
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, n_t)]
        layers = [MetaBlock(c_in, c_hid, mlp_ratio, drop, dpr[0])]
        layers += [MetaBlock(c_hid, c_hid, mlp_ratio, drop, dpr[i]) for i in range(1, n_t - 1)]
        layers += [MetaBlock(c_hid, c_in, mlp_ratio, drop, dpr[-1])]
        self.enc = nn.Sequential(*layers)

    def forward(self, x):
        return self.enc(x)


class SimVPBackbone(nn.Module):
    """(B, T, C_in, H, W) -> decoder features (B, T, hid_s, H, W) and latent (B, T*hid_s, H/f, W/f)."""

    def __init__(
        self,
        in_channels: int,
        t_in: int,
        hid_s: int = 64,
        hid_t: int = 256,
        n_s: int = 4,
        n_t: int = 8,
        spatio_kernel_enc: int = 3,
        spatio_kernel_dec: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.t_in = t_in
        self.hid_s = hid_s
        self.spatial_factor = 2 ** sum(sampling_generator(n_s))
        self.enc = Encoder(in_channels, hid_s, n_s, spatio_kernel_enc)
        self.hid = MidMetaNet(t_in * hid_s, hid_t, n_t, mlp_ratio, drop, drop_path)
        self.dec = Decoder(hid_s, hid_s, n_s, spatio_kernel_dec)
        self.out_channels = hid_s
        self.latent_channels = t_in * hid_s

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C, H, W = x.shape
        embed, skip = self.enc(x.reshape(B * T, C, H, W))
        _, C_, H_, W_ = embed.shape
        z = self.hid(embed.reshape(B, T * C_, H_, W_))
        dec = self.dec(z.reshape(B * T, C_, H_, W_), skip)
        return dec.reshape(B, T, self.hid_s, H, W), z
