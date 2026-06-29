"""SegMamba-style 3D segmentation network (roadmap S4/S5, the Mamba/state-space arm).

Modeled on SegMamba (Xing et al., 2024): a U-Net-shaped encoder whose stages mix spatial
context with a Mamba state-space layer (flatten voxels -> sequence -> Mamba -> reshape)
alongside convolution, with a convolutional decoder. Faithful to the *mechanism*; built
under our matched protocol (4-ch in, 3 overlapping regions out) so it's measured like every
other model.

Requires ``mamba-ssm`` (and ``causal-conv1d``), which need a CUDA build -> this runs on the
GPU machine only. The classes import cleanly without mamba-ssm; instantiation is guarded.
"""
from __future__ import annotations

import torch
from torch import nn

try:
    from mamba_ssm import Mamba
    _HAS_MAMBA = True
except ImportError:  # mamba-ssm not installed (e.g. CPU dev box)
    _HAS_MAMBA = False


def _norm_act(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.InstanceNorm3d(channels, affine=True), nn.LeakyReLU(0.01, inplace=True))


class MambaLayer(nn.Module):
    """Flatten a 3D feature map to a voxel sequence, mix with Mamba, reshape back (residual)."""

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, *spatial = x.shape
        tokens = x.flatten(2).transpose(1, 2)          # (B, N, C)
        tokens = tokens + self.mamba(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(b, c, *spatial)


class MambaStage(nn.Module):
    """Conv (changes width) + a Mamba spatial-mixing layer."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False), *_norm_act(out_ch))
        self.mamba = MambaLayer(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.conv(x))


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False), *_norm_act(out_ch),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False), *_norm_act(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SegMamba(nn.Module):
    """U-Net with Mamba-mixing encoder stages and a convolutional decoder."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3,
                 features: tuple[int, ...] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(MambaStage(prev, f))
            prev = f
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = MambaStage(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(prev, f, kernel_size=2, stride=2))
            self.decoders.append(_ConvBlock(f * 2, f))
            prev = f
        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for upconv, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


def build_segmamba(in_channels: int = 4, out_channels: int = 3,
                   features: tuple[int, ...] = (32, 64, 128, 256)) -> SegMamba:
    """Construct SegMamba; raises if mamba-ssm isn't installed (CPU dev box)."""
    if not _HAS_MAMBA:
        raise ImportError(
            "SegMamba needs the Mamba kernels: `pip install mamba-ssm causal-conv1d` "
            "(requires a CUDA build). Run it on the GPU machine."
        )
    return SegMamba(in_channels, out_channels, tuple(features))
