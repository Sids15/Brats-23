"""Single shared U-Net scaffold / Instrument (roadmap S4)."""
from __future__ import annotations

import torch
from torch import nn

from .base import IN_CHANNELS, OUT_CHANNELS, BLOCKS, _align_to


class UNet3D(nn.Module):
    """Symmetric 3D U-Net with a configurable conv block and width schedule."""

    def __init__(
        self,
        in_channels: int = IN_CHANNELS,
        out_channels: int = OUT_CHANNELS,
        features: tuple[int, ...] = (32, 64, 128, 256),
        block: str = "conv",
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if block not in BLOCKS:
            raise ValueError(f"unknown block {block!r}; choose from {sorted(BLOCKS)}")
        blk = BLOCKS[block]

        self.encoders = nn.ModuleList()
        prev = in_channels
        for f in features:
            self.encoders.append(blk(prev, f, kernel_size))
            prev = f
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = blk(features[-1], features[-1] * 2, kernel_size)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        prev = features[-1] * 2
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose3d(prev, f, kernel_size=2, stride=2))
            self.decoders.append(blk(f * 2, f, kernel_size))
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
            x = _align_to(x, skip)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


def build_scaffold(
    block: str = "conv",
    features: tuple[int, ...] = (32, 64, 128, 256),
    kernel_size: int = 3,
    in_channels: int = IN_CHANNELS,
    out_channels: int = OUT_CHANNELS,
) -> UNet3D:
    """Construct the shared scaffold with the chosen pluggable block (roadmap S4)."""
    return UNet3D(in_channels, out_channels, tuple(features), block, kernel_size)
