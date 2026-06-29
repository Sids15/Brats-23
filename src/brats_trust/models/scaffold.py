"""Single shared U-Net scaffold (roadmap S4).

'Matched stages/widths/skips/patch/optimizer/schedule/init/data/seeds; one pluggable
block.' Everything except the swapped block is held identical so the probes vary one
thing at a time:

  - Probe 3 (PRIMARY): receptive field via the conv ``block`` (``conv`` vs depthwise-
    separable ``dwsep``) and ``kernel_size`` -> the ERF<->faithfulness curve (S4.1).
  - Probe 1/2: encoder/decoder swaps reuse the same scaffold with a different block.

A compact, dependency-light 3D U-Net (not MONAI's) so the pluggable block and receptive
field are under our direct control for the RF sweep. Input 4 channels [FLAIR,T1,T1CE,T2],
output 3 overlapping regions [WT,TC,ET] (sigmoid head; DiceCE handles activation).
"""
from __future__ import annotations

import torch
from torch import nn

IN_CHANNELS = 4
OUT_CHANNELS = 3


def _norm_act(channels: int) -> nn.Sequential:
    # InstanceNorm is the standard choice for small-batch 3D medical segmentation.
    return nn.Sequential(nn.InstanceNorm3d(channels, affine=True), nn.LeakyReLU(0.01, inplace=True))


class ConvBlock(nn.Module):
    """Standard double 3D convolution (the default scaffold block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            _norm_act(out_ch),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            _norm_act(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable double conv: changes receptive field while keeping params/FLOPs
    comparable to :class:`ConvBlock` (roadmap S4 Probe 3 control)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2

        def ds(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv3d(cin, cin, kernel_size, padding=pad, groups=cin, bias=False),  # depthwise
                nn.Conv3d(cin, cout, 1, bias=False),                                    # pointwise
                *_norm_act(cout),
            )

        self.net = nn.Sequential(ds(in_ch, out_ch), ds(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


BLOCKS = {"conv": ConvBlock, "dwsep": DepthwiseSeparableBlock}


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
