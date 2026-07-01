"""Shared model primitives (roadmap S4).

Channel contract and low-level building blocks reused across the architectures so every
model in the matched protocol reads the same 4-channel input [FLAIR, T1, T1CE, T2] and
emits the same 3 overlapping regions [WT, TC, ET]. Kept in one place so the per-model
modules stay small and consistent.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

IN_CHANNELS = 4    # [FLAIR, T1, T1CE, T2] (frozen order; see constants.CHANNEL_ORDER)
OUT_CHANNELS = 3   # overlapping regions [WT, TC, ET] (sigmoid head; DiceCE activates)


def align_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Pad/crop x's spatial dims to match ref's, so skip-concat works for any input size.

    Pooling floor-divides odd dimensions while transposed conv doubles them, so an
    arbitrary-sized (e.g. brain-cropped) volume can produce off-by-one skip mismatches.
    """
    diffs = [r - s for s, r in zip(x.shape[2:], ref.shape[2:])]
    if any(d != 0 for d in diffs):
        pad: list[int] = []
        for d in reversed(diffs):
            pad += [0, max(d, 0)]
        x = F.pad(x, pad)
        x = x[:, :, : ref.shape[2], : ref.shape[3], : ref.shape[4]]
    return x


def norm_act(channels: int) -> nn.Sequential:
    # InstanceNorm is the standard choice for small-batch 3D medical segmentation.
    return nn.Sequential(nn.InstanceNorm3d(channels, affine=True), nn.LeakyReLU(0.01, inplace=True))


class ConvBlock(nn.Module):
    """Standard double 3D convolution (the default scaffold block)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size, padding=pad, bias=False),
            norm_act(out_ch),
            nn.Conv3d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            norm_act(out_ch),
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
                *norm_act(cout),
            )

        self.net = nn.Sequential(ds(in_ch, out_ch), ds(out_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


BLOCKS = {"conv": ConvBlock, "dwsep": DepthwiseSeparableBlock}
