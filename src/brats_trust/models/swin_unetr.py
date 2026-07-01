"""Swin-transformer anchor (roadmap S5) — MONAI ``SwinUNETR``.

Hierarchical shifted-window transformer encoder with a convolutional decoder, used as an
architecture under the matched global protocol. Input must be divisible by 32 and >= 64^3
(our 96^3/128^3 patches qualify; the CPU test uses 64^3); ``feature_size`` divisible by 12.
"""
from __future__ import annotations

from monai.networks.nets import SwinUNETR
from torch import nn

from .base import IN_CHANNELS, OUT_CHANNELS


def build(cfg) -> nn.Module:
    """Build SwinUNETR (auto-sizes to divisible-by-32 inputs)."""
    return SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        feature_size=24,
    )
