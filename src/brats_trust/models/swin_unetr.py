"""Swin transformer anchor (MONAI SwinUNETR, S5)."""
from __future__ import annotations

from monai.networks.nets import SwinUNETR
from torch import nn

from .base import IN_CHANNELS, OUT_CHANNELS


def build_swin_unetr(cfg) -> nn.Module:
    # SwinUNETR auto-handles input size if divisible by 32; feature_size divisible by 12.
    return SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        feature_size=24,
    )
