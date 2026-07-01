"""ViT transformer anchor (MONAI UNETR, S5)."""
from __future__ import annotations

from monai.networks.nets import UNETR
from torch import nn

from .base import IN_CHANNELS, OUT_CHANNELS


def build_unetr(cfg) -> nn.Module:
    return UNETR(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        img_size=tuple(cfg.train.patch_size),  # must be divisible by 16
        feature_size=16,
        norm_name="instance",
        spatial_dims=3,
    )
