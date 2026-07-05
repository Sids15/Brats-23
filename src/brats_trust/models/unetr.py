"""Transformer anchor (roadmap S5) — MONAI ``UNETR``.

A ViT encoder with a convolutional decoder, used as an architecture under the matched
global protocol. ``img_size`` must be divisible by 16 (our 96^3/128^3 patches qualify).
"""
from __future__ import annotations

from monai.networks.nets import UNETR
from torch import nn

from .base import IN_CHANNELS, OUT_CHANNELS


def build(cfg) -> nn.Module:
    """Build UNETR sized to the training patch (``cfg.train.patch_size``, divisible by 16)."""
    return UNETR(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        img_size=tuple(cfg.train.patch_size),
        feature_size=16,
        norm_name="instance",
        spatial_dims=3,
    )
