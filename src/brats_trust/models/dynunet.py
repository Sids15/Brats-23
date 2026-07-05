"""nnU-Net architecture anchor (roadmap S5) — MONAI ``DynUNet``.

The CNN anchor: nnU-Net's proven topology, used here as an *architecture* under our matched
global protocol (not via nnU-Net's own training framework), so it is measured like every
other model.
"""
from __future__ import annotations

from monai.networks.nets import DynUNet
from torch import nn

from .base import IN_CHANNELS, OUT_CHANNELS


def build(cfg) -> nn.Module:
    """Build DynUNet. 4 resolution levels (downsample factor 8); input dims divisible by 8
    (our 96^3/128^3 patches and the 32^3 CPU smoke all satisfy this)."""
    return DynUNet(
        spatial_dims=3,
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=[3, 3, 3, 3],
        strides=[1, 2, 2, 2],
        upsample_kernel_size=[2, 2, 2],
        norm_name="instance",
        res_block=True,
    )
