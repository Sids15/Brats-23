"""nnU-Net CNN anchor (MONAI DynUNet, S5)."""
from __future__ import annotations

from monai.networks.nets import DynUNet
from torch import nn

from .base import IN_CHANNELS, OUT_CHANNELS


def build_dynunet(cfg) -> nn.Module:
    # nnU-Net topology: 4 resolution levels (downsample factor 8). Input dims must be
    # divisible by 8 (our 128^3 patch is fine; 32^3 for CPU smoke is fine).
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
