"""Model factory (roadmap S4 Probe 1 mechanism swap + S5 Tier-A anchors).

One entry point, :func:`build_model`, builds any architecture from ``cfg.model.name`` so the
shared training/measurement pipeline runs each under the *matched global protocol* (S9) --
the off-the-shelf nets are used as architectures, not via their own training frameworks.

Available names:
- ``unet3d``     our shared scaffold (conv/dwsep block; the RF-sweep model, S4 Probe 3).
- ``dynunet``    nnU-Net's architecture (MONAI DynUNet) -- the CNN anchor (S5).
- ``unetr``      transformer anchor (MONAI UNETR).
- ``swin_unetr`` Swin transformer anchor (MONAI SwinUNETR).
- ``segmamba``   Mamba/state-space anchor (optional; needs mamba-ssm + CUDA -> GPU only).
"""
from __future__ import annotations

from monai.networks.nets import UNETR, DynUNet, SwinUNETR
from torch import nn

from .scaffold import IN_CHANNELS, OUT_CHANNELS, build_scaffold


def _dynunet(cfg) -> nn.Module:
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


def _unetr(cfg) -> nn.Module:
    return UNETR(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        img_size=tuple(cfg.train.patch_size),  # must be divisible by 16
        feature_size=16,
        norm_name="instance",
        spatial_dims=3,
    )


def _swin_unetr(cfg) -> nn.Module:
    # SwinUNETR auto-handles input size if divisible by 32; feature_size divisible by 12.
    return SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        feature_size=24,
    )


def _segmamba(cfg) -> nn.Module:
    from .segmamba import build_segmamba  # optional import (needs mamba-ssm)

    return build_segmamba(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS,
                          features=tuple(cfg.model.features))


_BUILDERS = {"dynunet": _dynunet, "unetr": _unetr, "swin_unetr": _swin_unetr, "segmamba": _segmamba}


def build_model(cfg) -> nn.Module:
    """Build the architecture named by ``cfg.model.name`` under the matched protocol."""
    name = cfg.model.name
    if name == "unet3d":
        return build_scaffold(
            block=cfg.model.block, features=tuple(cfg.model.features), kernel_size=cfg.model.kernel_size
        )
    if name not in _BUILDERS:
        raise ValueError(f"unknown model name {name!r}; choose from "
                         f"{['unet3d', *sorted(_BUILDERS)]}")
    return _BUILDERS[name](cfg)
