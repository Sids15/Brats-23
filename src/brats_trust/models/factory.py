"""Model factory (roadmap S4 Probe 1 mechanism swap + S5 Tier-A anchors).

One entry point, :func:`build_model`, builds any architecture from ``cfg.model.name`` so the
shared training/measurement pipeline runs each under the *matched global protocol* (S9) --
the off-the-shelf nets are used as architectures, not via their own training frameworks.
"""
from __future__ import annotations

from torch import nn

from .base import IN_CHANNELS, OUT_CHANNELS
from .unet3d import build_scaffold
from .dynunet import build_dynunet
from .unetr import build_unetr
from .swin_unetr import build_swin_unetr
from .segmamba import build_segmamba


_BUILDERS = {
    "unet3d": lambda cfg: build_scaffold(
        block=cfg.model.block,
        features=tuple(cfg.model.features),
        kernel_size=cfg.model.kernel_size,
    ),
    "dynunet": build_dynunet,
    "unetr": build_unetr,
    "swin_unetr": build_swin_unetr,
    "segmamba": lambda cfg: build_segmamba(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        features=tuple(cfg.model.features),
    ),
}


def build_model(cfg) -> nn.Module:
    """Build the architecture named by ``cfg.model.name`` under the matched protocol."""
    name = cfg.model.name
    if name not in _BUILDERS:
        raise ValueError(
            f"unknown model name {name!r}; choose from {sorted(_BUILDERS)}"
        )
    return _BUILDERS[name](cfg)
