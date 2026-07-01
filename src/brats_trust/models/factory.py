"""Model registry (roadmap S4 Probe 1 mechanism swap + S5 Tier-A anchors).

One entry point, :func:`build_model`, builds any architecture from ``cfg.model.name`` so the
shared training/measurement pipeline runs each under the *matched global protocol* (S9) --
the off-the-shelf nets are used as architectures, not via their own training frameworks.
Each architecture lives in its own module exposing ``build(cfg)``; this file only maps names
to those builders.

Available names (`MODELS`):
- ``unet3d``     our shared scaffold (conv/dwsep block; the RF-sweep model, S4 Probe 3).
- ``dynunet``    nnU-Net's architecture (MONAI DynUNet) -- the CNN anchor (S5).
- ``unetr``      transformer anchor (MONAI UNETR).
- ``swin_unetr`` Swin transformer anchor (MONAI SwinUNETR).
- ``segmamba``   Mamba/state-space anchor (optional; needs mamba-ssm + CUDA -> GPU only).
"""
from __future__ import annotations

from torch import nn

from . import dynunet, segmamba, swin_unetr, unet3d, unetr

_REGISTRY = {
    "unet3d": unet3d.build,
    "dynunet": dynunet.build,
    "unetr": unetr.build,
    "swin_unetr": swin_unetr.build,
    "segmamba": segmamba.build,
}

MODELS = tuple(_REGISTRY)


def build_model(cfg) -> nn.Module:
    """Build the architecture named by ``cfg.model.name`` under the matched protocol."""
    name = cfg.model.name
    if name not in _REGISTRY:
        raise ValueError(f"unknown model name {name!r}; choose from {list(MODELS)}")
    return _REGISTRY[name](cfg)
