"""Model layer: the single shared U-Net scaffold and pluggable blocks for the
controlled probes (roadmap S4). Tier-A off-the-shelf anchors live here too (S5)."""
from __future__ import annotations

from .factory import MODELS, build_model
from .unet3d import build_scaffold, UNet3D
from .base import IN_CHANNELS, OUT_CHANNELS, estimate_flops

__all__ = ["MODELS", "build_model", "build_scaffold", "UNet3D",
           "IN_CHANNELS", "OUT_CHANNELS", "estimate_flops"]
