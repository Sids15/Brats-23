"""Swin transformer anchor (MONAI SwinUNETR, S5).

``feature_size``, ``num_heads``, and ``use_checkpoint`` are read from
``cfg.model.swin_unetr`` so the values live in the YAML reproducibility
record alongside every other protocol knob.

Why ``num_heads`` is exposed: MONAI's SwinUNETR does **not** rescale
``num_heads`` when ``feature_size`` changes.  The two must move together
to keep ``head_dim = feature_size / num_heads[i]`` intact at every stage;
without this, lowering ``feature_size`` silently shrinks ``head_dim`` and
degrades the attention mechanism in the one arm whose mechanism is the
experimental variable.
"""
from __future__ import annotations

from monai.networks.nets import SwinUNETR
from torch import nn

from .base import IN_CHANNELS, OUT_CHANNELS


def build_swin_unetr(cfg) -> nn.Module:
    """Build a SwinUNETR from the config-driven knobs in ``cfg.model.swin_unetr``.

    Reads:
        feature_size   – embedding dimension (must be divisible by 12).
        num_heads      – per-stage head counts; length must match SwinUNETR stages.
        use_checkpoint – gradient checkpointing (recompute activations in backward
                         to trade compute for VRAM; required to fit 24 GB at batch 2
                         with the published feature_size=24).
    """
    swin_cfg = cfg.model.swin_unetr
    return SwinUNETR(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        feature_size=swin_cfg.feature_size,
        num_heads=tuple(swin_cfg.num_heads),
        use_checkpoint=swin_cfg.use_checkpoint,
    )
