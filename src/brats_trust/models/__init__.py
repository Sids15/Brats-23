"""Model layer (roadmap S4/S5).

Each architecture is its own module exposing ``build(cfg)``: ``unet3d`` (the shared scaffold
+ pluggable blocks for the RF sweep) and the Tier-A anchors ``dynunet``, ``unetr``,
``swin_unetr``, ``segmamba``. Shared primitives (channel contract, blocks, skip alignment)
live in ``base``. Build any of them by name through :func:`factory.build_model`.
"""
from .factory import MODELS, build_model  # noqa: F401

__all__ = ["build_model", "MODELS"]
