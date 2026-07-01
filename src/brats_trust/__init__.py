"""BraTS-Trust: intervention-based conditional modality-reliance measurement
for 3D brain-tumor segmentation (roadmap: brain_tumor_roadmap_v4_final.md).

Implements the staged build order (roadmap S12): data pipeline, shared model
scaffold with MONAI anchors, and the measurement stack (conditional reliance,
comparative fragility, ERF, faithfulness, statistics, XAI checks).
"""
from __future__ import annotations

__version__ = "0.0.1"

from . import constants  # noqa: F401
from .config import load_config  # noqa: F401
from .logging_utils import MetricLogger, RunContext, set_seed, setup_run  # noqa: F401

__all__ = [
    "constants",
    "load_config",
    "setup_run",
    "set_seed",
    "RunContext",
    "MetricLogger",
    "__version__",
]
