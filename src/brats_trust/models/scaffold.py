"""Single shared U-Net scaffold (roadmap S4).

'Matched stages/widths/skips/patch/optimizer/schedule/init/data/seeds; one
pluggable block.' Everything except the swapped block is held identical so the
probes (S4) vary one thing at a time:

  - Probe 3 (PRIMARY): receptive field via depthwise-separable / bottleneck
    convs, params & FLOPs held fixed -> the ERF<->faithfulness curve (S4.1).
  - Probe 1 (SECONDARY): encoder mechanism (conv / attention / Mamba).
  - Probe 2 (CONTROL): decoder swap, identical conv encoder.

STUB: build a 4->3 channel 3D U-Net with a pluggable block factory in Stage 0/2.
"""
from __future__ import annotations

IN_CHANNELS = 4   # [FLAIR, T1, T1CE, T2]
OUT_CHANNELS = 3  # [WT, TC, ET] overlapping regions


def build_scaffold(block: str = "conv", **kwargs):
    """Return a 3D U-Net with the chosen pluggable ``block``. TODO(Stage 0/2)."""
    raise NotImplementedError("Stage 0/2: implement shared U-Net scaffold + block factory")
