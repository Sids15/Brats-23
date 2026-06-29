"""PRIMARY metric: conditional modality reliance via intervention (roadmap S3.1).

Per class (ET/TC/WT) x modality, measure the change in prediction when a single
modality is intervened on (mean-fill / healthy-prior fill) *conditional on the
other modalities remaining present*. Output is the per-class x per-modality
reliance matrix.

Honest proxy (roadmap S1): this approximates the conditional/unique-information
quantity; the rigorous conditional-MI / PID version is the Fork-B project (S10).

STUB: interface only. Implement in Stage 0 and unit-test against the synthetic
sanity check (S3.5) before trusting it on real MRI.
"""
from __future__ import annotations

from ..constants import CHANNEL_ORDER, REGION_ORDER


def reliance_matrix(model, dataloader, fill: str = "mean"):
    """Return a {region: {modality: score}} reliance matrix. TODO(Stage 0).

    Score = per-class prediction change (e.g. Dice/soft-prob delta) under
    single-modality intervention, conditional on others present.
    """
    _ = (CHANNEL_ORDER, REGION_ORDER)  # shape reference for the implementation
    raise NotImplementedError("Stage 0: implement conditional intervention reliance")
