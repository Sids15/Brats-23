"""Frozen conventions for BraTS-Trust (roadmap S9 'Global protocol').

These are *design-frozen*. The channel order and label/region definitions are
referenced by the ablation dataloader (indexes modalities by position) and by
every metric. Do not reorder without re-running everything.
"""
from __future__ import annotations

# --- Frozen channel order (roadmap S9): [FLAIR, T1, T1CE, T2] ----------------
# The ablation dataloader indexes modalities by *position* in this tuple.
CHANNEL_ORDER: tuple[str, ...] = ("FLAIR", "T1", "T1CE", "T2")

# Canonical modality name -> BraTS-2023 file suffix (files are `*-{suffix}.nii`).
MODALITY_SUFFIX: dict[str, str] = {
    "FLAIR": "t2f",   # T2 Fluid-Attenuated Inversion Recovery
    "T1": "t1n",      # T1 native (pre-contrast)
    "T1CE": "t1c",    # T1 contrast-enhanced (post-gadolinium)
    "T2": "t2w",      # T2-weighted
}
SEG_SUFFIX: str = "seg"

# --- BraTS-2023 segmentation label values (in the seg.nii volume) ------------
LABELS: dict[int, str] = {
    1: "NCR",   # necrotic tumor core
    2: "ED",    # peritumoral edematous/invaded tissue
    3: "ET",    # GD-enhancing tumor
}

# --- Evaluation regions (overlapping output channels), roadmap S9 ------------
# WT = whole tumor, TC = tumor core, ET = enhancing tumor.
REGIONS: dict[str, tuple[int, ...]] = {
    "WT": (1, 2, 3),
    "TC": (1, 3),
    "ET": (3,),
}
REGION_ORDER: tuple[str, ...] = ("WT", "TC", "ET")


def channel_index(modality: str) -> int:
    """Return the frozen channel position of a modality name (e.g. 'T1CE' -> 2)."""
    return CHANNEL_ORDER.index(modality)
