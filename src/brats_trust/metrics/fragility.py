"""CONSEQUENCE metric: comparative missing-modality fragility (roadmap S3.3).

MANDATORY for any 'wrong reasons' claim. Remove the modality the model *wrongly
leans on* vs the *physics-correct* one and compare the Dice drop. If leaning on
the correlate makes it break harder, the unfaithful reliance is consequential.

Preserve the COMPARATIVE structure: a generic 'Dice drop when missing' number
does NOT prove wrongness (roadmap S3.3).

STUB: interface only; implement in Stage 4.
"""
from __future__ import annotations


def comparative_fragility(model, dataloader, physics_key: dict, fill: str = "mean"):
    """Per class, compare Dice drop from removing the leaned-on vs physics-correct
    modality. Return both drops + their gap, with CIs. TODO(Stage 4)."""
    raise NotImplementedError("Stage 4: implement comparative missing-modality fragility")
