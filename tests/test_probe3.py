"""Tests for Probe 3 building blocks: ERF measurement and the faithfulness score."""
from __future__ import annotations

import math

from brats_trust.metrics.erf import measure_erf
from brats_trust.metrics.faithfulness import faithfulness_score
from brats_trust.models.scaffold import build_scaffold

_PHYSICS = {"classes": {
    "WT": {"physics_modality": "FLAIR"},
    "TC": {"physics_modality": "T1CE"},
    "ET": {"physics_modality": "T1CE"},
}}


def test_erf_grows_with_kernel_size():
    small = build_scaffold(block="conv", features=[8, 16], kernel_size=3)
    large = build_scaffold(block="conv", features=[8, 16], kernel_size=7)
    erf_small = measure_erf(small, (24, 24, 24), out_channel=2, n_samples=3)
    erf_large = measure_erf(large, (24, 24, 24), out_channel=2, n_samples=3)
    assert math.isfinite(erf_small) and math.isfinite(erf_large)
    assert erf_large > erf_small  # wider kernels integrate context from farther away


def test_faithfulness_score_rewards_physics_modality():
    reliance = [
        {"region": "ET", "modality": "T1CE", "score": 0.9},
        {"region": "ET", "modality": "FLAIR", "score": 0.1},
        {"region": "ET", "modality": "T1", "score": 0.0},
        {"region": "ET", "modality": "T2", "score": 0.0},
        {"region": "TC", "modality": "T1CE", "score": 0.5},
        {"region": "TC", "modality": "FLAIR", "score": 0.5},
        {"region": "WT", "modality": "FLAIR", "score": 1.0},
    ]
    score = faithfulness_score(reliance, _PHYSICS)
    assert abs(score["ET"] - 0.9) < 1e-9   # 0.9 of ET reliance on the physics modality
    assert abs(score["WT"] - 1.0) < 1e-9
    assert 0.0 <= score["overall"] <= 1.0


def test_faithfulness_handles_zero_reliance():
    reliance = [{"region": r, "modality": "T1CE", "score": 0.0} for r in ("WT", "TC", "ET")]
    score = faithfulness_score(reliance, _PHYSICS)
    assert score["ET"] != score["ET"]  # NaN when the model relied on nothing measurable
