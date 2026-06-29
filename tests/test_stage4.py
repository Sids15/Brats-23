"""Stage 4 tests: saliency/XAI checks and cross-run aggregation (torch-free where possible)."""
from __future__ import annotations

import torch

from brats_trust.aggregate import (
    aggregate_reliance,
    aggregate_segmentation,
    fragility_gap_significance,
)
from brats_trust.config import load_physics_key
from brats_trust.metrics.xai import msfi_score, saliency_map, xai_fails_check
from brats_trust.models.scaffold import build_scaffold


def test_saliency_and_xai_check_shapes():
    model = build_scaffold(features=[8, 16])
    image = torch.randn(4, 16, 16, 16)
    sal = saliency_map(model, image, region_channel=2)
    assert sal.shape == (4, 16, 16, 16)
    out = xai_fails_check(model, image, region="ET", intervened_modality="T1CE")
    assert -1.0 <= out["saliency_cosine"] <= 1.0
    assert set(out) >= {"saliency_cosine", "mass_on_modality_base", "mass_on_modality_intervened"}


def test_msfi_score_is_a_share():
    model = build_scaffold(features=[8, 16])
    scores = msfi_score(model, torch.randn(4, 16, 16, 16), load_physics_key())
    for region in ("WT", "TC", "ET"):
        assert 0.0 <= scores[region] <= 1.0


def test_aggregate_segmentation_and_reliance():
    per_case = [
        {"region": "ET", "dice": 0.7, "hd95": 3.0, "sensitivity": 0.8, "specificity": 0.99},
        {"region": "ET", "dice": 0.5, "hd95": 5.0, "sensitivity": 0.6, "specificity": 0.98},
    ]
    seg = aggregate_segmentation(per_case)
    et_dice = next(r for r in seg if r["region"] == "ET" and r["metric"] == "dice")
    assert abs(et_dice["mean"] - 0.6) < 1e-9 and et_dice["n"] == 2

    reliance = [
        {"region": "ET", "modality": "T1CE", "score": 0.9},
        {"region": "ET", "modality": "T1CE", "score": 0.7},
    ]
    agg = aggregate_reliance(reliance)
    assert abs(agg[0]["score"] - 0.8) < 1e-9 and agg[0]["n"] == 2


def test_fragility_gap_significance():
    # 5 "runs" where the leaned-on modality drops Dice more than the physics one.
    rows = []
    for i in range(5):
        rows.append({"run": f"s{i}", "region": "ET", "role": "leaned_on", "delta": 0.4 + 0.01 * i})
        rows.append({"run": f"s{i}", "region": "ET", "role": "physics_correct", "delta": 0.1 + 0.01 * i})
    out = fragility_gap_significance(rows)
    et = next(r for r in out if r["region"] == "ET")
    assert et["mean_gap"] > 0.2 and et["n"] == 5
    assert "p_holm" in et
