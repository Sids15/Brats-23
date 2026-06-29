"""Faithfulness score (roadmap S3.1, S4.1) — collapses the reliance matrix into the
y-axis of the money figure.

For each region we ask: *of all the reliance the model places across modalities, how much
sits on the physically-correct one?* (e.g. ET should rely on T1CE.) That share is the
faithfulness for that region; the overall score averages across regions. Higher = more
faithful. This is one principled summary of the per-class x per-modality reliance matrix;
the matrix itself remains the primary, interpretable result (we never reduce everything to
a single 'trust score' -- see roadmap CUT LIST).
"""
from __future__ import annotations

from ..constants import REGION_ORDER


def faithfulness_score(reliance_matrix: list[dict], physics_key: dict) -> dict[str, float]:
    """Return ``{region: share_on_physics_modality, ..., 'overall': mean}``.

    Args:
        reliance_matrix: rows ``{region, modality, score, ...}`` from
            ``metrics.reliance.aggregate_reliance``.
        physics_key: the loaded physics answer key (gives each region's physics modality).
    """
    by_region: dict[str, dict[str, float]] = {}
    for row in reliance_matrix:
        by_region.setdefault(row["region"], {})[row["modality"]] = max(0.0, float(row["score"]))

    scores: dict[str, float] = {}
    for region in REGION_ORDER:
        modality_scores = by_region.get(region, {})
        total = sum(modality_scores.values())
        physics_modality = physics_key["classes"][region]["physics_modality"]
        if total <= 0:
            scores[region] = float("nan")  # model relied on nothing measurable
        else:
            scores[region] = modality_scores.get(physics_modality, 0.0) / total

    valid = [v for v in scores.values() if v == v]
    scores["overall"] = sum(valid) / len(valid) if valid else float("nan")
    return scores
