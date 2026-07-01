"""Tests for the torch-free metrics layer: statistics and reliance aggregation."""
from __future__ import annotations

import math

import numpy as np

from brats_trust.metrics import fragility, reliance, stats


def test_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=5.0, scale=1.0, size=200)
    low, high = stats.bootstrap_ci(data, n_boot=2000, seed=0)
    assert low < data.mean() < high
    assert high - low < 1.0  # reasonably tight for n=200


def test_cohens_d_sign_and_magnitude():
    a = np.zeros(50) + 1.0
    b = np.zeros(50)
    # identical-variance shift -> large positive d; near-zero for equal groups
    assert stats.cohens_d(a + np.linspace(0, 0.1, 50), b + np.linspace(0, 0.1, 50)) > 5
    assert abs(stats.cohens_d(b, b.copy())) < 1e-6 or np.isnan(stats.cohens_d(b, b.copy()))


def test_cliffs_delta_bounds():
    a = np.array([10, 11, 12, 13])
    b = np.array([1, 2, 3, 4])
    assert stats.cliffs_delta(a, b) == 1.0       # a strictly dominates b
    assert stats.cliffs_delta(b, a) == -1.0


def test_holm_correction_is_at_least_raw():
    raw = np.array([0.01, 0.02, 0.03, 0.5])
    corrected = stats.holm_correction(raw)
    assert np.all(corrected >= raw - 1e-12)      # correction never lowers p-values
    assert np.all(corrected <= 1.0)


def test_summarize_schema():
    out = stats.summarize([1.0, 2.0, 3.0, 4.0])
    assert out["n"] == 4
    assert out["median"] == 2.5
    for key in ("mean", "std", "iqr_low", "iqr_high", "ci_low", "ci_high"):
        assert key in out


def test_aggregate_reliance_groups_and_scores():
    deltas = [
        {"region": "ET", "modality": "T1CE", "delta": 0.8},
        {"region": "ET", "modality": "T1CE", "delta": 0.6},
        {"region": "ET", "modality": "FLAIR", "delta": 0.05},
    ]
    rows = reliance.aggregate_reliance(deltas, fill="mean")
    by_mod = {r["modality"]: r for r in rows}
    assert abs(by_mod["T1CE"]["score"] - 0.7) < 1e-9
    assert by_mod["T1CE"]["score"] > by_mod["FLAIR"]["score"]
    assert all(r["fill"] == "mean" for r in rows)
    assert set(rows[0]) == {"region", "modality", "fill", "score", "ci_low", "ci_high"}


def test_summarize_drop_excludes_empty_gt_cases():
    # An empty-GT case gives NaN Dice (MONAI convention); it must be dropped, not
    # allowed to poison the region's mean the way plain sum/len did.
    nan = float("nan")
    full = [0.9, nan, 0.7]
    dropped = [0.5, nan, 0.4]
    out = fragility._summarize_drop(full, dropped)
    assert out["n_cases"] == 2
    assert math.isfinite(out["delta"])
    assert abs(out["delta"] - ((0.9 - 0.5) + (0.7 - 0.4)) / 2) < 1e-9
    assert abs(out["dice_full"] - 0.8) < 1e-9


def test_summarize_drop_all_empty_is_nan_not_crash():
    out = fragility._summarize_drop([float("nan")], [float("nan")])
    assert out["n_cases"] == 0
    assert all(math.isnan(out[k]) for k in ("dice_full", "dice_dropped", "delta"))
