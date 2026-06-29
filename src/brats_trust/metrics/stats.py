"""Statistics for reporting (roadmap S3, S4.2): effect sizes, CIs, corrected p-values.

Every comparison in the paper must come with an effect size, a confidence interval, and
a multiple-comparison-corrected p-value across seeds (roadmap S3). These helpers are the
single implementation used everywhere so the numbers are consistent and defensible.
NumPy/SciPy/statsmodels only -- no torch.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    values,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic`` over ``values``.

    Bootstrap (not a normal approximation) because per-case Dice / reliance deltas are
    skewed and small-n; percentile CIs make no distributional assumption.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.array([statistic(rng.choice(arr, size=arr.size, replace=True)) for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (float(np.quantile(boot, alpha)), float(np.quantile(boot, 1 - alpha)))


def cohens_d(a, b) -> float:
    """Standardized mean difference (pooled SD). Magnitude of an effect, unit-free."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def cliffs_delta(a, b) -> float:
    """Non-parametric effect size in [-1, 1]: P(a>b) - P(a<b). Robust to non-normality."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    diff = a[:, None] - b[None, :]
    return float((np.sign(diff).sum()) / (a.size * b.size))


def holm_correction(pvalues) -> np.ndarray:
    """Holm-Bonferroni corrected p-values (controls family-wise error rate)."""
    from statsmodels.stats.multitest import multipletests

    pvalues = np.asarray(pvalues, dtype=float)
    if pvalues.size == 0:
        return pvalues
    _, corrected, _, _ = multipletests(pvalues, method="holm")
    return corrected


def summarize(values) -> dict[str, float]:
    """Aggregate stats matching the ``AGGREGATE_COLUMNS`` schema (n/mean/std/median/IQR/CI)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {k: float("nan") for k in
                ("n", "mean", "std", "median", "iqr_low", "iqr_high", "ci_low", "ci_high")}
    ci_low, ci_high = bootstrap_ci(arr)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median": float(np.median(arr)),
        "iqr_low": float(np.quantile(arr, 0.25)),
        "iqr_high": float(np.quantile(arr, 0.75)),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }
