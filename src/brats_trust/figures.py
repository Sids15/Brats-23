"""Summary figure panels (roadmap S11 'the poster child').

Each function draws one panel onto a Matplotlib axis from the tidy result rows produced by
earlier stages, so the headline figure regenerates deterministically from disk:
  reliance matrix · ERF<->faithfulness curve · comparative-fragility gap · XAI-fails.
"""
from __future__ import annotations

import numpy as np

from .constants import CHANNEL_ORDER, REGION_ORDER


def plot_reliance_heatmap(ax, reliance_rows: list[dict]) -> None:
    """Regions x modalities heatmap of reliance scores (roadmap S3.1)."""
    grid = np.full((len(REGION_ORDER), len(CHANNEL_ORDER)), np.nan)
    for r in reliance_rows:
        if r["region"] in REGION_ORDER and r["modality"] in CHANNEL_ORDER:
            grid[REGION_ORDER.index(r["region"]), CHANNEL_ORDER.index(r["modality"])] = r["score"]
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(CHANNEL_ORDER)), CHANNEL_ORDER)
    ax.set_yticks(range(len(REGION_ORDER)), REGION_ORDER)
    ax.set_title("Conditional reliance")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
    im.figure.colorbar(im, ax=ax, fraction=0.046)


def plot_erf_faithfulness(ax, probe3_rows: list[dict]) -> None:
    """ERF vs faithfulness scatter, colored by variant (roadmap S4.1, the money panel)."""
    from scipy.stats import spearmanr

    erf = np.array([r["erf"] for r in probe3_rows], float)
    faith = np.array([r["faithfulness_overall"] for r in probe3_rows], float)
    mask = np.isfinite(erf) & np.isfinite(faith)
    for variant in sorted({r.get("variant", r.get("model", "?")) for r in probe3_rows}):
        idx = [i for i, r in enumerate(probe3_rows)
               if (r.get("variant", r.get("model")) == variant) and mask[i]]
        if idx:
            ax.scatter(erf[idx], faith[idx], label=variant, s=40)
    rho = spearmanr(erf[mask], faith[mask]).correlation if mask.sum() >= 3 else float("nan")
    ax.set_xlabel("Effective receptive field (vox)")
    ax.set_ylabel("Faithfulness")
    ax.set_title(f"ERF vs faithfulness (rho={rho:.2f})")
    ax.legend(fontsize=7)


def plot_fragility_gap(ax, gap_rows: list[dict]) -> None:
    """Per-region leaned-on minus physics-correct Dice drop, with CI (roadmap S3.3)."""
    regions = [r["region"] for r in gap_rows]
    gaps = [r["mean_gap"] for r in gap_rows]
    lo = [r["mean_gap"] - r["ci_low"] for r in gap_rows]
    hi = [r["ci_high"] - r["mean_gap"] for r in gap_rows]
    ax.bar(regions, gaps, yerr=[lo, hi], capsize=4, color="indianred")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Dice-drop gap")
    ax.set_title("Comparative fragility (leaned-on − physics)")


def plot_xai_fails(ax, xai_rows: list[dict]) -> None:
    """Saliency cosine before vs after modality intervention (roadmap S3.4)."""
    cos = [r["saliency_cosine"] for r in xai_rows if r.get("saliency_cosine") == r.get("saliency_cosine")]
    ax.hist(cos, bins=10, range=(0, 1), color="slateblue")
    ax.axvline(float(np.mean(cos)) if cos else 0, color="k", ls="--",
               label=f"mean={np.mean(cos):.2f}" if cos else "n/a")
    ax.set_xlabel("Saliency cosine (base vs T1CE-filled)")
    ax.set_ylabel("cases")
    ax.set_title("XAI-fails: saliency blind to shortcut")
    ax.legend(fontsize=7)
