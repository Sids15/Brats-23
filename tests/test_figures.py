"""Stage 5 test: the summary figure panels render from tidy rows without error."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brats_trust.figures import (  # noqa: E402
    plot_erf_faithfulness,
    plot_fragility_gap,
    plot_reliance_heatmap,
    plot_xai_fails,
)


def test_summary_panels_render(tmp_path):
    reliance = [{"region": r, "modality": m, "score": 0.5}
                for r in ("WT", "TC", "ET") for m in ("FLAIR", "T1", "T1CE", "T2")]
    probe3 = [{"variant": "rf_small", "erf": 7.0, "faithfulness_overall": 0.6},
              {"variant": "rf_med", "erf": 9.0, "faithfulness_overall": 0.5},
              {"variant": "rf_large", "erf": 11.0, "faithfulness_overall": 0.4}]
    gap = [{"region": "ET", "mean_gap": 0.3, "ci_low": 0.2, "ci_high": 0.4}]
    xai = [{"saliency_cosine": 0.9}, {"saliency_cosine": 0.85}]

    fig, axes = plt.subplots(2, 2)
    plot_reliance_heatmap(axes[0, 0], reliance)
    plot_erf_faithfulness(axes[0, 1], probe3)
    plot_fragility_gap(axes[1, 0], gap)
    plot_xai_fails(axes[1, 1], xai)
    out = tmp_path / "summary.png"
    fig.savefig(out)
    assert out.exists() and out.stat().st_size > 0
