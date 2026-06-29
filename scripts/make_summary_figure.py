"""Assemble the summary 'poster-child' figure (roadmap S11).

One dense 2x2 panel from the tidy outputs of earlier stages. Each panel is drawn only if
its input file is given, so you can build the figure incrementally as stages complete.

    python scripts/make_summary_figure.py \
        --reliance outputs/agg_rf_small/reliance_matrix.jsonl \
        --probe3   outputs/probe3/probe3_summary.jsonl \
        --fragility outputs/agg_rf_small/fragility_gap.jsonl \
        --xai      runs/<xai_run>/results/xai_fails.jsonl \
        --out      outputs/summary_figure.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from brats_trust.figures import (  # noqa: E402
    plot_erf_faithfulness,
    plot_fragility_gap,
    plot_reliance_heatmap,
    plot_xai_fails,
)


def _read(path):
    if not path:
        return None
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the summary figure (roadmap S11).")
    ap.add_argument("--reliance")
    ap.add_argument("--probe3")
    ap.add_argument("--fragility")
    ap.add_argument("--xai")
    ap.add_argument("--out", default="outputs/summary_figure.png")
    args = ap.parse_args()

    panels = [
        (plot_reliance_heatmap, _read(args.reliance), "reliance"),
        (plot_erf_faithfulness, _read(args.probe3), "ERF↔faithfulness"),
        (plot_fragility_gap, _read(args.fragility), "fragility"),
        (plot_xai_fails, _read(args.xai), "XAI-fails"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for (fn, data, name), ax in zip(panels, axes.flat):
        if data:
            fn(ax, data)
        else:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{name}\n(no data)", ha="center", va="center", color="gray")
    fig.suptitle("BraTS-Trust — faithful modality reliance", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print("summary figure ->", out)


if __name__ == "__main__":
    main()
