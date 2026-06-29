"""Analyze the Probe 3 sweep -> the ERF<->faithfulness curve (roadmap S4.1, the money figure).

Reads the per-run summary from run_probe3.py, tests whether effective receptive field
predicts faithfulness (Spearman correlation across runs), writes a stats JSON and a scatter
plot. Report the relationship honestly -- a clean negative trend (bigger ERF -> less
faithful) is the memorable result; a null is still a publishable focused negative result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: write PNG, no display
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402


def load_summary(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze the Probe 3 ERF<->faithfulness relationship.")
    ap.add_argument("--summary", required=True, help="probe3_summary.jsonl from run_probe3.py")
    ap.add_argument("--out", default=None, help="Output dir (default: alongside the summary).")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out) if args.out else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summary(summary_path)

    erf = np.array([r["erf"] for r in rows], dtype=float)
    faith = np.array([r["faithfulness_overall"] for r in rows], dtype=float)
    mask = np.isfinite(erf) & np.isfinite(faith)
    erf, faith = erf[mask], faith[mask]

    rho, pval = (spearmanr(erf, faith) if erf.size >= 3 else (float("nan"), float("nan")))
    stats = {"n_runs": int(erf.size), "spearman_rho": float(rho), "p_value": float(pval)}
    (out_dir / "erf_faithfulness_stats.json").write_text(json.dumps(stats, indent=2))

    # Scatter, colored by variant.
    fig, ax = plt.subplots(figsize=(6, 4))
    variants = sorted({r["variant"] for r in [rows[i] for i in np.where(mask)[0]]})
    for v in variants:
        idx = [i for i, r in enumerate(np.array(rows)[mask]) if r["variant"] == v]
        ax.scatter(erf[idx], faith[idx], label=v, s=40)
    ax.set_xlabel("Effective receptive field (voxels)")
    ax.set_ylabel("Faithfulness (reliance share on physics modality)")
    ax.set_title(f"ERF vs faithfulness  (Spearman rho={rho:.2f}, p={pval:.3g}, n={erf.size})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "erf_vs_faithfulness.png", dpi=150)

    print(json.dumps(stats, indent=2))
    print("figure ->", out_dir / "erf_vs_faithfulness.png")


if __name__ == "__main__":
    main()
