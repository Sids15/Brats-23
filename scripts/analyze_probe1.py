"""Analyze the Probe 1 / Tier-A architecture sweep (roadmap S4 Probe 1, S5, S4.2).

Turns the per-run summary from `run_phase3_robust.py` (or `scripts/run_probe1.py`) into the
three things the paper needs from Stage 3:

  1. a per-architecture table -- faithfulness, ERF, val Dice, params, FLOPs, each with a
     bootstrap CI across seeds, so capacity differences are visible next to the reliance
     differences they might explain;
  2. pairwise faithfulness comparisons with a non-parametric effect size (Cliff's delta) and
     Holm-corrected p-values, because five architectures means ten tests (S4.2);
  3. the ERF<->faithfulness scatter across mechanisms -- the Probe 3 relationship re-tested
     where receptive field varies for a *different* reason.

The Spearman rho printed here is NOT the decisive S4.1 result: across architectures, ERF
covaries with optimization and inductive bias. It corroborates or contradicts Probe 3; it
never replaces it.

    python scripts/analyze_probe1.py --summary outputs/phase3/phase3_summary.jsonl
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: write PNG, no display
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu, spearmanr  # noqa: E402

from brats_trust.figures import plot_erf_faithfulness  # noqa: E402
from brats_trust.logging_utils import write_json, write_tidy  # noqa: E402
from brats_trust.metrics.stats import bootstrap_ci, cliffs_delta, holm_correction  # noqa: E402

ARCH_COLUMNS = (
    "model", "n_seeds", "faithfulness_mean", "faithfulness_ci_low", "faithfulness_ci_high",
    "erf_mean", "val_dice_mean", "params_m", "flops_g",
)


def load_summary(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _finite(rows: list[dict], key: str) -> np.ndarray:
    vals = np.array([float(r.get(key, np.nan)) for r in rows], dtype=float)
    return vals[np.isfinite(vals)]


def per_architecture(rows: list[dict]) -> list[dict]:
    """One row per architecture: central tendency + bootstrap CI of faithfulness across seeds."""
    out = []
    for model in sorted({r["model"] for r in rows}):
        runs = [r for r in rows if r["model"] == model]
        faith = _finite(runs, "faithfulness_overall")
        ci_low, ci_high = bootstrap_ci(faith) if faith.size else (float("nan"), float("nan"))
        params = _finite(runs, "params")
        flops = _finite(runs, "flops")
        out.append({
            "model": model,
            "n_seeds": len(runs),
            "faithfulness_mean": float(faith.mean()) if faith.size else float("nan"),
            "faithfulness_ci_low": ci_low,
            "faithfulness_ci_high": ci_high,
            "erf_mean": float(_finite(runs, "erf").mean()) if _finite(runs, "erf").size else float("nan"),
            "val_dice_mean": float(_finite(runs, "val_dice").mean()) if _finite(runs, "val_dice").size else float("nan"),
            "params_m": float(params.mean() / 1e6) if params.size else float("nan"),
            "flops_g": float(flops.mean() / 1e9) if flops.size else float("nan"),
        })
    return out


def pairwise_faithfulness(rows: list[dict]) -> list[dict]:
    """Every architecture pair: Cliff's delta + Mann-Whitney p, Holm-corrected across pairs."""
    by_model = {m: _finite([r for r in rows if r["model"] == m], "faithfulness_overall")
                for m in sorted({r["model"] for r in rows})}
    pairs = []
    for a, b in combinations(by_model, 2):
        va, vb = by_model[a], by_model[b]
        if va.size < 2 or vb.size < 2:
            continue  # a single seed cannot support a test; report it as absent, not as p=1
        pairs.append({
            "model_a": a,
            "model_b": b,
            "delta_faithfulness": float(va.mean() - vb.mean()),
            "cliffs_delta": cliffs_delta(va, vb),
            "p_raw": float(mannwhitneyu(va, vb, alternative="two-sided").pvalue),
        })
    if pairs:
        for pair, p_holm in zip(pairs, holm_correction([p["p_raw"] for p in pairs])):
            pair["p_holm"] = float(p_holm)
    return pairs


def make_figure(rows: list[dict], arch_rows: list[dict], out_path: Path) -> None:
    fig, (ax_bar, ax_scatter) = plt.subplots(1, 2, figsize=(11, 4.2))

    names = [a["model"] for a in arch_rows]
    means = [a["faithfulness_mean"] for a in arch_rows]
    lo = [a["faithfulness_mean"] - a["faithfulness_ci_low"] for a in arch_rows]
    hi = [a["faithfulness_ci_high"] - a["faithfulness_mean"] for a in arch_rows]
    ax_bar.bar(names, means, yerr=[lo, hi], capsize=4, color="steelblue")
    ax_bar.set_ylabel("Faithfulness (reliance share on physics modality)")
    ax_bar.set_title("Faithfulness by architecture (95% CI over seeds)")
    ax_bar.tick_params(axis="x", rotation=30)

    plot_erf_faithfulness(ax_scatter, rows)  # colors by `model` when `variant` is absent

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze the Probe 1 / Tier-A architecture sweep.")
    ap.add_argument("--summary", required=True, help="phase3_summary.jsonl / probe1_summary.jsonl")
    ap.add_argument("--out", default=None, help="Output dir (default: alongside the summary).")
    args = ap.parse_args()

    summary_path = Path(args.summary)
    out_dir = Path(args.out) if args.out else summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_summary(summary_path)
    if not rows:
        raise SystemExit(f"no runs found in {summary_path}")

    arch_rows = per_architecture(rows)
    pairs = pairwise_faithfulness(rows)

    erf = np.array([r["erf"] for r in rows], dtype=float)
    faith = np.array([r["faithfulness_overall"] for r in rows], dtype=float)
    mask = np.isfinite(erf) & np.isfinite(faith)
    rho, pval = (spearmanr(erf[mask], faith[mask]) if mask.sum() >= 3
                 else (float("nan"), float("nan")))

    write_tidy(out_dir / "architecture_summary", arch_rows, ARCH_COLUMNS)
    write_tidy(out_dir / "faithfulness_pairwise", pairs)
    write_json(out_dir / "probe1_stats.json", {
        "n_runs": len(rows),
        "n_architectures": len(arch_rows),
        "erf_faithfulness_spearman_rho": float(rho),
        "erf_faithfulness_p_value": float(pval),
    })
    make_figure(rows, arch_rows, out_dir / "architecture_faithfulness.png")

    print(f"{'model':<12}{'n':>3}{'faith':>9}{'  95% CI':>18}{'ERF':>8}{'valDice':>9}"
          f"{'params(M)':>11}{'FLOPs(G)':>10}")
    for a in arch_rows:
        print(f"{a['model']:<12}{a['n_seeds']:>3}{a['faithfulness_mean']:>9.3f}"
              f"  [{a['faithfulness_ci_low']:.3f}, {a['faithfulness_ci_high']:.3f}]"
              f"{a['erf_mean']:>8.1f}{a['val_dice_mean']:>9.3f}"
              f"{a['params_m']:>11.2f}{a['flops_g']:>10.1f}")

    significant = [p for p in pairs if p.get("p_holm", 1.0) < 0.05]
    print(f"\nERF vs faithfulness across architectures: Spearman rho={rho:.2f}, p={pval:.3g}, "
          f"n={int(mask.sum())}")
    print("  (corroboration only -- architecture swaps move RF, optimization and inductive "
          "bias together; Probe 3 remains the controlled test.)")
    print(f"\n{len(significant)}/{len(pairs)} architecture pairs differ in faithfulness "
          f"after Holm correction:")
    for p in significant:
        print(f"  {p['model_a']} vs {p['model_b']}: delta={p['delta_faithfulness']:+.3f}, "
              f"cliffs_delta={p['cliffs_delta']:+.2f}, p_holm={p['p_holm']:.3g}")

    print(f"\nwrote {out_dir / 'architecture_summary.csv'}, "
          f"{out_dir / 'faithfulness_pairwise.csv'}, {out_dir / 'probe1_stats.json'}")
    print("figure ->", out_dir / "architecture_faithfulness.png")


if __name__ == "__main__":
    main()
