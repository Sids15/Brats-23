"""Aggregate results across runs into paper tables + statistics (roadmap S3, S4.2).

    python scripts/aggregate.py --runs runs/probe3_rf_small_seed* --out outputs/agg_rf_small

Writes (under --out): aggregate_segmentation.{csv,jsonl}, reliance_matrix.{csv,jsonl},
fragility_gap.{csv,jsonl} (with Holm-corrected p-values), and summary.json.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from brats_trust.aggregate import aggregate_runs
from brats_trust.logging_utils import write_json, write_tidy


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate run results across seeds/variants.")
    ap.add_argument("--runs", nargs="+", required=True, help="Run directories (shell-glob ok).")
    ap.add_argument("--out", default="outputs/aggregate")
    args = ap.parse_args()

    out = aggregate_runs(args.runs)
    out_dir = Path(args.out)
    write_tidy(out_dir / "aggregate_segmentation", out["segmentation"])
    write_tidy(out_dir / "reliance_matrix", out["reliance"])
    write_tidy(out_dir / "fragility_gap", out["fragility_gap"])
    write_json(out_dir / "summary.json", {"n_runs": out["n_runs"], "fragility_gap": out["fragility_gap"]})
    print(f"aggregated {out['n_runs']} runs -> {out_dir}")
    for row in out["fragility_gap"]:
        print(f"  {row['region']}: mean gap={row['mean_gap']:.3f} "
              f"CI[{row['ci_low']:.3f},{row['ci_high']:.3f}] p_holm={row['p_holm']:.3g}")


if __name__ == "__main__":
    main()
