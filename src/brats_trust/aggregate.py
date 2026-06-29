"""Cross-run aggregation + statistics (roadmap S3, S4.2).

Pools the tidy ``results/`` of many runs (e.g. seeds of one variant) into paper-ready
summaries: segmentation metrics with bootstrap CIs, the reliance matrix averaged across
seeds, and the significance of the comparative-fragility *gap* (leaned-on vs physics-correct)
with Holm-corrected p-values across regions (S3.3 + S3 stats requirement).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .constants import REGION_ORDER
from .metrics.stats import bootstrap_ci, holm_correction, summarize

_SEG_METRICS = ("dice", "hd95", "sensitivity", "specificity")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_run_results(run_dirs, stem: str) -> list[dict]:
    """Concatenate ``results/<stem>.jsonl`` across runs, tagging each row with its run name."""
    rows: list[dict] = []
    for d in run_dirs:
        path = Path(d) / "results" / f"{stem}.jsonl"
        if path.exists():
            rows.extend({"run": Path(d).name, **r} for r in _read_jsonl(path))
    return rows


def aggregate_segmentation(per_case_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for region in REGION_ORDER:
        for metric in _SEG_METRICS:
            vals = [r[metric] for r in per_case_rows
                    if r["region"] == region and isinstance(r.get(metric), (int, float)) and r[metric] == r[metric]]
            out.append({"region": region, "metric": metric, **summarize(vals)})
    return out


def aggregate_reliance(reliance_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in reliance_rows:
        grouped[(r["region"], r["modality"])].append(float(r["score"]))
    out: list[dict] = []
    for (region, modality), vals in sorted(grouped.items()):
        lo, hi = bootstrap_ci(vals)
        out.append({"region": region, "modality": modality, "n": len(vals),
                    "score": float(np.mean(vals)), "ci_low": lo, "ci_high": hi})
    return out


def fragility_gap_significance(fragility_rows: list[dict]) -> list[dict]:
    """Per region, test whether the leaned-on modality drops Dice MORE than the physics one.

    gap_run = delta(leaned_on) - delta(physics_correct) for each run; we report the mean gap,
    a bootstrap CI, and a Holm-corrected one-sample p-value (gap != 0) across regions. A
    positive, significant gap is the consequential-unfaithfulness evidence (S3.3).
    """
    from scipy.stats import wilcoxon

    per_run: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for r in fragility_rows:
        per_run[(r["run"], r["region"])][r["role"]] = r["delta"]

    gaps: dict[str, list[float]] = defaultdict(list)
    for (_, region), roles in per_run.items():
        if "leaned_on" in roles and "physics_correct" in roles:
            gaps[region].append(roles["leaned_on"] - roles["physics_correct"])

    regions = [r for r in REGION_ORDER if gaps.get(r)]
    rows: list[dict] = []
    raw_p: list[float] = []
    for region in regions:
        vals = np.asarray(gaps[region], dtype=float)
        lo, hi = bootstrap_ci(vals)
        try:
            p = float(wilcoxon(vals).pvalue) if vals.size >= 1 and np.any(vals != 0) else float("nan")
        except ValueError:
            p = float("nan")
        raw_p.append(p)
        rows.append({"region": region, "n": int(vals.size), "mean_gap": float(vals.mean()),
                     "ci_low": lo, "ci_high": hi, "p_raw": p})

    finite = [p for p in raw_p if p == p]
    corrected = list(holm_correction(finite)) if finite else []
    it = iter(corrected)
    for row in rows:
        row["p_holm"] = float(next(it)) if row["p_raw"] == row["p_raw"] else float("nan")
    return rows


def aggregate_runs(run_dirs) -> dict:
    """Load + aggregate all three result kinds across ``run_dirs``."""
    run_dirs = list(run_dirs)
    return {
        "segmentation": aggregate_segmentation(load_run_results(run_dirs, "per_case_metrics")),
        "reliance": aggregate_reliance(load_run_results(run_dirs, "reliance_matrix")),
        "fragility_gap": fragility_gap_significance(load_run_results(run_dirs, "fragility")),
        "n_runs": len(run_dirs),
    }
