"""End-to-end evaluation that writes the paper-ready outputs (roadmap S3).

One function, :func:`evaluate_and_log`, takes a trained model + a set of cases and emits
every tidy artifact the paper needs into ``ctx.results_dir`` using the canonical schemas:
segmentation quality (per-case + aggregate), the conditional reliance matrix (S3.1), and
comparative missing-modality fragility (S3.3). Reused by both ``scripts/evaluate.py`` and
the synthetic sanity check so the outputs are always identical in shape.

(Inference is run per concern for clarity; on full volumes this repeats baseline forward
passes -- acceptable now, an obvious optimization later.)
"""
from __future__ import annotations

import torch

from .constants import REGION_ORDER
from .data.splits import patient_id
from .engine import get_device, infer_volume
from .logging_utils import (
    AGGREGATE_COLUMNS,
    FRAGILITY_COLUMNS,
    PER_CASE_COLUMNS,
    RELIANCE_COLUMNS,
    write_tidy,
)
from .metrics.fragility import comparative_fragility
from .metrics.reliance import _load_case, aggregate_reliance, collect_reliance_deltas
from .metrics.segmentation import compute_case_metrics, postprocess
from .metrics.stats import summarize

_SEG_METRICS = ("dice", "hd95", "sensitivity", "specificity")


def _per_case_metrics(model, case_dirs, cfg, device) -> list[dict]:
    rows: list[dict] = []
    with torch.no_grad():
        for case_dir in case_dirs:
            cid = case_dir.name
            image, label = _load_case(case_dir, cfg)
            pred = postprocess(infer_volume(model, image.unsqueeze(0), cfg, device))[0].cpu()
            for m in compute_case_metrics(pred, label):
                rows.append({"case_id": cid, "patient_id": patient_id(cid), **m})
    return rows


def _aggregate(per_case: list[dict]) -> list[dict]:
    agg: list[dict] = []
    for region in REGION_ORDER:
        for metric in _SEG_METRICS:
            vals = [r[metric] for r in per_case if r["region"] == region and r[metric] == r[metric]]
            agg.append({"region": region, "metric": metric, **summarize(vals)})
    return agg


def evaluate_and_log(model, case_dirs, cfg, ctx, device=None, physics_key=None, fill=None) -> dict:
    """Run full evaluation on ``case_dirs`` and write all tidy result files.

    Returns a small summary dict (also handy for assertions/tests).
    """
    device = device or get_device()
    fill = fill or cfg.ablation.fill
    case_dirs = list(case_dirs)
    model = model.to(device).eval()

    per_case = _per_case_metrics(model, case_dirs, cfg, device)
    write_tidy(ctx.result("per_case_metrics"), per_case, PER_CASE_COLUMNS)

    aggregate = _aggregate(per_case)
    write_tidy(ctx.result("aggregate_metrics"), aggregate, AGGREGATE_COLUMNS)

    reliance_rows = collect_reliance_deltas(model, case_dirs, cfg, device=device, fill=fill)
    reliance_matrix = aggregate_reliance(reliance_rows, fill=fill)
    write_tidy(ctx.result("reliance_matrix"), reliance_matrix, RELIANCE_COLUMNS)

    fragility = []
    if physics_key is not None:
        fragility = comparative_fragility(
            model, case_dirs, cfg, physics_key, reliance_matrix, device=device, fill=fill
        )
        write_tidy(ctx.result("fragility"), fragility, FRAGILITY_COLUMNS)

    ctx.logger.info(
        "Evaluation: %d cases | %d reliance cells | %d fragility rows",
        len(case_dirs), len(reliance_matrix), len(fragility),
    )
    return {
        "n_cases": len(case_dirs),
        "reliance_matrix": reliance_matrix,
        "fragility": fragility,
    }
