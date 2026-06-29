"""CONSEQUENCE metric: comparative missing-modality fragility (roadmap S3.3).

MANDATORY for any 'wrong reasons' claim. For each region we compare the Dice drop from
removing the modality the model *leans on most* (argmax reliance) versus the *physics-
correct* modality (from the physics key). If leaning on a correlate makes it break
harder, the unfaithful reliance is consequential.

Preserve the COMPARATIVE structure (S3.3): a generic 'Dice drop when missing' number
does not prove wrongness -- the leaned-on-vs-physics-correct *gap* is the evidence.
"""
from __future__ import annotations

from collections import defaultdict

from ..constants import CHANNEL_ORDER, REGION_ORDER
from .reliance import _load_case, intervene_tensor
from .stats import bootstrap_ci


def _leaned_on_by_region(reliance_rows: list[dict]) -> dict[str, str]:
    """Region -> modality with the highest aggregated reliance score."""
    best: dict[str, tuple[float, str]] = {}
    for row in reliance_rows:
        score = row["score"]
        cur = best.get(row["region"])
        if cur is None or score > cur[0]:
            best[row["region"]] = (score, row["modality"])
    return {region: mod for region, (_, mod) in best.items()}


def comparative_fragility(
    model,
    case_dirs,
    cfg,
    physics_key: dict,
    reliance_rows: list[dict],
    device=None,
    fill: str = "mean",
) -> list[dict]:
    """Compare GT-Dice drop from removing the leaned-on vs physics-correct modality.

    Returns ``FRAGILITY_COLUMNS`` rows: for each region, one row for the ``leaned_on``
    modality and one for the ``physics_correct`` modality, with mean full/dropped Dice,
    the mean drop (``delta``), and a bootstrap CI on the drop across cases.
    """
    import torch

    from ..engine import get_device, infer_volume
    from .segmentation import compute_case_metrics, postprocess

    device = device or get_device()
    model = model.to(device).eval()
    leaned = _leaned_on_by_region(reliance_rows)
    physics = {r: physics_key["classes"][r]["physics_modality"] for r in REGION_ORDER}

    # Per (region, modality): collect (dice_full, dice_dropped) across cases.
    full: dict[str, list[float]] = defaultdict(list)
    dropped: dict[tuple[str, str], list[float]] = defaultdict(list)
    with torch.no_grad():
        for case_dir in case_dirs:
            image, label = _load_case(case_dir, cfg)
            base_pred = postprocess(infer_volume(model, image.unsqueeze(0), cfg, device))[0].cpu()
            base_metrics = {m["region"]: m["dice"] for m in compute_case_metrics(base_pred, label)}
            for region in REGION_ORDER:
                full[region].append(base_metrics[region])
            for ci, modality in enumerate(CHANNEL_ORDER):
                interv = intervene_tensor(image, ci, fill)
                pred = postprocess(infer_volume(model, interv.unsqueeze(0), cfg, device))[0].cpu()
                m = {x["region"]: x["dice"] for x in compute_case_metrics(pred, label)}
                for region in REGION_ORDER:
                    dropped[(region, modality)].append(m[region])

    rows: list[dict] = []
    for region in REGION_ORDER:
        for role, modality in (("leaned_on", leaned.get(region)), ("physics_correct", physics[region])):
            if modality is None:
                continue
            full_vals = full[region]
            drop_vals = dropped[(region, modality)]
            deltas = [f - d for f, d in zip(full_vals, drop_vals)]
            ci_low, ci_high = bootstrap_ci(deltas)
            rows.append({
                "region": region,
                "removed_modality": modality,
                "role": role,
                "dice_full": sum(full_vals) / len(full_vals),
                "dice_dropped": sum(drop_vals) / len(drop_vals),
                "delta": sum(deltas) / len(deltas),
                "ci_low": ci_low,
                "ci_high": ci_high,
            })
    return rows
