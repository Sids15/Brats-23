"""Segmentation quality metrics, BraTS-2023 conventions (Dice + HD95 per WT/TC/ET).

Computed per case and per region so the paper can report distributions (CIs, box plots),
not just means. Lesion-wise Dice/HD95 (the additional BraTS-2023 metric) have reserved
columns in ``logging_utils.PER_CASE_COLUMNS`` and are left for a later pass.
"""
from __future__ import annotations

import torch
from monai.metrics import compute_dice, compute_hausdorff_distance

from ..constants import REGION_ORDER


def postprocess(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Sigmoid + threshold logits into a binary multi-region mask (overlapping channels)."""
    return (torch.sigmoid(logits) >= threshold).to(torch.float32)


def _sens_spec(pred: torch.Tensor, target: torch.Tensor) -> tuple[float, float]:
    p = pred.bool()
    t = target.bool()
    tp = (p & t).sum().item()
    tn = (~p & ~t).sum().item()
    fp = (p & ~t).sum().item()
    fn = (~p & t).sum().item()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def compute_case_metrics(pred: torch.Tensor, target: torch.Tensor) -> list[dict]:
    """Per-region metrics for one case. ``pred``/``target`` are binary ``(3, X, Y, Z)``.

    Returns rows ``{region, dice, hd95, sensitivity, specificity}`` (one per WT/TC/ET).
    """
    p = pred.unsqueeze(0)  # (1, 3, X, Y, Z) as MONAI expects (B, C, ...)
    t = target.unsqueeze(0)
    dice = compute_dice(p, t, include_background=True)[0]
    hd95 = compute_hausdorff_distance(p, t, include_background=True, percentile=95)[0]
    rows: list[dict] = []
    for i, region in enumerate(REGION_ORDER):
        sens, spec = _sens_spec(pred[i], target[i])
        rows.append({
            "region": region,
            "dice": float(dice[i]),
            "hd95": float(hd95[i]),
            "sensitivity": sens,
            "specificity": spec,
        })
    return rows
