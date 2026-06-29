"""Saliency-based checks (roadmap S3.4 XAI-fails, S3.2 MSFI cross-check).

These exist to *justify the intervention-first stack*, not to replace it:

- **XAI-fails (S3.4):** compute a saliency map, then mean-fill T1CE and recompute. If the
  map barely changes, saliency is blind to a modality-level shortcut -- so you cannot audit
  modality reliance with saliency alone (which is exactly why S3.1 uses interventions).
- **MSFI cross-check (S3.2):** the share of saliency attribution that lands on the physics-
  correct modality channel. A convergent-validity check on the reliance result, not the
  main metric (decodable/visible != used).

We use vanilla input-gradient saliency (no extra deps, robust for 3D); Grad-CAM gives the
same qualitative conclusion and can be swapped in.
"""
from __future__ import annotations

import torch

from ..constants import REGION_ORDER, channel_index
from .reliance import intervene_tensor


def saliency_map(model, image, region_channel: int, device=None) -> torch.Tensor:
    """Per-channel input-gradient saliency ``(C, X, Y, Z)`` for one region's logit."""
    device = device or next(model.parameters()).device
    model.eval()
    x = image.unsqueeze(0).to(device).clone().requires_grad_(True)
    score = model(x)[0, region_channel].sum()
    model.zero_grad(set_to_none=True)
    score.backward()
    return x.grad[0].detach().abs()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.flatten(), b.flatten()
    denom = (a.norm() * b.norm()).item()
    return float((a @ b).item() / denom) if denom > 0 else float("nan")


def xai_fails_check(model, image, region: str = "ET", intervened_modality: str = "T1CE",
                    device=None) -> dict:
    """Quantify how little saliency changes when a modality is removed (roadmap S3.4).

    High ``saliency_cosine`` despite removing ``intervened_modality`` = saliency is blind to
    the modality-level intervention. Returns the cosine similarity of the saliency maps and
    the attribution mass on the intervened channel before/after.
    """
    ri = REGION_ORDER.index(region)
    ci = channel_index(intervened_modality)
    base = saliency_map(model, image, ri, device)
    interv_image = intervene_tensor(image, ci, "mean")
    interv = saliency_map(model, interv_image, ri, device)
    return {
        "region": region,
        "intervened_modality": intervened_modality,
        "saliency_cosine": _cosine(base.sum(0), interv.sum(0)),
        "mass_on_modality_base": float(base[ci].sum() / base.sum()) if base.sum() > 0 else float("nan"),
        "mass_on_modality_intervened": float(interv[ci].sum() / interv.sum()) if interv.sum() > 0 else float("nan"),
    }


def msfi_score(model, image, physics_key: dict, device=None) -> dict[str, float]:
    """Per region: share of saliency attribution on the physics-correct modality channel."""
    scores: dict[str, float] = {}
    for ri, region in enumerate(REGION_ORDER):
        sal = saliency_map(model, image, ri, device)  # (C, X, Y, Z)
        per_channel = sal.flatten(1).sum(dim=1)        # (C,)
        total = per_channel.sum()
        ci = channel_index(physics_key["classes"][region]["physics_modality"])
        scores[region] = float(per_channel[ci] / total) if total > 0 else float("nan")
    return scores
