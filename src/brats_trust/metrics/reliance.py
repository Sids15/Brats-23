"""PRIMARY metric: conditional modality reliance via intervention (roadmap S3.1).

Per class (ET/TC/WT) x modality, measure the change in prediction when a single
modality is intervened on (mean-fill / healthy-prior fill) *conditional on the
other modalities remaining present*. Output is the per-class x per-modality
reliance matrix.

Honest proxy (roadmap S1): this approximates the conditional/unique-information
quantity; the rigorous conditional-MI / PID version is the Fork-B project (S10).

STUB: interface only. Implement in Stage 0 and unit-test against the synthetic
sanity check (S3.5) before trusting it on real MRI.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .stats import bootstrap_ci

# ----------------------------------------------------------------------------- #
# Shared torch helpers (intervention + binary Dice). Imported by fragility too.
# ----------------------------------------------------------------------------- #
def intervene_tensor(image, channel: int, fill: str = "mean"):
    """Mean/zero-fill one channel of a torch image tensor (delegates to the NumPy
    implementation in ``data.preprocess`` so reliance and training share one definition)."""
    import torch

    from ..data.preprocess import intervene

    arr = image.detach().cpu().numpy()
    return torch.from_numpy(intervene(arr, channel, fill)).to(image)


def dice_score(a, b) -> float:
    """Dice between two binary masks; 1.0 if both empty (a no-change case)."""
    a = a.bool()
    b = b.bool()
    inter = (a & b).sum().item()
    denom = a.sum().item() + b.sum().item()
    if denom == 0:
        return 1.0
    return 2.0 * inter / denom


def _load_case(case_dir, cfg):
    from ..data.dataset import build_transforms, case_to_dict

    sample = build_transforms(cfg, train=False)(case_to_dict(case_dir))
    return sample["image"], sample["label"]


# ----------------------------------------------------------------------------- #
# Model-dependent intervention loop (roadmap S3.1).
# ----------------------------------------------------------------------------- #
def collect_reliance_deltas(model, case_dirs, cfg, device=None, fill: str = "mean") -> list[dict]:
    """Run the conditional intervention per case and return reliance deltas.

    For each case x region (ET/TC/WT) x modality, intervene on that modality (``fill``)
    *while the others stay present* and measure how much the model's prediction changes
    -- ``delta = 1 - Dice(baseline_pred, intervened_pred)`` for that region. Measuring the
    change in the *prediction* (not the GT Dice) is what makes this reliance/faithfulness
    (S3.1) rather than fragility (S3.3, which is vs GT). Returns rows
    ``{case_id, region, modality, fill, delta}`` for :func:`aggregate_reliance`.
    """
    import torch

    from ..constants import CHANNEL_ORDER, REGION_ORDER
    from ..engine import get_device, infer_volume
    from .segmentation import postprocess

    device = device or get_device()
    model = model.to(device).eval()
    rows: list[dict] = []
    with torch.no_grad():
        for case_dir in case_dirs:
            case_id = Path(case_dir).name
            image, _ = _load_case(case_dir, cfg)
            base = postprocess(infer_volume(model, image.unsqueeze(0), cfg, device))[0].cpu()
            for ci, modality in enumerate(CHANNEL_ORDER):
                interv = intervene_tensor(image, ci, fill)
                pred = postprocess(infer_volume(model, interv.unsqueeze(0), cfg, device))[0].cpu()
                for ri, region in enumerate(REGION_ORDER):
                    rows.append({
                        "case_id": case_id,
                        "region": region,
                        "modality": modality,
                        "fill": fill,
                        "delta": 1.0 - dice_score(base[ri], pred[ri]),
                    })
    return rows


# ----------------------------------------------------------------------------- #
# Pure-data aggregation (framework-free; verified now).
# ----------------------------------------------------------------------------- #
def aggregate_reliance(deltas: list[dict], fill: str = "mean") -> list[dict]:
    """Aggregate per-case deltas into the reliance matrix (``RELIANCE_COLUMNS`` rows).

    Input rows: ``{region, modality, delta, ...}``. Output one row per (region,
    modality): mean delta as the reliance ``score`` with a bootstrap 95% CI. Tidy rows
    are written via ``logging_utils.write_tidy`` to ``results/reliance_matrix.*``.
    """
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in deltas:
        grouped[(row["region"], row["modality"])].append(float(row["delta"]))

    out: list[dict] = []
    for (region, modality), vals in sorted(grouped.items()):
        ci_low, ci_high = bootstrap_ci(vals)
        out.append({
            "region": region,
            "modality": modality,
            "fill": fill,
            "score": sum(vals) / len(vals),
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    return out
