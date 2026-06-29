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

from .stats import bootstrap_ci

# ----------------------------------------------------------------------------- #
# Model-dependent intervention loop (needs torch; lands on the training machine).
# ----------------------------------------------------------------------------- #
def collect_reliance_deltas(model, dataloader, fill: str = "mean") -> list[dict]:
    """Run the conditional intervention and return per-case reliance deltas.

    For each case, region (ET/TC/WT) and modality, measure the prediction change when
    that modality is intervened on (``fill``) *while the others stay present* -- the
    conditional structure that distinguishes unique reliance from correlated inference
    (roadmap S3.1). Returns rows ``{case_id, region, modality, fill, delta}`` that feed
    :func:`aggregate_reliance`.

    STUB (Stage 0/2): requires the torch model + ablation dataloader. The aggregation
    side below is implemented and unit-tested now so the metric is validated end-to-end
    on the synthetic check the moment the model exists.
    """
    raise NotImplementedError("Stage 0/2: torch intervention loop (use data.preprocess.intervene)")


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
