"""Shared single-run experiment driver (used by the Probe sweeps).

One model run = train -> evaluate (all tidy outputs) -> measure ERF + faithfulness. Both
the RF sweep (Probe 3) and the architecture sweep (Probe 1 / Tier-A anchors) call this so
the per-run logic and recorded columns are identical.
"""
from __future__ import annotations

import torch

from .data.dataset import make_dataloader
from .engine import get_device, train_model
from .logging_utils import setup_run
from .metrics.erf import measure_erf
from .metrics.faithfulness import faithfulness_score
from .models.factory import build_model
from .pipeline import evaluate_and_log

# Output channel to probe for ERF: 2 = ET (the most modality-specific region).
_ERF_CHANNEL = 2


def run_single(cfg, run_name, train_dirs, val_dirs, eval_dirs, physics_key,
               device=None, base_dir="runs", epochs=None, seed=0) -> dict:
    """Train + evaluate one model (per ``cfg``); return a summary row of key metrics.

    Matched-protocol invariant (why cross-architecture comparisons are fair, roadmap S5):
    the caller passes the SAME ``train/val/eval_dirs`` (one leakage-free split) and the SAME
    ``cfg`` (preprocessing, patch, optimizer, schedule, loss) to every architecture; only
    ``cfg.model.name`` changes. Crucially the seed is set *after* ``build_model`` but *before*
    the data is iterated, so architecture construction can't perturb the data RNG -- every
    model at a given seed therefore sees the *identical patch sequence*. Do not reorder these
    two lines: model first, then ``setup_run(set_global_seed=...)``.
    """
    device = device or get_device()
    train_loader = make_dataloader(train_dirs, cfg, train=True, num_workers=0)
    val_loader = make_dataloader(val_dirs, cfg, train=False, batch_size=1)
    model = build_model(cfg)

    ctx = setup_run(run_name, cfg, base_dir=base_dir, set_global_seed=seed)
    val_dice = train_model(model, train_loader, val_loader, cfg, ctx, device=device, max_epochs=epochs)
    out = evaluate_and_log(model, eval_dirs, cfg, ctx, device=device, physics_key=physics_key)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    erf = measure_erf(model, tuple(cfg.train.patch_size), out_channel=_ERF_CHANNEL, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    faith = faithfulness_score(out["reliance_matrix"], physics_key)
    ctx.finalize(erf=erf, faithfulness=faith, val_dice=val_dice)
    
    ctx.logger.info("Run finished: ERF %.2f | Faithfulness %.3f | Val Dice %.4f", erf, faith["overall"], val_dice)

    return {
        "model": cfg.model.name, "block": cfg.model.block, "kernel_size": cfg.model.kernel_size,
        "seed": seed, "erf": erf, "faithfulness_overall": faith["overall"],
        "faith_WT": faith["WT"], "faith_TC": faith["TC"], "faith_ET": faith["ET"],
        "val_dice": val_dice,
    }
