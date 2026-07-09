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
from .models import model_cost
from .models.factory import build_model
from .pipeline import evaluate_and_log

# Output channel to probe for ERF: 2 = ET (the most modality-specific region).
_ERF_CHANNEL = 2

# Columns every sweep summary carries, in order. Probe scripts prepend their own key
# (`variant` for Probe 3) and pass this to `write_tidy` so the CSVs line up across sweeps.
SUMMARY_COLUMNS = (
    "model", "block", "kernel_size", "seed", "epochs", "batch_size", "num_workers",
    "params", "flops", "erf", "faithfulness_overall",
    "faith_WT", "faith_TC", "faith_ET", "val_dice",
)


def run_single(cfg, run_name, train_dirs, val_dirs, eval_dirs, physics_key,
               device=None, base_dir="runs", epochs=None, seed=0, num_workers=0) -> dict:
    """Train + evaluate one model (per ``cfg``); return a summary row of key metrics.

    Matched-protocol invariant (why cross-architecture comparisons are fair, roadmap S5):
    the caller passes the SAME ``train/val/eval_dirs`` (one leakage-free split) and the SAME
    ``cfg`` (preprocessing, patch, batch size, optimizer, schedule, loss) to every
    architecture; only ``cfg.model.name`` changes. Crucially the seed is set *after*
    ``build_model`` but *before* the data is iterated, so architecture construction can't
    perturb the data RNG -- every model at a given seed therefore sees the *identical patch
    sequence*. Do not reorder these two lines: model first, then
    ``setup_run(set_global_seed=...)``.

    ``num_workers`` is an explicit argument, NOT read from ``cfg.train.num_workers``, and
    defaults to 0. MONAI re-seeds its random transforms per worker, so the worker count is
    part of the augmentation stream: runs executed at different ``num_workers`` are not
    comparable under the invariant above. The default preserves the stream of every sweep
    already on disk; a new sweep may raise it, provided it does so for all of its runs.
    """
    device = device or get_device()
    num_workers = int(num_workers or 0)
    # Resolve now, so the summary row records the epochs actually run rather than a null
    # when the caller left the count to the config (sweep resume keys off this number).
    epochs = int(epochs if epochs is not None else cfg.train.max_epochs)
    train_loader = make_dataloader(train_dirs, cfg, train=True, num_workers=num_workers)
    val_loader = make_dataloader(val_dirs, cfg, train=False, batch_size=1)
    model = build_model(cfg)

    ctx = setup_run(run_name, cfg, base_dir=base_dir, set_global_seed=seed, resume=True)
    val_dice = train_model(model, train_loader, val_loader, cfg, ctx, device=device, max_epochs=epochs)
    cost = model_cost(model, cfg.train.patch_size)
    out = evaluate_and_log(model, eval_dirs, cfg, ctx, device=device, physics_key=physics_key)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    erf = measure_erf(model, tuple(cfg.train.patch_size), out_channel=_ERF_CHANNEL, device=device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    faith = faithfulness_score(out["reliance_matrix"], physics_key)
    ctx.finalize(erf=erf, faithfulness=faith, val_dice=val_dice, **cost)

    ctx.logger.info("Run finished: ERF %.2f | Faithfulness %.3f | Val Dice %.4f", erf, faith["overall"], val_dice)

    return {
        "model": cfg.model.name, "block": cfg.model.block, "kernel_size": cfg.model.kernel_size,
        "seed": seed, "epochs": epochs, "batch_size": cfg.train.batch_size,
        "num_workers": num_workers, "params": cost["params"], "flops": cost["flops"],
        "erf": erf, "faithfulness_overall": faith["overall"],
        "faith_WT": faith["WT"], "faith_TC": faith["TC"], "faith_ET": faith["ET"],
        "val_dice": val_dice,
    }
