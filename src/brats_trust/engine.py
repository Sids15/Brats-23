"""Training and inference engine (roadmap S9 global protocol).

Config-driven so the same code runs the synthetic CPU sanity check and the full
GPU training on the RTX 4500 Ada. AMP is enabled only on CUDA (auto-off on CPU);
inference uses sliding-window + Gaussian blending (S9).
"""
from __future__ import annotations

import sys
import time

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch import nn
from tqdm import tqdm

from .constants import REGION_ORDER
from .logging_utils import log_banner
from .metrics.segmentation import compute_case_metrics, postprocess


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mean_fill_channel(image: torch.Tensor, channel: int) -> torch.Tensor:
    """Torch-native mean-fill of one channel of a ``(C, X, Y, Z)`` image (matches
    ``data.preprocess.intervene``); used in the training-time modality-dropout fix."""
    out = image.clone()
    brain = (image != 0).any(dim=0)
    vals = image[channel][brain]
    fill = vals.mean() if vals.numel() > 0 else image.new_tensor(0.0)
    out[channel] = 0.0
    out[channel][brain] = fill
    return out


def apply_modality_dropout(images: torch.Tensor, prob: float) -> torch.Tensor:
    """Per sample in a batch ``(B, C, X, Y, Z)``, with probability ``prob`` mean-fill one
    random modality (roadmap S6: the *acceptable* robustness fix). Mean-fill, not zero."""
    out = images.clone()
    n_channels = images.shape[1]
    for b in range(images.shape[0]):
        if torch.rand(1).item() < prob:
            out[b] = _mean_fill_channel(images[b], int(torch.randint(0, n_channels, (1,)).item()))
    return out


def build_loss() -> nn.Module:
    # sigmoid=True for overlapping (non-mutually-exclusive) WT/TC/ET channels.
    return DiceCELoss(sigmoid=True)


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)


def infer_volume(model: nn.Module, image: torch.Tensor, cfg, device: torch.device) -> torch.Tensor:
    """Full-volume logits via sliding-window inference. ``image`` is ``(B, 4, X, Y, Z)``."""
    roi = tuple(cfg.inference.roi_size)
    return sliding_window_inference(
        image.to(device),
        roi_size=roi,
        sw_batch_size=cfg.inference.sw_batch_size,
        predictor=model,
        overlap=cfg.inference.overlap,
        mode="gaussian" if cfg.inference.blend == "gaussian" else "constant",
    )


@torch.no_grad()
def validate(model: nn.Module, loader, cfg, device: torch.device) -> dict[str, float]:
    """Per-region and overall validation Dice.

    Returns ``{WT, TC, ET, mean}``; each is NaN-safe (empty-GT regions yield NaN Dice and
    are dropped before averaging). ``mean`` (over all non-empty region/case values) is the
    model-selection signal; the per-region values give the training curves the paper needs.
    """
    model.eval()
    per_region: dict[str, list[float]] = {r: [] for r in REGION_ORDER}
    for batch in loader:
        logits = infer_volume(model, batch["image"], cfg, device)
        preds = postprocess(logits).cpu()
        labels = batch["label"]
        for pred, label in zip(preds, labels):
            for row in compute_case_metrics(pred, label):
                per_region[row["region"]].append(row["dice"])

    out: dict[str, float] = {}
    all_finite: list[float] = []
    for region, vals in per_region.items():
        finite = [d for d in vals if d == d]  # drop NaN (empty regions)
        out[region] = sum(finite) / len(finite) if finite else float("nan")
        all_finite += finite
    out["mean"] = sum(all_finite) / len(all_finite) if all_finite else 0.0
    return out


def train_model(model, train_loader, val_loader, cfg, ctx, device=None, max_epochs=None) -> float:
    """Train the scaffold, logging curves and saving the best checkpoint to the run dir.

    Returns the best validation mean-Dice. ``ctx`` is a ``logging_utils.RunContext``.
    """
    device = device or get_device()
    model = model.to(device)
    loss_fn = build_loss()
    optimizer = build_optimizer(model, cfg)
    use_amp = bool(cfg.train.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    epochs = max_epochs if max_epochs is not None else cfg.train.max_epochs
    steps_per_epoch = max(1, len(train_loader))
    log_every = int(getattr(cfg.train, "log_every_steps", 0) or 0)

    # Modality-dropout fix (roadmap S6): on for the last (1 - start_frac) of training.
    md = getattr(cfg.train, "modality_dropout", None)
    md_enabled = bool(getattr(md, "enabled", False)) if md is not None else False
    md_start = int(getattr(md, "start_frac", 0.8) * epochs) if md_enabled else epochs
    md_prob = float(getattr(md, "prob", 0.25)) if md_enabled else 0.0

    n_params = sum(p.numel() for p in model.parameters())
    log_banner(ctx.logger, "BraTS-Trust training", {
        "Run": ctx.name,
        "Device": device,
        "AMP": "enabled" if use_amp else "disabled",
        "Model": f"{cfg.model.name} / {cfg.model.block} block, kernel {cfg.model.kernel_size}",
        "Params": f"{n_params / 1e6:.2f}M",
        "Epochs": epochs,
        "Steps/epoch": steps_per_epoch,
        "Batch size": cfg.train.batch_size,
        "Patch size": list(cfg.train.patch_size),
        "LR": cfg.train.lr,
        "Val every": f"{cfg.train.val_interval} epochs",
        "Log every": f"{log_every} steps" if log_every else "off",
        "Modality dropout": f"on @ epoch {md_start} (p={md_prob})" if md_enabled else "off",
    })

    best_dice = 0.0
    epoch_times: list[float] = []
    for epoch in range(epochs):
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
        running_loss = 0.0
        bar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}", unit="step",
                   leave=False, file=sys.stdout, dynamic_ncols=True)
        for step, batch in enumerate(bar):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            if md_enabled and epoch >= md_start:
                images = apply_modality_dropout(images, md_prob)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            bar.set_postfix(loss=f"{running_loss / (step + 1):.4f}")
            if log_every and (step + 1) % log_every == 0:
                its = (step + 1) / max(1e-9, time.time() - epoch_start)
                # File-only (no_console): the live bar already shows step/loss/it-s in the
                # terminal; this keeps the granular trace in run.log without fighting the bar.
                ctx.logger.info("    epoch %d/%d  step %d/%d  loss=%.4f  %.2f it/s",
                                epoch + 1, epochs, step + 1, steps_per_epoch,
                                running_loss / (step + 1), its, extra={"no_console": True})
        bar.close()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        train_loss = running_loss / steps_per_epoch
        its = steps_per_epoch / max(1e-9, epoch_time)
        gpu_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
        lr = optimizer.param_groups[0]["lr"]
        eta_h = (sum(epoch_times) / len(epoch_times)) * (epochs - epoch - 1) / 3600.0
        ctx.metrics.log(step=epoch, split="train", loss=train_loss, epoch_time_s=round(epoch_time, 2),
                        it_per_s=round(its, 3), gpu_mem_gb=round(gpu_gb, 3), lr=lr,
                        eta_hours=round(eta_h, 3))
        ctx.logger.info(
            "epoch %d/%d done   loss=%.4f  %.2f it/s  %.0fs  GPU %.2fGiB  lr=%g  ETA %.2fh",
            epoch + 1, epochs, train_loss, its, epoch_time, gpu_gb, lr, eta_h,
        )

        if (epoch + 1) % cfg.train.val_interval == 0 or epoch == epochs - 1:
            dice = validate(model, val_loader, cfg, device)
            improved = dice["mean"] >= best_dice
            ctx.metrics.log(step=epoch, split="val", dice_WT=dice["WT"], dice_TC=dice["TC"],
                            dice_ET=dice["ET"], mean_dice=dice["mean"])
            if improved:
                best_dice = dice["mean"]
                torch.save(model.state_dict(), ctx.run_dir / "best_model.pt")
            ctx.logger.info(
                "epoch %d/%d   val Dice: WT=%.3f  TC=%.3f  ET=%.3f  mean=%.4f  best=%.4f%s",
                epoch + 1, epochs, dice["WT"], dice["TC"], dice["ET"], dice["mean"], best_dice,
                " *saved" if improved else "",
            )

    log_banner(ctx.logger, "Training complete", {
        "Best val Dice": f"{best_dice:.4f}",
        "Epochs run": epochs,
        "Total train time": f"{sum(epoch_times):.0f}s",
    })
    return best_dice
