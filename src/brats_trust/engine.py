"""Training and inference engine (roadmap S9 global protocol).

Config-driven so the same code runs the synthetic CPU sanity check and the full
GPU training on the RTX 4500 Ada. AMP is enabled only on CUDA (auto-off on CPU);
inference uses sliding-window + Gaussian blending (S9).
"""
from __future__ import annotations

import torch
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from torch import nn
from tqdm import tqdm

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
def validate(model: nn.Module, loader, cfg, device: torch.device) -> float:
    """Return mean Dice over all regions/cases (for model selection)."""
    model.eval()
    dices: list[float] = []
    for batch in loader:
        logits = infer_volume(model, batch["image"], cfg, device)
        preds = postprocess(logits).cpu()
        labels = batch["label"]
        for pred, label in zip(preds, labels):
            dices += [r["dice"] for r in compute_case_metrics(pred, label)]
    valid = [d for d in dices if d == d]  # drop NaN (empty regions)
    return float(sum(valid) / len(valid)) if valid else 0.0


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

    # Modality-dropout fix (roadmap S6): on for the last (1 - start_frac) of training.
    md = getattr(cfg.train, "modality_dropout", None)
    md_enabled = bool(getattr(md, "enabled", False)) if md is not None else False
    md_start = int(getattr(md, "start_frac", 0.8) * epochs) if md_enabled else epochs
    md_prob = float(getattr(md, "prob", 0.25)) if md_enabled else 0.0

    best_dice = 0.0
    for epoch in tqdm(range(epochs), desc="train", unit="epoch"):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
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
            epoch_loss += loss.item()
        ctx.metrics.log(step=epoch, split="train", loss=epoch_loss / max(1, len(train_loader)))

        if (epoch + 1) % cfg.train.val_interval == 0 or epoch == epochs - 1:
            dice = validate(model, val_loader, cfg, device)
            ctx.metrics.log(step=epoch, split="val", mean_dice=dice)
            ctx.logger.info("epoch %d | val mean Dice %.4f", epoch, dice)
            if dice >= best_dice:
                best_dice = dice
                torch.save(model.state_dict(), ctx.run_dir / "best_model.pt")
    return best_dice
