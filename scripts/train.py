"""Train the shared U-Net scaffold on BraTS-2023 (roadmap S9, S4).

    python scripts/train.py --config configs/default.yaml --name baseline_seed42

Builds (or loads) patient-level splits, trains with the frozen protocol, logs curves and
the best checkpoint into a timestamped run directory. Designed to run unchanged on the
RTX 4500 Ada (CUDA + AMP) and on CPU (synthetic/debug).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from brats_trust.config import load_config
from brats_trust.data.dataset import make_dataloader
from brats_trust.data.splits import load_splits, make_splits
from brats_trust.engine import get_device, train_model
from brats_trust.logging_utils import setup_run
from brats_trust.models.factory import build_model


def resolve_splits(cfg):
    root = Path(cfg.data.root)
    splits_path = Path(cfg.data.splits_path)
    if splits_path.exists():
        sp = load_splits(splits_path)
    else:
        sp = make_splits(root, cfg.data.split_fractions.to_dict(), cfg.seed, splits_path)
    return root, sp


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the BraTS-Trust scaffold.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="train")
    ap.add_argument("--epochs", type=int, default=None, help="Override max_epochs.")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    root, sp = resolve_splits(cfg)
    train_dirs = [root / c for c in sp["train"]]
    val_dirs = [root / c for c in sp["val"]]

    train_loader = make_dataloader(train_dirs, cfg, train=True, num_workers=cfg.train.num_workers)
    val_loader = make_dataloader(val_dirs, cfg, train=False, batch_size=1)

    model = build_model(cfg)
    ctx = setup_run(args.name, cfg, set_global_seed=cfg.seed)
    ctx.logger.info("train=%d val=%d cases | device=%s", len(train_dirs), len(val_dirs), get_device())
    best = train_model(model, train_loader, val_loader, cfg, ctx, max_epochs=args.epochs)
    ctx.finalize(best_val_dice=best)


if __name__ == "__main__":
    main()
