"""Evaluate a trained checkpoint and write the paper-ready outputs (roadmap S3).

    python scripts/evaluate.py --config configs/default.yaml --checkpoint runs/<run>/best_model.pt

Produces per-case + aggregate segmentation metrics, the conditional reliance matrix
(S3.1), and comparative missing-modality fragility (S3.3) under ``runs/<eval>/results/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data.splits import load_splits
from brats_trust.engine import get_device
from brats_trust.logging_utils import setup_run
from brats_trust.models.factory import build_model
from brats_trust.pipeline import evaluate_and_log


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a BraTS-Trust checkpoint.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--name", default="eval")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    root = Path(cfg.data.root)
    sp = load_splits(cfg.data.splits_path)
    case_dirs = [root / c for c in sp[args.split]]

    device = get_device()
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))

    ctx = setup_run(args.name, cfg)
    ctx.logger.info("evaluating %d %s cases on %s", len(case_dirs), args.split, device)
    evaluate_and_log(model, case_dirs, cfg, ctx, device=device, physics_key=load_physics_key(cfg))
    ctx.finalize()


if __name__ == "__main__":
    main()
