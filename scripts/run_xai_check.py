"""XAI-fails check + MSFI cross-check (roadmap S3.4, S3.2).

Loads a trained checkpoint and, on a few cases, shows that a saliency map barely changes
when T1CE is mean-filled -- i.e. saliency is blind to the modality-level shortcut that the
intervention metric (S3.1) catches. Also reports MSFI (saliency share on the physics
modality) as a convergent-validity check.

    python scripts/run_xai_check.py --config configs/default.yaml --checkpoint runs/<run>/best_model.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data.splits import load_splits
from brats_trust.engine import get_device
from brats_trust.logging_utils import setup_run, write_tidy
from brats_trust.metrics.reliance import _load_case
from brats_trust.metrics.xai import msfi_score, xai_fails_check
from brats_trust.models.factory import build_model


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the XAI-fails + MSFI check.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=5, help="Number of cases to check.")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    physics_key = load_physics_key(cfg)
    device = get_device()
    model = build_model(cfg)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)

    root = Path(cfg.data.root)
    sp = load_splits(cfg.data.splits_path)
    case_dirs = [root / c for c in sp[args.split][: args.n]]

    ctx = setup_run("xai_check", cfg)
    xai_rows, msfi_rows = [], []
    for case_dir in case_dirs:
        image, _ = _load_case(case_dir, cfg)
        xai_rows.append({"case_id": case_dir.name, **xai_fails_check(model, image, device=device)})
        msfi = msfi_score(model, image, physics_key, device=device)
        msfi_rows.append({"case_id": case_dir.name, **msfi})

    write_tidy(ctx.result("xai_fails"), xai_rows)
    write_tidy(ctx.result("msfi"), msfi_rows)
    mean_cos = sum(r["saliency_cosine"] for r in xai_rows) / len(xai_rows)
    ctx.logger.info("mean saliency cosine after T1CE mean-fill: %.3f (high => saliency is blind)", mean_cos)
    print(json.dumps({"mean_saliency_cosine": mean_cos, "n_cases": len(xai_rows)}, indent=2))
    ctx.finalize()


if __name__ == "__main__":
    main()
