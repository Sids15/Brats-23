"""Probe 3 — the DECISIVE receptive-field sweep (roadmap S4, S4.1).

Trains the shared scaffold across receptive-field variants (only the conv block / kernel
size changes; the rest of the protocol is fixed) over multiple seeds, recording per run:
effective receptive field (ERF), faithfulness, val Dice. analyze_probe3.py turns these into
the ERF<->faithfulness curve.

Real sweep (GPU):
    python scripts/run_probe3.py --config configs/default.yaml --out outputs/probe3
Smoke test (CPU, synthetic; proves the loop wires up, NOT a scientific result):
    python scripts/run_probe3.py --smoke --out outputs/probe3_smoke
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits, synthetic
from brats_trust.engine import get_device
from brats_trust.experiments import run_single
from brats_trust.logging_utils import write_tidy

# RF variants: small -> large. Depthwise-separable keeps params modest as the kernel grows,
# so receptive field is the dominant thing that changes (roadmap S4 Probe 3).
VARIANTS = [
    {"name": "rf_small", "block": "conv", "kernel_size": 3},
    {"name": "rf_med", "block": "dwsep", "kernel_size": 5},
    {"name": "rf_large", "block": "dwsep", "kernel_size": 7},
]

PROBE3_COLUMNS = (
    "variant", "model", "block", "kernel_size", "seed",
    "erf", "faithfulness_overall", "faith_WT", "faith_TC", "faith_ET", "val_dice",
)


def setup_smoke(cfg):
    cfg.train.patch_size = [24, 24, 24]
    cfg.inference.roi_size = [24, 24, 24]
    cfg.inference.sw_batch_size = 1
    cfg.model.features = [16, 32]
    cfg.train.batch_size = 2
    cfg.train.lr = 0.001
    cfg.train.val_interval = 10
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "data"
    synthetic.generate_dataset(root, n_cases=12, shape=(32, 32, 32),
                               class_channels={3: 2, 2: 0}, seed=0)
    sp = splits.make_splits(root, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
    return cfg, root, sp, tmp


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Probe 3 receptive-field sweep.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/probe3")
    ap.add_argument("--smoke", action="store_true", help="Tiny CPU run on synthetic data.")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    cfg.model.name = "unet3d"  # the RF sweep always uses our pluggable scaffold
    physics_key = load_physics_key(cfg)
    device = torch.device("cpu") if args.smoke else get_device()

    if args.smoke:
        cfg, root, sp, base = setup_smoke(cfg)
        seeds = args.seeds or [0]
        epochs = args.epochs if args.epochs is not None else 30
    else:
        root = Path(cfg.data.root)
        splits_path = Path(cfg.data.splits_path)
        sp = (splits.load_splits(splits_path) if splits_path.exists()
              else splits.make_splits(root, cfg.data.split_fractions.to_dict(), cfg.seed, splits_path))
        seeds = args.seeds or cfg.seeds_confirmatory  # >=5 confirmatory (S4.2)
        epochs = args.epochs
        base = "runs"

    train_dirs = [root / c for c in sp["train"]]
    val_dirs = [root / c for c in sp["val"]]
    eval_dirs = [root / c for c in (sp["val"] + sp["test"])]

    try:
        rows = []
        for variant in VARIANTS:
            cfg.model.block = variant["block"]
            cfg.model.kernel_size = variant["kernel_size"]
            for seed in seeds:
                row = run_single(cfg, f"probe3_{variant['name']}_seed{seed}", train_dirs, val_dirs,
                                 eval_dirs, physics_key, device=device, base_dir=base, epochs=epochs, seed=seed)
                rows.append({"variant": variant["name"], **row})

        out_dir = Path(args.out)
        write_tidy(out_dir / "probe3_summary", rows, PROBE3_COLUMNS)
        print(f"wrote {len(rows)} runs -> {out_dir / 'probe3_summary.csv'}")
        print("next: python scripts/analyze_probe3.py --summary", out_dir / "probe3_summary.jsonl")
    finally:
        # Smoke data + intermediate run dirs live under a temp base; results are in --out.
        if args.smoke:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
