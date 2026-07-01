"""Probe 1 — mechanism swap + Tier-A anchors (roadmap S4 Probe 1, S5).

Runs the SAME training/measurement pipeline across different architectures (CNN vs
transformer vs Mamba) under the matched protocol, recording faithfulness / ERF / Dice per
architecture x seed. This serves both Probe 1 (mechanism comparison) and the Tier-A
off-the-shelf anchors (S5). CAVEAT (roadmap S4): swapping architecture changes RF +
optimization + inductive bias at once -- report as 'contribution under matched protocol',
never 'causes'.

    python scripts/run_probe1.py --config configs/default.yaml --out outputs/probe1
    python scripts/run_probe1.py --smoke --models unet3d dynunet --out outputs/probe1_smoke

`segmamba` needs `pip install mamba-ssm causal-conv1d` (CUDA) -> GPU only.
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

# Default architecture set: CNN (our scaffold + nnU-Net), transformer (UNETR/Swin), Mamba.
DEFAULT_MODELS = ["unet3d", "dynunet", "unetr", "swin_unetr", "segmamba"]

PROBE1_COLUMNS = (
    "model", "block", "kernel_size", "seed",
    "erf", "faithfulness_overall", "faith_WT", "faith_TC", "faith_ET", "val_dice",
)


def setup_smoke(cfg):
    # Swin needs >=64^3; keep the smoke volume at 64 so every architecture is exercisable.
    cfg.train.patch_size = [64, 64, 64]
    cfg.inference.roi_size = [64, 64, 64]
    cfg.inference.sw_batch_size = 1
    cfg.model.features = [16, 32]
    cfg.train.batch_size = 1
    cfg.train.lr = 0.001
    cfg.train.val_interval = 10
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "data"
    synthetic.generate_dataset(root, n_cases=8, shape=(64, 64, 64),
                               class_channels={3: 2, 2: 0}, seed=0)
    sp = splits.make_splits(root, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=0)
    return cfg, root, sp, tmp


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Probe 1 / Tier-A architecture sweep.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/probe1")
    ap.add_argument("--models", nargs="*", default=None, help=f"Subset of {DEFAULT_MODELS}")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    physics_key = load_physics_key(cfg)
    device = torch.device("cpu") if args.smoke else get_device()
    models = args.models or DEFAULT_MODELS

    if args.smoke:
        cfg, root, sp, base = setup_smoke(cfg)
        seeds = args.seeds or [0]
        epochs = args.epochs if args.epochs is not None else 20
    else:
        root = Path(cfg.data.root)
        splits_path = Path(cfg.data.splits_path)
        sp = (splits.load_splits(splits_path) if splits_path.exists()
              else splits.make_splits(root, cfg.data.split_fractions.to_dict(), cfg.seed, splits_path))
        seeds = args.seeds or cfg.seeds_exploration  # >=3 for non-confirmatory probes (S4.2)
        epochs = args.epochs
        base = "runs"

    train_dirs = [root / c for c in sp["train"]]
    val_dirs = [root / c for c in sp["val"]]
    eval_dirs = [root / c for c in (sp["val"] + sp["test"])]

    try:
        rows = []
        for name in models:
            cfg.model.name = name
            for seed in seeds:
                try:
                    rows.append(run_single(cfg, f"probe1_{name}_seed{seed}", train_dirs, val_dirs,
                                           eval_dirs, physics_key, device=device, base_dir=base,
                                           epochs=epochs, seed=seed))
                except ImportError as e:  # e.g. segmamba without mamba-ssm
                    print(f"SKIP {name}: {e}")

        out_dir = Path(args.out)
        write_tidy(out_dir / "probe1_summary", rows, PROBE1_COLUMNS)
        print(f"wrote {len(rows)} runs -> {out_dir / 'probe1_summary.csv'}")
    finally:
        # Smoke data + intermediate run dirs live under a temp base; results are in --out.
        if args.smoke:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
