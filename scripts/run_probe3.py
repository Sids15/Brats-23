"""Probe 3 — the DECISIVE receptive-field sweep (roadmap S4, S4.1).

Trains the shared scaffold across receptive-field variants (the only thing that changes is
the conv block / kernel size, holding the rest of the protocol fixed) over multiple seeds,
then records, per run: effective receptive field (ERF), faithfulness, and val Dice. The
analysis step (scripts/analyze_probe3.py) turns these into the ERF<->faithfulness curve.

Real sweep (GPU):
    python scripts/run_probe3.py --config configs/default.yaml --out outputs/probe3

Smoke test (CPU, synthetic, proves the whole loop wires up; NOT a scientific result):
    python scripts/run_probe3.py --smoke --out outputs/probe3_smoke
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits, synthetic
from brats_trust.data.dataset import make_dataloader
from brats_trust.engine import get_device, train_model
from brats_trust.logging_utils import setup_run, write_tidy
from brats_trust.metrics.erf import measure_erf
from brats_trust.metrics.faithfulness import faithfulness_score
from brats_trust.models.scaffold import build_scaffold
from brats_trust.pipeline import evaluate_and_log

# Receptive-field variants: small -> large. Depthwise-separable keeps params modest as the
# kernel grows, so RF is the dominant thing that changes (roadmap S4 Probe 3).
VARIANTS = [
    {"name": "rf_small", "block": "conv", "kernel_size": 3},
    {"name": "rf_med", "block": "dwsep", "kernel_size": 5},
    {"name": "rf_large", "block": "dwsep", "kernel_size": 7},
]

PROBE3_COLUMNS = (
    "variant", "block", "kernel_size", "seed",
    "erf", "faithfulness_overall", "faith_WT", "faith_TC", "faith_ET", "val_dice",
)


def run_one(cfg, variant, seed, train_dirs, val_dirs, eval_dirs, physics_key, device, base_dir, epochs):
    cfg.model.block = variant["block"]
    cfg.model.kernel_size = variant["kernel_size"]

    train_loader = make_dataloader(train_dirs, cfg, train=True, num_workers=0)
    val_loader = make_dataloader(val_dirs, cfg, train=False, batch_size=1)
    model = build_scaffold(block=cfg.model.block, features=cfg.model.features,
                           kernel_size=cfg.model.kernel_size)

    ctx = setup_run(f"probe3_{variant['name']}_seed{seed}", cfg, base_dir=base_dir,
                    set_global_seed=seed)
    val_dice = train_model(model, train_loader, val_loader, cfg, ctx, device=device, max_epochs=epochs)
    out = evaluate_and_log(model, eval_dirs, cfg, ctx, device=device, physics_key=physics_key)
    erf = measure_erf(model, tuple(cfg.train.patch_size), out_channel=2, device=device)  # probe ET
    faith = faithfulness_score(out["reliance_matrix"], physics_key)
    ctx.finalize(erf=erf, faithfulness=faith, val_dice=val_dice)
    return {
        "variant": variant["name"], "block": variant["block"], "kernel_size": variant["kernel_size"],
        "seed": seed, "erf": erf, "faithfulness_overall": faith["overall"],
        "faith_WT": faith["WT"], "faith_TC": faith["TC"], "faith_ET": faith["ET"],
        "val_dice": val_dice,
    }


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
        base = Path("runs")

    train_dirs = [root / c for c in sp["train"]]
    val_dirs = [root / c for c in sp["val"]]
    eval_dirs = [root / c for c in (sp["val"] + sp["test"])]

    rows = []
    for variant in VARIANTS:
        for seed in seeds:
            rows.append(run_one(cfg, variant, seed, train_dirs, val_dirs, eval_dirs,
                                physics_key, device, base, epochs))

    out_dir = Path(args.out)
    write_tidy(out_dir / "probe3_summary", rows, PROBE3_COLUMNS)
    print(f"wrote {len(rows)} runs -> {out_dir / 'probe3_summary.csv'}")
    print("next: python scripts/analyze_probe3.py --summary", out_dir / "probe3_summary.jsonl")


if __name__ == "__main__":
    main()
