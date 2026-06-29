"""Stage 6 — evaluate the modality-dropout fix (roadmap S6).

Trains two models under the identical protocol/seed -- baseline vs modality-dropout -- and
compares faithfulness and Dice. The roadmap's evidence bar: faithfulness IMPROVES while
full-modality Dice HOLDS (and, via aggregate.py on the two run sets, the fragility gap
shrinks). We MEASURE this; we do not assume the fix 'forces faithful representations'.

    python scripts/run_fix.py --config configs/default.yaml --out outputs/fix
    python scripts/run_fix.py --smoke --out outputs/fix_smoke
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits, synthetic
from brats_trust.engine import get_device
from brats_trust.experiments import run_single
from brats_trust.logging_utils import write_json, write_tidy


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
    ap = argparse.ArgumentParser(description="Train baseline vs modality-dropout fix and compare.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="outputs/fix")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    cfg.model.name = "unet3d"
    physics_key = load_physics_key(cfg)
    device = torch.device("cpu") if args.smoke else get_device()

    if args.smoke:
        cfg, root, sp, base = setup_smoke(cfg)
        epochs = args.epochs if args.epochs is not None else 30
    else:
        root = Path(cfg.data.root)
        splits_path = Path(cfg.data.splits_path)
        sp = (splits.load_splits(splits_path) if splits_path.exists()
              else splits.make_splits(root, cfg.data.split_fractions.to_dict(), cfg.seed, splits_path))
        epochs = args.epochs
        base = "runs"

    train_dirs = [root / c for c in sp["train"]]
    val_dirs = [root / c for c in sp["val"]]
    eval_dirs = [root / c for c in (sp["val"] + sp["test"])]

    def go(label):
        return run_single(cfg, f"fix_{label}_seed{args.seed}", train_dirs, val_dirs, eval_dirs,
                          physics_key, device=device, base_dir=base, epochs=epochs, seed=args.seed)

    cfg.train.modality_dropout.enabled = False
    baseline = go("baseline")
    cfg.train.modality_dropout.enabled = True
    fixed = go("dropout")

    comparison = {
        "baseline": baseline, "fixed": fixed,
        "delta_faithfulness": fixed["faithfulness_overall"] - baseline["faithfulness_overall"],
        "delta_val_dice": fixed["val_dice"] - baseline["val_dice"],
    }
    out_dir = Path(args.out)
    write_tidy(out_dir / "fix_runs", [baseline, fixed])
    write_json(out_dir / "fix_comparison.json", comparison)
    print(json.dumps({k: comparison[k] for k in ("delta_faithfulness", "delta_val_dice")}, indent=2))
    print("Evidence (S6): want delta_faithfulness > 0 with delta_val_dice ~>= 0; "
          "also check the fragility gap via scripts/aggregate.py on the two run sets.")


if __name__ == "__main__":
    main()
