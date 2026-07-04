"""Train ONLY the rf_large variant with reduced settings to fit in GPU memory.

This is a one-off recovery script. It trains rf_large (kernel 7) with:
  - Patch size 96^3 (down from 128^3) to fit in 24 GiB VRAM
  - 15 epochs (down from 30) for speed
  - 50 training cases (down from 100)
Then appends the result row to outputs/probe3/probe3_summary.jsonl.
"""
import json
import time
from pathlib import Path

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits
from brats_trust.engine import get_device
from brats_trust.experiments import run_single


def main():
    cfg = load_config("configs/sweep.yaml")
    cfg.model.name = "unet3d"
    cfg.model.block = "dwsep"
    cfg.model.kernel_size = 7

    # Reduce patch size to fit in 24 GiB VRAM
    cfg.train.patch_size = [96, 96, 96]
    cfg.inference.roi_size = [96, 96, 96]

    physics_key = load_physics_key(cfg)
    device = get_device()

    root = Path(cfg.data.root)
    splits_path = Path(cfg.data.splits_path)
    sp = (splits.load_splits(splits_path) if splits_path.exists()
          else splits.make_splits(root, cfg.data.split_fractions.to_dict(), cfg.seed, splits_path))

    train_dirs = [root / c for c in sp["train"]][:50]
    val_dirs = [root / c for c in sp["val"]][:12]
    test_dirs = [root / c for c in sp["test"]][:13]
    eval_dirs = val_dirs + test_dirs

    print(f"Training rf_large (kernel 7) with patch 96^3, 15 epochs, {len(train_dirs)} train / {len(eval_dirs)} eval cases")
    start = time.time()

    row = run_single(
        cfg, "probe3_rf_large_seed42",
        train_dirs, val_dirs, eval_dirs, physics_key,
        device=device, base_dir="runs", epochs=15, seed=42,
    )
    row["variant"] = "rf_large"

    elapsed = time.time() - start
    print(f"rf_large finished in {elapsed / 60:.1f} min")

    out_dir = Path("outputs/probe3")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "probe3_summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"Appended rf_large row: {row}")


if __name__ == "__main__":
    main()
