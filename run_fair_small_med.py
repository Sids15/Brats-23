"""Train rf_small and rf_med with the exact same restricted settings as rf_large to ensure a mathematically fair comparison."""
import json
import time
from pathlib import Path

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits
from brats_trust.engine import get_device
from brats_trust.experiments import run_single


def main():
    device = get_device()
    out_dir = Path("outputs/probe3_fair")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "probe3_fair_summary.jsonl"
    
    # We clear the file if it exists to avoid duplicate appends on re-runs
    if summary_file.exists():
        summary_file.unlink()
        
    variants = [
        {"name": "rf_small", "block": "conv", "kernel_size": 3},
        {"name": "rf_med", "block": "dwsep", "kernel_size": 5},
    ]

    for variant in variants:
        cfg = load_config("configs/sweep.yaml")
        cfg.model.name = "unet3d"
        cfg.model.block = variant["block"]
        cfg.model.kernel_size = variant["kernel_size"]
        
        # EXACT restricted settings used for rf_large
        cfg.train.patch_size = [96, 96, 96]
        cfg.inference.roi_size = [96, 96, 96]
        
        # MAX POWER SETTINGS: Use multi-core CPU loading and cuDNN benchmarking
        # to feed the GPU at 100% capacity without altering the math.
        cfg.train.num_workers = 4
        import torch
        torch.backends.cudnn.benchmark = True

        physics_key = load_physics_key(cfg)

        root = Path(cfg.data.root)
        splits_path = Path(cfg.data.splits_path)
        sp = (splits.load_splits(splits_path) if splits_path.exists()
              else splits.make_splits(root, cfg.data.split_fractions.to_dict(), cfg.seed, splits_path))

        train_dirs = [root / c for c in sp["train"]][:50]
        val_dirs = [root / c for c in sp["val"]][:12]
        test_dirs = [root / c for c in sp["test"]][:13]
        eval_dirs = val_dirs + test_dirs

        print(f"\n============================================================")
        print(f"--- Training {variant['name']} (kernel {variant['kernel_size']}) ---")
        print(f"Patch 96^3, 15 epochs, {len(train_dirs)} train / {len(eval_dirs)} eval cases")
        start = time.time()

        run_name = f"probe3_fair_{variant['name']}_seed42"
        row = run_single(
            cfg, run_name,
            train_dirs, val_dirs, eval_dirs, physics_key,
            device=device, base_dir="runs", epochs=15, seed=42,
        )
        row["variant"] = variant["name"]

        elapsed = time.time() - start
        print(f"{variant['name']} finished in {elapsed / 60:.1f} min")

        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"Appended {variant['name']} row: {row}")

if __name__ == "__main__":
    main()
