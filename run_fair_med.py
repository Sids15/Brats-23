"""Train rf_med fairly."""
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
    summary_file = out_dir / "probe3_fair_summary_med.jsonl"
    
    cfg = load_config("configs/sweep.yaml")
    cfg.model.name = "unet3d"
    cfg.model.block = "dwsep"
    cfg.model.kernel_size = 5
    
    cfg.train.patch_size = [96, 96, 96]
    cfg.inference.roi_size = [96, 96, 96]
    # Maximize CPU usage for data loading to keep GPU fed
    cfg.train.num_workers = 4

    physics_key = load_physics_key(cfg)

    root = Path(cfg.data.root)
    splits_path = Path(cfg.data.splits_path)
    sp = (splits.load_splits(splits_path) if splits_path.exists()
          else splits.make_splits(root, cfg.data.split_fractions.to_dict(), cfg.seed, splits_path))

    train_dirs = [root / c for c in sp["train"]][:50]
    val_dirs = [root / c for c in sp["val"]][:12]
    test_dirs = [root / c for c in sp["test"]][:13]
    eval_dirs = val_dirs + test_dirs

    print(f"Training rf_med (kernel 5) - PARALLEL MODE")
    start = time.time()
    row = run_single(
        cfg, "probe3_fair_rf_med_seed42",
        train_dirs, val_dirs, eval_dirs, physics_key,
        device=device, base_dir="runs", epochs=15, seed=42,
    )
    row["variant"] = "rf_med"

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"rf_med finished in {(time.time() - start) / 60:.1f} min")

if __name__ == "__main__":
    main()
