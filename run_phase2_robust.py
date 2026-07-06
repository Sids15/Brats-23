"""Robust execution script for Phase 2 (Anchor Models).
Runs 3 variants x 5 seeds = 15 total runs for 300 epochs.

Safety features implemented:
1. MAX POWER: num_workers=4 and cudnn.benchmark=True.
2. RESUME/CHECKPOINTING: Reads the summary JSONL. If a run finished, it skips it.
3. LIVE LOGGING: Tees all stdout/stderr to a master terminal log file.
4. ERROR ISOLATION: Wraps runs in try/except so one failure doesn't stop the whole sweep.
"""
import json
import sys
import time
import traceback
from pathlib import Path

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits
from brats_trust.engine import get_device
from brats_trust.experiments import run_single

class TeeLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main():
    out_dir = Path("outputs/phase2")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_file = out_dir / "phase2_summary.jsonl"
    log_file = out_dir / "phase2_master_terminal.log"

    # 1. LIVE LOGGING
    sys.stdout = TeeLogger(log_file)
    sys.stderr = sys.stdout  # redirect stderr to the same tee logger
    print("\n" + "="*80)
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] STARTING PHASE 2 ROBUST SWEEP")
    print("="*80)

    # 2. MAX POWER CONFIG
    torch.backends.cudnn.benchmark = True
    device = get_device()

    variants = [
        {"name": "rf_small", "block": "conv", "kernel_size": 3},
        {"name": "rf_med", "block": "dwsep", "kernel_size": 5},
        {"name": "rf_large", "block": "dwsep", "kernel_size": 7},
    ]
    seeds = [42, 43, 44, 45, 46]
    epochs = 30

    # 3. CHECKPOINT / RESUME LOGIC
    completed_runs = set()
    if summary_file.exists():
        with open(summary_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                if data.get("epochs", 0) >= epochs:
                    completed_runs.add(f"{data['variant']}_seed{data['seed']}")
    
    print(f"Found {len(completed_runs)} completed runs in {summary_file}. They will be skipped.")

    for variant in variants:
        for seed in seeds:
            run_id = f"{variant['name']}_seed{seed}"
            if run_id in completed_runs:
                print(f"[SKIP] {run_id} is already completed.")
                continue

            print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] STARTING: {run_id}")
            cfg = load_config("configs/sweep.yaml")
            cfg.model.name = "unet3d"
            cfg.model.block = variant["block"]
            cfg.model.kernel_size = variant["kernel_size"]
            
            # SAFE CONFIG for 24GB VRAM
            cfg.train.patch_size = [96, 96, 96]
            cfg.inference.roi_size = [96, 96, 96]
            cfg.train.num_workers = 4

            physics_key = load_physics_key(cfg)
            root = Path(cfg.data.root)
            splits_path = Path(cfg.data.splits_path)
            sp = (splits.load_splits(splits_path) if splits_path.exists()
                  else splits.make_splits(root, cfg.data.split_fractions.to_dict(), cfg.seed, splits_path))
            
            # Use full splits for Phase 2
            train_dirs = [root / c for c in sp["train"]]
            val_dirs = [root / c for c in sp["val"]]
            test_dirs = [root / c for c in sp["test"]]
            eval_dirs = val_dirs + test_dirs

            # 4. ERROR ISOLATION
            try:
                start = time.time()
                run_name = f"phase2_{run_id}"
                row = run_single(
                    cfg, run_name,
                    train_dirs, val_dirs, eval_dirs, physics_key,
                    device=device, base_dir="runs", epochs=epochs, seed=seed,
                )
                row["variant"] = variant["name"]
                
                # Save immediately
                with open(summary_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                
                elapsed = (time.time() - start) / 60
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS: {run_id} finished in {elapsed:.1f} min")
                
                # Cleanup to prevent OOM across loops
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ERROR IN {run_id}: {str(e)}")
                traceback.print_exc(file=sys.stdout)
                torch.cuda.empty_cache()
                print(f"Continuing to next run...")

    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] PHASE 2 COMPLETE.")

if __name__ == "__main__":
    main()
