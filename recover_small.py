import csv
import json
from pathlib import Path

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.engine import get_device
from brats_trust.experiments import _ERF_CHANNEL
from brats_trust.metrics.erf import measure_erf
from brats_trust.metrics.faithfulness import faithfulness_score
from brats_trust.models.factory import build_model

def main():
    run_dir = Path("runs/20260704T060142Z__probe3_rf_small_seed42")
    if not run_dir.exists():
        run_dir = Path("runs/20260704T055712Z__probe3_rf_small_seed42")
        
    cfg = load_config("configs/sweep.yaml")
    cfg.model.name = "unet3d"
    cfg.model.block = "conv"
    cfg.model.kernel_size = 3
    physics_key = load_physics_key(cfg)
    device = get_device()

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device))
    erf = measure_erf(model, tuple(cfg.train.patch_size), out_channel=_ERF_CHANNEL, device=device)
    
    reliance_matrix = []
    with open(run_dir / "results" / "reliance_matrix.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["score"] = float(row["score"])
            reliance_matrix.append(row)
            
    faith = faithfulness_score(reliance_matrix, physics_key)
    
    val_dice = 0.0
    with open(run_dir / "results" / "aggregate_metrics.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        vals = [float(row["mean"]) for row in reader if row["metric"] == "dice"]
        val_dice = sum(vals) / len(vals) if vals else 0.0

    row = {
        "variant": "rf_small", "model": "unet3d", "block": "conv", "kernel_size": 3,
        "seed": 42, "erf": erf, "faithfulness_overall": faith["overall"],
        "faith_WT": faith["WT"], "faith_TC": faith["TC"], "faith_ET": faith["ET"],
        "val_dice": val_dice,
    }
    
    out_dir = Path("outputs/probe3")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "probe3_summary.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(f"Recovered rf_small row: {row}")

if __name__ == "__main__":
    main()
