"""Manually evaluate and save the 30-epoch model to the summary file."""
import json
import time
from pathlib import Path
import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits
from brats_trust.engine import get_device
from brats_trust.experiments import _ERF_CHANNEL
from brats_trust.logging_utils import setup_run
from brats_trust.metrics.erf import measure_erf
from brats_trust.metrics.faithfulness import faithfulness_score
from brats_trust.models.factory import build_model
from brats_trust.pipeline import evaluate_and_log

def main():
    device = get_device()
    cfg = load_config("configs/sweep.yaml")
    cfg.model.name = "unet3d"
    cfg.model.block = "conv"
    cfg.model.kernel_size = 3
    cfg.train.patch_size = [96, 96, 96]
    cfg.inference.roi_size = [96, 96, 96]
    cfg.train.num_workers = 4

    run_dir = Path("runs/phase2_rf_small_seed42")
    ckpt_path = run_dir / "latest_checkpoint.pt"
    
    if not ckpt_path.exists():
        print("Checkpoint not found!")
        return

    print(f"Loading checkpoint from {ckpt_path}")
    model = build_model(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    
    physics_key = load_physics_key(cfg)
    root = Path(cfg.data.root)
    splits_path = Path(cfg.data.splits_path)
    sp = splits.load_splits(splits_path)
    eval_dirs = [root / c for c in sp["val"] + sp["test"]]

    ctx = setup_run("phase2_rf_small_seed42", cfg, base_dir="runs", resume=True)
    
    print("Evaluating 30-epoch model on full dataset...")
    out = evaluate_and_log(model, eval_dirs, cfg, ctx, device=device, physics_key=physics_key)
    
    print("Measuring ERF...")
    torch.cuda.empty_cache()
    erf = measure_erf(model, tuple(cfg.train.patch_size), out_channel=_ERF_CHANNEL, device=device)
    
    torch.cuda.empty_cache()
    faith = faithfulness_score(out["reliance_matrix"], physics_key)
    
    result = {
        "variant": "rf_small",
        "model": cfg.model.name, "block": cfg.model.block, "kernel_size": cfg.model.kernel_size,
        "seed": 42, "epochs": ckpt["epoch"], "erf": erf, "faithfulness_overall": faith["overall"],
        "faith_WT": faith["WT"], "faith_TC": faith["TC"], "faith_ET": faith["ET"],
    }
    
    summary_file = Path("outputs/phase2/phase2_summary.jsonl")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    
    print(f"Saved 30-epoch model to {summary_file}. You can now continue with the sweep!")

if __name__ == "__main__":
    main()
