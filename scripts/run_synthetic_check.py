"""Synthetic sanity check (roadmap S3.5): does the reliance metric flag the modality we
*planted* the signal in?

We generate cases where the enhancing-tumor signal (label 3 -> ET/TC) lives only in the
T1CE channel, train the small scaffold, then run the full evaluation. The check passes if
the conditional reliance for ET (and TC) is highest on T1CE -- i.e. the metric recovers
the known ground-truth reliance. Scope (S3.5): this proves the metric isn't broken, NOT
that real-MRI reliance is correct, and is never a headline figure.

    python scripts/run_synthetic_check.py --epochs 40

Runs end-to-end on CPU; no real dataset required.
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits, synthetic
from brats_trust.data.dataset import make_dataloader
from brats_trust.engine import train_model
from brats_trust.logging_utils import setup_run
from brats_trust.models.unet3d import build_scaffold
from brats_trust.pipeline import evaluate_and_log

# Plant ET (label 3) in T1CE (channel 2) and edema (label 2) in FLAIR (channel 0).
PLANTED = {3: 2, 2: 0}
DESIGNATED_REGION = "ET"
DESIGNATED_MODALITY = "T1CE"


def build_tiny_cfg():
    cfg = load_config()
    # Train on the full (small) volume so every sample contains the planted spheres,
    # and use a higher LR than the full-scale protocol so it converges in CPU time.
    cfg.train.patch_size = [32, 32, 32]
    cfg.inference.roi_size = [32, 32, 32]
    cfg.inference.sw_batch_size = 1
    cfg.model.features = [16, 32]
    cfg.train.batch_size = 2
    cfg.train.lr = 0.001
    cfg.train.val_interval = 10
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the S3.5 synthetic reliance sanity check.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--cases", type=int, default=16)
    ap.add_argument("--workdir", default=None, help="Defaults to a temp dir.")
    args = ap.parse_args()

    cfg = build_tiny_cfg()
    device = torch.device("cpu")
    created_tmp = args.workdir is None
    tmp = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp())
    try:
        data_root = tmp / "synthetic_data"
        synthetic.generate_dataset(
            data_root, n_cases=args.cases, shape=(32, 32, 32), class_channels=PLANTED, seed=0
        )
        sp = splits.make_splits(data_root, {"train": 0.6, "val": 0.2, "test": 0.2}, seed=0)
        train_dirs = [data_root / c for c in sp["train"]]
        eval_dirs = [data_root / c for c in (sp["val"] + sp["test"])]

        train_loader = make_dataloader(train_dirs, cfg, train=True, num_workers=0)
        val_loader = make_dataloader([data_root / c for c in sp["val"]], cfg, train=False, batch_size=1)

        model = build_scaffold(block=cfg.model.block, features=cfg.model.features)
        ctx = setup_run("synthetic_check", cfg, base_dir=tmp / "runs", set_global_seed=0)
        train_model(model, train_loader, val_loader, cfg, ctx, device=device, max_epochs=args.epochs)

        summary = evaluate_and_log(
            model, eval_dirs, cfg, ctx, device=device, physics_key=load_physics_key(cfg)
        )
        ctx.finalize()

        # Did the metric recover the planted reliance?
        et_rows = [r for r in summary["reliance_matrix"] if r["region"] == DESIGNATED_REGION]
        top = max(et_rows, key=lambda r: r["score"])
        print("\nReliance for", DESIGNATED_REGION, "(higher = more relied on):")
        for r in sorted(et_rows, key=lambda r: -r["score"]):
            print(f"  {r['modality']:5} score={r['score']:.3f}  CI[{r['ci_low']:.3f},{r['ci_high']:.3f}]")
        passed = top["modality"] == DESIGNATED_MODALITY
        print(f"\n{'PASS' if passed else 'FAIL'}: {DESIGNATED_REGION} top reliance = "
              f"{top['modality']} (expected {DESIGNATED_MODALITY})")
        if not created_tmp:
            print("results:", ctx.run_dir / "results")
        return 0 if passed else 1
    finally:
        # Only remove the working dir if we created it; respect a user-supplied --workdir.
        if created_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
