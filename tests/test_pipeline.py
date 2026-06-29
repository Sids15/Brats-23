"""Fast CPU end-to-end wiring test: synthetic data -> train -> evaluate -> tidy outputs.

This proves the whole chain is wired correctly (shapes, dataloader, training step,
intervention, metrics, logging) so the GPU machine is plug-and-play. It does NOT assert
the scientific reliance result -- that needs real training and lives in
``scripts/run_synthetic_check.py`` (roadmap S3.5).
"""
from __future__ import annotations

import csv

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits, synthetic
from brats_trust.data.dataset import make_dataloader
from brats_trust.engine import train_model
from brats_trust.logging_utils import (
    FRAGILITY_COLUMNS,
    PER_CASE_COLUMNS,
    RELIANCE_COLUMNS,
    setup_run,
)
from brats_trust.models.scaffold import build_scaffold
from brats_trust.pipeline import evaluate_and_log


def _tiny_cfg():
    cfg = load_config()
    cfg.train.patch_size = [16, 16, 16]
    cfg.inference.roi_size = [16, 16, 16]
    cfg.inference.sw_batch_size = 1
    cfg.model.features = [8, 16]
    cfg.train.batch_size = 2
    cfg.train.val_interval = 1
    return cfg


def _csv_header(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return next(csv.reader(fh))


def test_end_to_end_pipeline_on_synthetic(tmp_path):
    cfg = _tiny_cfg()
    device = torch.device("cpu")
    data_root = tmp_path / "data"
    synthetic.generate_dataset(data_root, n_cases=6, shape=(20, 20, 20),
                               class_channels={3: 2, 2: 0}, seed=0)
    sp = splits.make_splits(data_root, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=0)
    train_dirs = [data_root / c for c in sp["train"]]
    eval_dirs = [data_root / c for c in (sp["val"] + sp["test"])]

    train_loader = make_dataloader(train_dirs, cfg, train=True, num_workers=0)
    val_loader = make_dataloader([data_root / c for c in sp["val"]], cfg, train=False, batch_size=1)
    model = build_scaffold(block="conv", features=cfg.model.features)

    ctx = setup_run("pytest_e2e", cfg, base_dir=tmp_path / "runs", set_global_seed=0)
    best = train_model(model, train_loader, val_loader, cfg, ctx, device=device, max_epochs=2)
    assert isinstance(best, float)
    assert (ctx.run_dir / "best_model.pt").exists()

    out = evaluate_and_log(model, eval_dirs, cfg, ctx, device=device, physics_key=load_physics_key())
    ctx.finalize()

    # 4 modalities x 3 regions = 12 reliance cells; 3 regions x 2 roles = 6 fragility rows.
    assert len(out["reliance_matrix"]) == 12
    assert len(out["fragility"]) == 6

    results = ctx.run_dir / "results"
    assert _csv_header(results / "per_case_metrics.csv")[: len(PER_CASE_COLUMNS)] == list(PER_CASE_COLUMNS)
    assert _csv_header(results / "reliance_matrix.csv") == list(RELIANCE_COLUMNS)
    assert _csv_header(results / "fragility.csv") == list(FRAGILITY_COLUMNS)
    # Reliance scores are valid fractions in [0, 1].
    assert all(0.0 <= r["score"] <= 1.0 for r in out["reliance_matrix"])


def test_block_swap_builds(tmp_path):
    # Both pluggable blocks (Probe 3) instantiate and run a forward pass.
    x = torch.randn(1, 4, 16, 16, 16)
    for block in ("conv", "dwsep"):
        model = build_scaffold(block=block, features=[8, 16])
        assert model(x).shape == (1, 3, 16, 16, 16)
