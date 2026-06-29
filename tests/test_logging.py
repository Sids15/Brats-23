"""Tests for the research-grade logging/reproducibility layer (torch-free)."""
from __future__ import annotations

import json

from brats_trust import logging_utils as lu
from brats_trust.config import load_config


def test_set_seed_is_deterministic():
    import random

    lu.set_seed(123)
    a = [random.random() for _ in range(5)]
    lu.set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_capture_environment_has_required_keys():
    env = lu.capture_environment(seeds={"seed": 7})
    for key in ("python", "platform", "git", "packages", "torch", "seeds", "command_line"):
        assert key in env
    assert env["seeds"] == {"seed": 7}
    assert "dirty" in env["git"]  # reproducibility-critical flag present


def test_setup_run_creates_reproducibility_artifacts(tmp_path):
    cfg = load_config()
    ctx = lu.setup_run("unit_test", cfg, base_dir=tmp_path, set_global_seed=42)
    assert (ctx.run_dir / "config.yaml").exists()
    assert (ctx.run_dir / "env.json").exists()
    assert (ctx.run_dir / "run.log").exists()
    env = json.loads((ctx.run_dir / "env.json").read_text())
    assert env["seeds"]["seed"] == 42
    summary = ctx.finalize()
    assert (ctx.run_dir / "run_summary.json").exists()
    assert "gpu_hours" in summary


def test_metric_logger_round_trip(tmp_path):
    ml = lu.MetricLogger(tmp_path / "metrics.jsonl", tmp_path / "metrics.csv")
    ml.log(step=0, split="train", loss=1.0, dice_wt=0.5)
    ml.log(step=1, split="val", loss=0.8, dice_wt=0.6)
    lines = (tmp_path / "metrics.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["dice_wt"] == 0.5
    assert (tmp_path / "metrics.csv").read_text().startswith("step,split")


def test_write_tidy_emits_csv_and_jsonl(tmp_path):
    rows = [
        {"region": "WT", "modality": "FLAIR", "fill": "mean", "score": 0.12},
        {"region": "ET", "modality": "T1CE", "fill": "mean", "score": 0.71},
    ]
    stem = tmp_path / "reliance_matrix"
    lu.write_tidy(stem, rows, columns=lu.RELIANCE_COLUMNS)
    assert stem.with_suffix(".csv").exists()
    assert stem.with_suffix(".jsonl").exists()
    jl = stem.with_suffix(".jsonl").read_text().strip().splitlines()
    assert json.loads(jl[1])["modality"] == "T1CE"
