"""Tests for Phase 3 — the Stage 3 architecture sweep (roadmap S4 Probe 1, S5).

Covers the three things that cannot be checked on the GPU machine after the fact: that the
SegMamba stem really does shrink the Mamba sequence, that a run's summary row carries the
capacity numbers the matched-protocol claim rests on, and that the sweep's resume bookkeeping
and cross-architecture statistics behave.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from brats_trust.config import load_config
from brats_trust.experiments import SUMMARY_COLUMNS
from brats_trust.models import build_model, model_cost
from brats_trust.models import segmamba as segmamba_module

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a `scripts/` CLI by path -- they are entry points, not an installed package."""
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubMamba(nn.Module):
    """Stands in for the CUDA-only `mamba_ssm.Mamba` so the surrounding 3D plumbing is
    testable on CPU. Shape contract only -- it says nothing about state-space dynamics."""

    def __init__(self, d_model: int, **_: object) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


@pytest.fixture
def stub_mamba(monkeypatch):
    monkeypatch.setattr(segmamba_module, "Mamba", _StubMamba, raising=False)
    monkeypatch.setattr(segmamba_module, "_HAS_MAMBA", True)


def test_segmamba_stem_halves_the_mamba_sequence(stub_mamba):
    # The whole point of the stem: Mamba scans (patch/2)^3 tokens, not patch^3. At the
    # production 96^3 patch that is 110k tokens instead of 885k.
    model = segmamba_module.build_segmamba(features=(8, 16)).eval()
    seen: list[tuple[int, ...]] = []
    model.encoders[0].register_forward_hook(lambda _m, inp, _o: seen.append(tuple(inp[0].shape[2:])))

    with torch.no_grad():
        model(torch.randn(1, 4, 32, 32, 32))

    assert seen == [(16, 16, 16)]


def test_segmamba_output_matches_input_grid(stub_mamba):
    # The stem downsamples; the decoder must put the prediction back on the input grid.
    model = segmamba_module.build_segmamba(features=(8, 16)).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 32, 32, 32))
    assert out.shape == (1, 3, 32, 32, 32)


def test_segmamba_build_still_guarded_without_mamba(monkeypatch):
    monkeypatch.setattr(segmamba_module, "_HAS_MAMBA", False)
    with pytest.raises(ImportError, match="mamba-ssm"):
        segmamba_module.build_segmamba()


def test_model_cost_reports_params_and_flops():
    cfg = load_config()
    cfg.model.features = [8, 16]
    cfg.train.patch_size = [32, 32, 32]
    cost = model_cost(build_model(cfg), cfg.train.patch_size)
    assert cost["params"] > 0
    assert cost["flops"] > 0


def test_summary_columns_carry_capacity_and_epochs():
    # analyze_probe1 divides params/flops; run_phase*_robust resumes off epochs. Guard drift.
    for column in ("epochs", "batch_size", "params", "flops"):
        assert column in SUMMARY_COLUMNS


def test_run_single_defaults_to_zero_dataloader_workers():
    # MONAI re-seeds random transforms per worker, so the worker count is part of the
    # augmentation stream. Sweeps already on disk (Phase 2) ran at 0; a change to this
    # default would silently make their completed and resumed runs incomparable.
    import inspect

    from brats_trust.experiments import run_single

    assert inspect.signature(run_single).parameters["num_workers"].default == 0


def test_load_completed_skips_only_finished_runs(tmp_path):
    import run_phase3_robust

    summary = tmp_path / "phase3_summary.jsonl"
    summary.write_text(
        json.dumps({"model": "unet3d", "seed": 42, "epochs": 30}) + "\n"
        + json.dumps({"model": "unetr", "seed": 42, "epochs": 12}) + "\n",  # crashed partway
        encoding="utf-8",
    )
    assert run_phase3_robust.load_completed(summary, epochs=30) == {"unet3d_seed42"}


def test_load_completed_on_missing_summary(tmp_path):
    import run_phase3_robust

    assert run_phase3_robust.load_completed(tmp_path / "absent.jsonl", epochs=30) == set()


def _rows() -> list[dict]:
    """Two architectures x 3 seeds; `wide` is less faithful and has a larger ERF."""
    rows = []
    for seed, (f_narrow, f_wide) in enumerate([(0.80, 0.40), (0.82, 0.44), (0.78, 0.42)]):
        rows.append({"model": "narrow", "seed": seed, "erf": 10.0 + seed,
                     "faithfulness_overall": f_narrow, "val_dice": 0.7,
                     "params": 1e6, "flops": 2e9})
        rows.append({"model": "wide", "seed": seed, "erf": 30.0 + seed,
                     "faithfulness_overall": f_wide, "val_dice": 0.71,
                     "params": 4e6, "flops": 8e9})
    return rows


def test_per_architecture_summarizes_each_model():
    analyze = _load_script("analyze_probe1")
    arch = {r["model"]: r for r in analyze.per_architecture(_rows())}

    assert arch["narrow"]["n_seeds"] == 3
    assert arch["narrow"]["faithfulness_mean"] == pytest.approx(0.80, abs=0.01)
    assert arch["wide"]["faithfulness_mean"] == pytest.approx(0.42, abs=0.01)
    assert arch["narrow"]["faithfulness_ci_low"] <= arch["narrow"]["faithfulness_mean"]
    assert arch["wide"]["params_m"] == pytest.approx(4.0)
    assert arch["wide"]["flops_g"] == pytest.approx(8.0)


def test_pairwise_faithfulness_reports_effect_size_and_corrected_p():
    analyze = _load_script("analyze_probe1")
    pairs = analyze.pairwise_faithfulness(_rows())

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["delta_faithfulness"] > 0  # narrow is the more faithful arm
    assert pair["cliffs_delta"] == pytest.approx(1.0)  # every narrow seed beats every wide one
    assert pair["p_holm"] >= pair["p_raw"]


def test_pairwise_faithfulness_skips_single_seed_architectures():
    analyze = _load_script("analyze_probe1")
    rows = _rows() + [{"model": "lonely", "seed": 0, "erf": 20.0,
                       "faithfulness_overall": 0.5, "val_dice": 0.6}]
    pairs = analyze.pairwise_faithfulness(rows)

    assert {p["model_a"] for p in pairs} | {p["model_b"] for p in pairs} == {"narrow", "wide"}


def test_analyze_probe1_writes_table_stats_and_figure(tmp_path):
    analyze = _load_script("analyze_probe1")
    arch_rows = analyze.per_architecture(_rows())
    out = tmp_path / "architecture_faithfulness.png"
    analyze.make_figure(_rows(), arch_rows, out)
    assert out.exists() and out.stat().st_size > 0
