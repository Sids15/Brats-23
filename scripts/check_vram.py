#!/usr/bin/env python
"""VRAM survey: build each architecture, run one forward+backward on a synthetic
(batch, 4, 96, 96, 96) tensor under AMP, and report peak GPU memory.

Run on the GPU box to find out what actually fits before committing to
feature_size / use_checkpoint settings.  No data needed.

Usage:
    python scripts/check_vram.py                   # all architectures, batch 2
    python scripts/check_vram.py --batch 1         # lower batch
    python scripts/check_vram.py --models swin_unetr unetr  # subset
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch

# ── repo imports ────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from brats_trust.config import load_config          # noqa: E402
from brats_trust.models.factory import _BUILDERS     # noqa: E402
from brats_trust.models.base import IN_CHANNELS, OUT_CHANNELS  # noqa: E402


def _measure(model_name: str, cfg, batch: int, patch: int) -> dict:
    """Build *model_name*, run one fwd+bwd under AMP, return peak VRAM (MB)."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()

    builder = _BUILDERS[model_name]
    model = builder(cfg).cuda().train()

    x = torch.randn(batch, IN_CHANNELS, patch, patch, patch, device="cuda")

    # Support both old (torch.cuda.amp) and new (torch.amp) APIs.
    try:
        scaler = torch.amp.GradScaler("cuda")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler()

    def _autocast():
        """Return the right autocast context manager for the installed torch."""
        try:
            return torch.amp.autocast("cuda")
        except TypeError:
            return torch.cuda.amp.autocast()

    try:
        with _autocast():
            y = model(x)
            loss = y.mean()
        scaler.scale(loss).backward()
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        status = "OK"
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            status = "OOM"
        else:
            raise
    finally:
        del model, x
        torch.cuda.empty_cache()
        gc.collect()

    return {"model": model_name, "status": status, "peak_mb": round(peak_mb, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=2,
                        help="Batch size (default 2, the Phase 3 protocol value)")
    parser.add_argument("--patch", type=int, default=96,
                        help="Isotropic patch side (default 96, Phase 3 value)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of architectures to test (default: all)")
    parser.add_argument("--config", nargs="*", default=None,
                        help="Override YAML(s) to deep-merge over default.yaml")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. Run this on the GPU box.", file=sys.stderr)
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"GPU: {gpu_name}  ({gpu_mem:.1f} GB)")
    print(f"Patch: {args.patch}³   Batch: {args.batch}   AMP: on\n")

    overrides = args.config or []
    cfg = load_config(*overrides)
    # Override patch/batch to the values being tested.
    cfg.train.patch_size = [args.patch] * 3
    cfg.train.batch_size = args.batch

    names = args.models or list(_BUILDERS.keys())
    rows: list[dict] = []

    for name in names:
        if name not in _BUILDERS:
            print(f"  SKIP {name}: unknown architecture")
            continue
        print(f"  {name:20s} ... ", end="", flush=True)
        try:
            row = _measure(name, cfg, args.batch, args.patch)
        except Exception as exc:
            row = {"model": name, "status": f"ERROR: {exc}", "peak_mb": 0}
        rows.append(row)
        print(f"{row['status']:5s}  peak {row['peak_mb']:,.0f} MB")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"{'Model':20s}  {'Status':5s}  {'Peak MB':>10s}  {'% of GPU':>8s}")
    print("-" * 50)
    for r in rows:
        pct = r["peak_mb"] / (gpu_mem * 1024) * 100 if gpu_mem else 0
        print(f"{r['model']:20s}  {r['status']:5s}  {r['peak_mb']:10,.0f}  {pct:7.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
