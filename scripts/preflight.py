"""Dataset preflight CLI — run this FIRST on the real BraTS-2023 data (roadmap S8/S9).

    python scripts/preflight.py --config configs/default.yaml
    python scripts/preflight.py --root D:/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData

Validates the cohort before any GPU time is spent (see brats_trust.preflight). Writes a
manifest CSV + JSON summary and exits non-zero if any case fails.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from brats_trust.config import load_config
from brats_trust.preflight import run_preflight


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the BraTS-2023 dataset before training.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=None, help="Override data root.")
    ap.add_argument("--out", default="outputs/preflight", help="Where to write manifest + summary.")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    root = Path(args.root) if args.root else Path(cfg.data.root)
    if not root.is_dir():
        print(f"ERROR: data root not found: {root}", file=sys.stderr)
        return 2

    summary = run_preflight(root, out_dir=Path(args.out))
    print(json.dumps(summary, indent=2))
    if summary["n_cases"] == 0:
        print("ERROR: no cases discovered.", file=sys.stderr)
        return 2
    return 1 if summary["n_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
