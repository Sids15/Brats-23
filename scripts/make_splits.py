"""CLI: build leakage-free patient-level splits and write them to disk.

    python scripts/make_splits.py --config configs/default.yaml

Parses the config and calls brats_trust.data.splits.make_splits to write the
patient-level, leakage-free splits to disk.
"""
from __future__ import annotations

import argparse

from brats_trust.config import load_config
from brats_trust.data import splits


def main() -> None:
    ap = argparse.ArgumentParser(description="Build patient-level BraTS splits.")
    ap.add_argument("--config", default=None, help="Override YAML on top of defaults.")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    splits.make_splits(
        root=cfg.data.root,
        fractions=cfg.data.split_fractions.to_dict(),
        seed=cfg.seed,
        out_path=cfg.data.splits_path,
    )


if __name__ == "__main__":
    main()
