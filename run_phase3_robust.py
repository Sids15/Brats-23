"""Phase 3 — the Stage 3 architecture sweep (roadmap S4 Probe 1, S5 Tier-A anchors).

Trains five architectures x five seeds under one matched protocol and records, per run, the
effective receptive field, the conditional-reliance faithfulness score, and val Dice. Phase 2
varied receptive field *within* the conv family; this varies the mechanism itself (conv vs
transformer vs state-space) to show the reliance phenomenon is not an artefact of our own
scaffold.

CAVEAT to carry into the write-up (roadmap S4): swapping architecture changes receptive
field, optimization, inductive bias, and normalization simultaneously. Report as
"contribution under a matched protocol" -- never "causes".

Built for a multi-day detached run:
  * resumes -- a completed run is skipped, an interrupted one restarts from its last epoch;
  * a sweep-level transcript survives crashes that kill a single run's logger;
  * one architecture failing (e.g. no mamba-ssm) never takes the sweep down with it.

    python run_phase3_robust.py
    python run_phase3_robust.py --models unet3d dynunet --seeds 42 43
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from brats_trust.config import load_config, load_physics_key
from brats_trust.data import splits
from brats_trust.engine import get_device
from brats_trust.experiments import SUMMARY_COLUMNS, run_single
from brats_trust.logging_utils import tee_stdout, write_tidy

CONFIG = "configs/sweep.yaml"
OUT_DIR = Path("outputs/phase3")

# Removed unet3d as we already trained it extensively in Phase 2.
ARCHITECTURES = ["dynunet", "unetr", "swin_unetr", "segmamba"]


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_completed(summary_file: Path, epochs: int, suffix: str = "") -> set[str]:
    """Run ids in ``summary_file`` that already finished the full ``epochs`` budget.

    A row is only appended after evaluation succeeds, so its presence at the target epoch
    count means there is nothing left to do for that architecture x seed.
    """
    if not summary_file.exists():
        return set()
    completed = set()
    for line in summary_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row.get("epochs", 0)) >= epochs:
            completed.add(f"{row['model']}{suffix}_seed{row['seed']}")
    return completed


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Phase 3 architecture sweep.")
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--models", nargs="*", default=None, help=f"Subset of {ARCHITECTURES}")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--suffix", default="", help="Optional suffix for model names, e.g. '_tuned'")
    args = ap.parse_args()

    out_dir = Path(args.out)
    summary_file = out_dir / "phase3_summary.jsonl"
    tee_stdout(out_dir / "phase3_master_terminal.log")

    # cudnn autotuning pays for itself: the patch size is fixed for the whole sweep.
    torch.backends.cudnn.benchmark = True
    device = get_device()

    base_cfg = load_config(args.config)
    physics_key = load_physics_key(base_cfg)
    models = args.models or ARCHITECTURES
    # 5 seeds for perfect rigorous matching with Phase 2, thanks to NVMe speed!
    seeds = args.seeds or [42, 43, 44, 45, 46]
    epochs = args.epochs if args.epochs is not None else 30

    root = Path(base_cfg.data.root)
    splits_path = Path(base_cfg.data.splits_path)
    sp = (splits.load_splits(splits_path) if splits_path.exists() else splits.make_splits(
        root, base_cfg.data.split_fractions.to_dict(), base_cfg.seed, splits_path))
    train_dirs = [root / c for c in sp["train"]]
    val_dirs = [root / c for c in sp["val"]]
    eval_dirs = [root / c for c in (sp["val"] + sp["test"])]

    completed = load_completed(summary_file, epochs, suffix=args.suffix)
    _log(f"PHASE 3 SWEEP | device={device} | {len(models)} models x {len(seeds)} seeds "
         f"x {epochs} epochs | {len(completed)} runs already complete")
    _log(f"train={len(train_dirs)} val={len(val_dirs)} eval={len(eval_dirs)} cases")

    rows: list[dict] = []
    for name in models:
        for seed in seeds:
            run_id = f"{name}{args.suffix}_seed{seed}"
            if run_id in completed:
                _log(f"SKIP {run_id} (already complete)")
                continue

            cfg = load_config(args.config)  # fresh, so one run can never mutate the next
            cfg.model.name = name

            # EXACT PHASE 2 CONTINUITY MATCH:
            cfg.train.patch_size = [96, 96, 96]
            cfg.inference.roi_size = [96, 96, 96]
            cfg.train.num_workers = getattr(cfg.train, "num_workers", 2)

            _log(f"START {run_id}")
            start = time.time()
            try:
                row = run_single(cfg, f"phase3_{run_id}", train_dirs, val_dirs, eval_dirs,
                                 physics_key, device=device, base_dir="runs",
                                 epochs=epochs, seed=seed,
                                 num_workers=cfg.train.num_workers)
            except ImportError as exc:  # segmamba without mamba-ssm: an expected absence
                _log(f"SKIP {run_id}: {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 -- one bad run must not end the sweep
                _log(f"FAIL {run_id}: {exc}")
                traceback.print_exc()
                continue
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            rows.append(row)
            with open(summary_file, "a", encoding="utf-8") as fh:  # persist before the next run
                fh.write(json.dumps(row) + "\n")
            _log(f"DONE {run_id} in {(time.time() - start) / 60:.1f} min | "
                 f"ERF {row['erf']:.2f} | faithfulness {row['faithfulness_overall']:.3f} | "
                 f"val Dice {row['val_dice']:.4f}")

    # Rewrite the tidy CSV from every row on disk, so a resumed sweep still emits a
    # complete table rather than only the runs this invocation happened to execute.
    all_rows = [json.loads(line) for line in
                summary_file.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if summary_file.exists() else []
    write_tidy(out_dir / "phase3_summary", all_rows, SUMMARY_COLUMNS)

    _log(f"PHASE 3 COMPLETE | {len(rows)} new runs, {len(all_rows)} total "
         f"-> {out_dir / 'phase3_summary.csv'}")
    print(f"next: python scripts/analyze_probe1.py --summary {summary_file}")


if __name__ == "__main__":
    main()
