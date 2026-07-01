"""Research-grade run logging and reproducibility for BraTS-Trust.

Everything an experiment needs for the paper is captured *automatically* into one
timestamped run directory, so every table, figure, and reproducibility statement
can be regenerated from disk without re-running anything.

Run-directory schema (created by :func:`setup_run`)::

    runs/<UTC-timestamp>__<name>/
        config.yaml          # fully-resolved config snapshot (S9 protocol used)
        env.json             # python/OS/host, package versions, GPU, git hash, seeds
        run.log              # full console log (ISO-timestamped)
        metrics.jsonl        # time-series rows (training/val curves)  -> convergence (S4.2)
        metrics.csv          # same rows, flattened, for spreadsheets/plots
        results/             # paper-ready tidy outputs (one concern per file stem):
            per_case_metrics.{csv,jsonl}   # case x region x {dice,hd95,sens,spec,...}
            aggregate_metrics.{csv,jsonl}  # region x {mean,std,median,iqr,ci_low,ci_high}
            reliance_matrix.{csv,jsonl}    # region x modality x {score, fill}     (S3.1)
            fragility.{csv,jsonl}          # leaned-on vs physics-correct drop+gap (S3.3)
            erf.{csv,jsonl}                # variant x effective_receptive_field   (S4.1)
            stats.json                     # effect sizes, CIs, corrected p-values (S3/S4.2)
        run_summary.json     # written by RunContext.finalize(): timings, GPU-hours

The metric *values* are produced by later stages (data/models/metrics); this module
provides the writers + canonical schema so those stages just hand over rows of dicts.

Design notes:
- Cohort is BraTS-2023 adult glioma (GLI). BraTS-2023 scores Dice + HD95 per region
  and additionally defines *lesion-wise* Dice/HD95 -- both have columns reserved in the
  per-case schema (see ``PER_CASE_COLUMNS``).
- ``torch``/``numpy`` are imported lazily so this module works on a machine without the
  full ML stack installed (e.g. for unit tests and dry runs).
"""
from __future__ import annotations

import csv
import json
import logging
import platform
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml

from .config import REPO_ROOT

# Packages whose exact versions matter for reproducing results in the paper.
_TRACKED_PACKAGES = (
    "torch", "monai", "numpy", "scipy", "nibabel", "scikit-learn",
    "statsmodels", "pandas", "pyyaml", "matplotlib", "einops",
    "grad-cam", "pytest", "ruff",
)

# Canonical tidy schemas (documented here so every stage emits identical columns).
PER_CASE_COLUMNS = (
    "case_id", "patient_id", "region",      # region in {WT, TC, ET}
    "dice", "hd95", "sensitivity", "specificity",
    "lesionwise_dice", "lesionwise_hd95",   # BraTS-2023 lesion-wise metrics
)
AGGREGATE_COLUMNS = (
    "region", "metric", "n", "mean", "std", "median", "iqr_low", "iqr_high",
    "ci_low", "ci_high",                     # 95% bootstrap CI
)
RELIANCE_COLUMNS = (
    "region", "modality", "fill", "score", "ci_low", "ci_high",  # fill in {mean, zero}
)
FRAGILITY_COLUMNS = (
    "region", "removed_modality", "role",    # role in {leaned_on, physics_correct}
    "dice_full", "dice_dropped", "delta", "ci_low", "ci_high",
    "n_cases",                               # defined-reference cases (empty-GT excluded)
)


# --------------------------------------------------------------------------- #
# Reproducibility primitives
# --------------------------------------------------------------------------- #
def get_git_revision(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    """Return commit hash, short hash, branch, and a dirty-tree flag.

    The dirty flag is essential: a result produced from uncommitted changes is not
    reproducible, and the paper must be able to state the exact tree state.
    """
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=repo_root, capture_output=True, text=True, check=True
            )
            return out.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "short": commit[:9] if commit else None,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed ``random``, ``numpy``, and ``torch`` (if importable) for reproducibility.

    With ``deterministic=True`` also enables cuDNN deterministic mode so repeated runs
    with the same seed produce identical results -- required for the seeded sweeps
    (roadmap S4.2: >=5 confirmatory seeds).
    """
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    info: dict[str, Any] = {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "devices": [],
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info["devices"].append(
                {"name": props.name, "total_memory_mb": round(props.total_memory / 1024**2)}
            )
    return info


def capture_environment(seeds: dict[str, Any] | None = None) -> dict[str, Any]:
    """Snapshot everything needed to reproduce a run on another machine."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "hostname": socket.gethostname(),
        },
        "git": get_git_revision(),
        "packages": _package_versions(),
        "torch": _torch_info(),
        "seeds": seeds or {},
        "command_line": sys.argv,
        "cwd": str(Path.cwd()),
    }


# --------------------------------------------------------------------------- #
# Time-series metric logging (training/validation curves)
# --------------------------------------------------------------------------- #
class MetricLogger:
    """Append per-step metrics to ``metrics.jsonl`` and a flat ``metrics.csv``.

    Each call to :meth:`log` writes one row and flushes immediately (crash-safe, so a
    killed run still yields usable convergence curves). The CSV header is the union of
    all keys seen so far; new columns trigger a rewrite so the file stays rectangular.
    """

    def __init__(self, jsonl_path: Path, csv_path: Path) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self._rows: list[dict[str, Any]] = []
        self._columns: list[str] = []

    def log(self, step: int, split: str, **metrics: Any) -> None:
        row = {"step": step, "split": split, **metrics}
        self._rows.append(row)
        with open(self.jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
        new_cols = [k for k in row if k not in self._columns]
        if new_cols:
            self._columns.extend(new_cols)
            self._rewrite_csv()
        else:
            with open(self.csv_path, "a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=self._columns).writerow(row)

    def _rewrite_csv(self) -> None:
        with open(self.csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._columns)
            writer.writeheader()
            writer.writerows(self._rows)


# --------------------------------------------------------------------------- #
# Console/log formatting
# --------------------------------------------------------------------------- #
def log_banner(
    logger: logging.Logger, title: str, fields: dict[str, Any] | None = None, width: int = 60
) -> None:
    """Log a boxed title (and an aligned key/value block) through the run logger.

    Emitted line-by-line via ``logger`` so it lands in both the console and ``run.log``
    with timestamps -- a scannable, paper-ready header of what a run was configured with.
    """
    bar = "=" * width
    logger.info(bar)
    logger.info(title)
    logger.info(bar)
    if fields:
        key_width = max(len(k) for k in fields)
        for key, value in fields.items():
            logger.info("  %-*s  %s", key_width + 1, f"{key}:", value)
        logger.info(bar)


# --------------------------------------------------------------------------- #
# Tidy result writers (paper tables/figures source of truth)
# --------------------------------------------------------------------------- #
def write_json(path: Path, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path


def write_tidy(stem: Path, rows: list[dict[str, Any]], columns: tuple[str, ...] | None = None) -> None:
    """Write ``rows`` to both ``<stem>.csv`` and ``<stem>.jsonl``.

    Two formats on purpose: CSV for spreadsheets/plotting, JSONL for programmatic
    re-aggregation across runs. ``columns`` fixes/orders the CSV header (use the
    canonical ``*_COLUMNS`` constants); extra keys in rows are still written.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    with open(stem.with_suffix(".jsonl"), "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    if columns is None:
        seen: list[str] = []
        for row in rows:
            seen.extend(k for k in row if k not in seen)
        columns = tuple(seen)
    else:
        extra: list[str] = []
        for row in rows:
            extra.extend(k for k in row if k not in columns and k not in extra)
        columns = (*columns, *extra)
    with open(stem.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------- #
# Run context
# --------------------------------------------------------------------------- #
@dataclass
class RunContext:
    """Handle to one experiment's run directory, logger, and writers."""

    name: str
    run_dir: Path
    logger: logging.Logger
    metrics: MetricLogger
    start_time: float = field(default_factory=time.time)

    @property
    def results_dir(self) -> Path:
        d = self.run_dir / "results"
        d.mkdir(exist_ok=True)
        return d

    def result(self, stem: str) -> Path:
        """Path (without suffix) inside ``results/`` for a tidy output, e.g. 'reliance_matrix'."""
        return self.results_dir / stem

    def finalize(self, **extra: Any) -> dict[str, Any]:
        """Write ``run_summary.json`` with timings (and GPU-hours if a GPU is present)."""
        duration_s = time.time() - self.start_time
        torch_info = _torch_info()
        n_gpus = len(torch_info.get("devices", [])) if torch_info.get("available") else 0
        summary = {
            "name": self.name,
            "run_dir": str(self.run_dir),
            "end_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(duration_s, 3),
            "duration_hours": round(duration_s / 3600, 4),
            "gpu_count": n_gpus,
            "gpu_hours": round(duration_s / 3600 * n_gpus, 4),
            **extra,
        }
        write_json(self.run_dir / "run_summary.json", summary)
        self.logger.info("Run finalized: %.1fs (%.3f GPU-h)", duration_s, summary["gpu_hours"])
        self.close()
        return summary

    def close(self) -> None:
        """Close and detach log handlers (releases ``run.log`` so the dir can be removed)."""
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)


def _build_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"brats_trust.run.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def setup_run(
    name: str,
    cfg: Any,
    base_dir: str | Path = "runs",
    seeds: dict[str, Any] | None = None,
    set_global_seed: int | None = None,
) -> RunContext:
    """Create a timestamped run directory and return a ready-to-use :class:`RunContext`.

    Writes ``config.yaml`` (resolved snapshot) and ``env.json`` immediately, so even a
    run that crashes in epoch 0 leaves a complete reproducibility record.

    Args:
        name: short experiment name (e.g. ``"probe3_rf_k5"``); appears in the dir name.
        cfg: a :class:`~brats_trust.config.Config` or plain dict to snapshot.
        base_dir: parent for run dirs (default ``runs/``, git-ignored).
        seeds: dict recorded in ``env.json`` (e.g. ``{"seed": 42}``).
        set_global_seed: if given, also calls :func:`set_seed` for convenience.
    """
    if set_global_seed is not None:
        set_seed(set_global_seed)
        seeds = {**(seeds or {}), "seed": set_global_seed}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base_dir) / f"{timestamp}__{name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results").mkdir(exist_ok=True)

    cfg_dict = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg_dict, fh, sort_keys=False)
    write_json(run_dir / "env.json", capture_environment(seeds=seeds))

    logger = _build_logger(name, run_dir / "run.log")
    metrics = MetricLogger(run_dir / "metrics.jsonl", run_dir / "metrics.csv")
    logger.info("Run '%s' -> %s", name, run_dir)
    return RunContext(name=name, run_dir=run_dir, logger=logger, metrics=metrics)
