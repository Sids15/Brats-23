"""Lightweight config system for BraTS-Trust.

Loads ``configs/default.yaml`` (the frozen S9 protocol) and optionally deep-merges
one or more override YAMLs on top. Returns an attribute-accessible namespace so
callers can write ``cfg.train.patch_size`` while still being able to dump back to
a plain dict for logging/reproducibility.

Usage:
    from brats_trust.config import load_config
    cfg = load_config()                          # defaults only
    cfg = load_config("configs/probe3_rf.yaml")  # default <- override
    print(cfg.data.root, cfg.train.lr)
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

# Repo root = two levels up from this file (src/brats_trust/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


class Config(SimpleNamespace):
    """Attribute-accessible nested config. ``to_dict()`` returns plain dicts."""

    def to_dict(self) -> dict[str, Any]:
        return _to_dict(self)


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def _to_dict(obj: Any) -> Any:
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins on leaves)."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(*overrides: str | Path, default: str | Path = DEFAULT_CONFIG) -> Config:
    """Load the default protocol, deep-merging any override YAML paths on top."""
    with open(default, "r", encoding="utf-8") as fh:
        merged: dict = yaml.safe_load(fh) or {}
    for ov in overrides:
        with open(ov, "r", encoding="utf-8") as fh:
            merged = _deep_merge(merged, yaml.safe_load(fh) or {})
    return _to_namespace(merged)


def load_physics_key(cfg: Config | None = None) -> dict:
    """Load ``physics_answer_key.json`` (path from config). Documents the physics
    expectation that informs the reliance test; never used as a penalty (roadmap CUT LIST)."""
    cfg = cfg or load_config()
    return json.loads((REPO_ROOT / cfg.physics_key).read_text(encoding="utf-8"))
