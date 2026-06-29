"""Smoke tests for the skeleton: config + constants + physics key load and agree.

Run: pytest -q   (needs only pyyaml + the package; no torch/data required).
"""
from __future__ import annotations

import json

from brats_trust import constants
from brats_trust.config import REPO_ROOT, load_config


def test_config_loads_defaults():
    cfg = load_config()
    assert cfg.data.cohort == "BraTS2023-GLI"
    assert cfg.data.channel_order == list(constants.CHANNEL_ORDER)
    assert cfg.ablation.fill == "mean"  # roadmap S3.1: never zero by default


def test_channel_order_frozen():
    assert constants.CHANNEL_ORDER == ("FLAIR", "T1", "T1CE", "T2")
    assert constants.channel_index("T1CE") == 2


def test_physics_key_consistent_with_constants():
    cfg = load_config()
    key_path = REPO_ROOT / cfg.physics_key
    key = json.loads(key_path.read_text(encoding="utf-8"))
    assert key["channel_order"] == list(constants.CHANNEL_ORDER)
    # Every class names a physics modality that exists in the frozen channel order.
    for cls, spec in key["classes"].items():
        assert cls in constants.REGIONS
        assert spec["physics_modality"] in constants.CHANNEL_ORDER
    # File-suffix mapping matches constants.
    assert key["modality_files"] == constants.MODALITY_SUFFIX
