"""Tests for the model factory (Stage 3 architectures).

The MONAI-based anchors are verified on CPU. SegMamba needs mamba-ssm (CUDA) and is skipped
where that isn't installed -- its forward pass is verified on the GPU machine.
"""
from __future__ import annotations

import pytest
import torch

from brats_trust.config import load_config
from brats_trust.models.factory import build_model
from brats_trust.models.segmamba import _HAS_MAMBA


def _cfg(name, size):
    cfg = load_config()
    cfg.model.name = name
    cfg.train.patch_size = [size, size, size]
    cfg.model.features = [16, 32]
    return cfg


@pytest.mark.parametrize("name,size", [
    ("unet3d", 32),
    ("dynunet", 32),
    ("unetr", 32),
    ("swin_unetr", 64),  # Swin needs >=64^3 (downsamples by 32)
])
def test_anchor_builds_and_forwards(name, size):
    model = build_model(_cfg(name, size)).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, size, size, size))
    assert out.shape == (1, 3, size, size, size)


def test_scaffold_handles_arbitrary_input_size():
    # Brain-cropped volumes have odd, non-power-of-2 dims; the U-Net must still run.
    model = build_model(_cfg("unet3d", 32)).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 28, 30, 26))
    assert out.shape == (1, 3, 28, 30, 26)


def test_unknown_model_raises():
    cfg = _cfg("not_a_model", 32)
    with pytest.raises(ValueError):
        build_model(cfg)


def test_registry_lists_the_five_models():
    # Each architecture is its own module registered by name; guard against drift.
    from brats_trust.models import MODELS

    assert set(MODELS) == {"unet3d", "dynunet", "unetr", "swin_unetr", "segmamba"}


@pytest.mark.skipif(not _HAS_MAMBA, reason="mamba-ssm not installed (CUDA-only); verify on GPU")
def test_segmamba_forward():
    model = build_model(_cfg("segmamba", 32)).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 4, 32, 32, 32))
    assert out.shape == (1, 3, 32, 32, 32)
