"""Stage 6 tests: modality-dropout augmentation and linear concept probing."""
from __future__ import annotations

import torch

from brats_trust.config import load_config
from brats_trust.data import synthetic
from brats_trust.engine import apply_modality_dropout
from brats_trust.metrics.probing import extract_bottleneck_features, linear_probe_modality_presence
from brats_trust.models.unet3d import build_scaffold


def test_modality_dropout_changes_some_samples_and_keeps_shape():
    torch.manual_seed(0)
    # Varied (non-constant) intensities so mean-fill actually changes the channel.
    images = torch.randn(8, 4, 8, 8, 8).abs() + 0.5
    out = apply_modality_dropout(images, prob=1.0)  # always drop one channel per sample
    assert out.shape == images.shape
    assert not torch.equal(out, images)            # at least some channel was mean-filled

    keep = apply_modality_dropout(images, prob=0.0)  # never drop
    assert torch.equal(keep, images)


def test_bottleneck_feature_extraction_shape():
    model = build_scaffold(features=[8, 16])
    feats = extract_bottleneck_features(model, torch.randn(4, 16, 16, 16))
    assert feats.ndim == 1 and feats.shape[0] == 32  # bottleneck width = features[-1]*2


def test_linear_probe_runs(tmp_path):
    model = build_scaffold(features=[8, 16])
    cfg = load_config()
    cfg.train.patch_size = [32, 32, 32]
    # 32^3 so that after brain-crop + two poolings the bottleneck stays > 1 voxel.
    dirs = synthetic.generate_dataset(tmp_path, n_cases=4, shape=(32, 32, 32),
                                      class_channels={3: 2, 2: 0}, seed=0)
    out = linear_probe_modality_presence(model, dirs, cfg, modality="T1CE")
    assert 0.0 <= out["probe_accuracy"] <= 1.0 and out["n"] == 8
