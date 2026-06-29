"""Linear concept probing (roadmap S6, triangulation only).

Train a 1-layer logistic probe on frozen bottleneck features to predict 'is modality X
present?' (present case vs the same case with X mean-filled). High accuracy means the
modality is *decodable* from the representation. Stated limit (roadmap S6): **decodable !=
used** -- this only triangulates the reliance result, it does not replace it.

Works on models exposing a ``.bottleneck`` module (our scaffold). NumPy + scikit-learn.
"""
from __future__ import annotations

import numpy as np
import torch

from ..constants import channel_index
from .reliance import _load_case, intervene_tensor


def extract_bottleneck_features(model, image, device=None) -> np.ndarray:
    """Global-average-pooled bottleneck feature vector for one image (needs ``model.bottleneck``)."""
    device = device or next(model.parameters()).device
    captured: dict[str, torch.Tensor] = {}
    handle = model.bottleneck.register_forward_hook(lambda m, i, o: captured.__setitem__("f", o.detach()))
    model.eval()
    with torch.no_grad():
        model(image.unsqueeze(0).to(device))
    handle.remove()
    return captured["f"].mean(dim=(2, 3, 4)).cpu().numpy().ravel()


def linear_probe_modality_presence(model, case_dirs, cfg, modality: str, device=None) -> dict:
    """Cross-validated accuracy of a linear probe predicting whether ``modality`` is present."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    ci = channel_index(modality)
    features, labels = [], []
    for case_dir in case_dirs:
        image, _ = _load_case(case_dir, cfg)
        features.append(extract_bottleneck_features(model, image, device))
        labels.append(1)
        features.append(extract_bottleneck_features(model, intervene_tensor(image, ci, "mean"), device))
        labels.append(0)

    x = np.asarray(features)
    y = np.asarray(labels)
    cv = max(2, min(5, len(y) // 2))
    acc = cross_val_score(LogisticRegression(max_iter=1000), x, y, cv=cv)
    return {"modality": modality, "probe_accuracy": float(acc.mean()), "n": int(len(y))}
