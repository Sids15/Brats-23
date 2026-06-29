"""Array-level preprocessing and the modality intervention (roadmap S9, S3.1).

All functions operate on plain NumPy arrays (channel-first images ``(C, X, Y, Z)``,
labels ``(X, Y, Z)``) so they are framework-free and unit-testable without torch.
The torch/MONAI dataloader (``data.dataset``) wraps these for training.

Conventions:
- *Brain mask* = voxels that are nonzero in any channel (BraTS volumes are skull-
  stripped, so background is exactly zero).
- z-score is computed over brain voxels only, per channel; background stays 0.
- Intervention defaults to **mean-fill** (roadmap S3.1): replace a channel's brain
  voxels with that channel's mean intensity, destroying its spatial information while
  keeping a realistic intensity level. **Zero-fill** is offered only for the OOD
  sensitivity comparison, never as the default.
"""
from __future__ import annotations

import numpy as np

Bbox = tuple[slice, ...]


def brain_mask(image: np.ndarray) -> np.ndarray:
    """Boolean ``(X, Y, Z)`` mask of voxels nonzero in any channel."""
    return np.any(image != 0, axis=0)


def nonzero_bbox(image: np.ndarray) -> Bbox:
    """Spatial bounding box (slices over X, Y, Z) of the brain in a ``(C,X,Y,Z)`` image."""
    mask = brain_mask(image)
    if not mask.any():
        return tuple(slice(0, s) for s in mask.shape)
    coords = np.array(np.nonzero(mask))
    lo = coords.min(axis=1)
    hi = coords.max(axis=1) + 1
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def crop_to_bbox(arr: np.ndarray, bbox: Bbox) -> np.ndarray:
    """Apply a spatial ``bbox`` to the trailing 3 dims of ``arr`` (2D or 3D-leading)."""
    if arr.ndim == len(bbox):  # labels (X, Y, Z)
        return arr[bbox]
    return arr[(slice(None), *bbox)]  # image (C, X, Y, Z)


def brain_crop(image: np.ndarray, label: np.ndarray | None = None):
    """Crop image (and matching label) to the brain bounding box (roadmap S9)."""
    bbox = nonzero_bbox(image)
    cropped_image = crop_to_bbox(image, bbox)
    if label is None:
        return cropped_image
    return cropped_image, crop_to_bbox(label, bbox)


def per_channel_zscore(image: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Z-score each channel over brain voxels; background remains 0."""
    if mask is None:
        mask = brain_mask(image)
    out = np.zeros_like(image, dtype=np.float32)
    for c in range(image.shape[0]):
        vals = image[c][mask]
        if vals.size == 0:
            continue
        std = vals.std()
        mean = vals.mean()
        out[c][mask] = ((image[c][mask] - mean) / std) if std > 0 else 0.0
    return out


def intervene(
    image: np.ndarray,
    channel: int,
    fill: str = "mean",
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a copy of ``image`` with one ``channel`` intervened on (roadmap S3.1).

    ``fill='mean'`` replaces brain voxels of the channel with their mean (spatial
    information removed, intensity level preserved). ``fill='zero'`` zeros the whole
    channel (OOD; sensitivity comparison only).
    """
    out = image.copy()
    if fill == "zero":
        out[channel] = 0.0
        return out
    if fill == "mean":
        if mask is None:
            mask = brain_mask(image)
        brain_vals = image[channel][mask]
        fill_value = float(brain_vals.mean()) if brain_vals.size else 0.0
        out[channel] = 0.0
        out[channel][mask] = fill_value
        return out
    raise ValueError(f"unknown fill mode: {fill!r} (expected 'mean' or 'zero')")
