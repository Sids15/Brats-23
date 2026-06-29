"""Synthetic calibration dataset (roadmap S3.5 — sanity check, NOT a lead result).

Generates small multi-channel volumes with a *known* class->channel coupling: each
class's signal is planted in a designated channel (e.g. class 1 in channel A, class 2
in channel B). Because ground-truth reliance is known by construction, this is how we
verify the conditional-reliance metric "isn't broken" before trusting it on real MRI.
An optional ``correlate`` leaks a fraction of a class's signal into a second channel,
so the *conditional* part of the metric (S3.1) has a legitimate correlate to discount.

Scope it exactly as the roadmap demands: a metric sanity check, not proof of real-MRI
correctness, and never a headline figure.

NumPy only (no torch); volumes can be returned as arrays or written as BraTS-style
NIfTI so the real pipeline can consume them unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..constants import CHANNEL_ORDER, MODALITY_SUFFIX, SEG_SUFFIX


def _ellipsoid_mask(shape: tuple[int, int, int], frac: float = 0.45) -> np.ndarray:
    """Centered ellipsoid 'brain' mask so background is exactly zero (skull-stripped)."""
    axes = [np.linspace(-1, 1, s) for s in shape]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    return (gx**2 + gy**2 + gz**2) <= (frac / 0.5) ** 2 * 0.25


def _sphere_mask(shape: tuple[int, int, int], center, radius: float) -> np.ndarray:
    ix, iy, iz = np.ogrid[: shape[0], : shape[1], : shape[2]]
    d2 = (ix - center[0]) ** 2 + (iy - center[1]) ** 2 + (iz - center[2]) ** 2
    return d2 <= radius**2


def synthetic_case(
    shape: tuple[int, int, int] = (48, 48, 48),
    n_channels: int = 4,
    class_channels: dict[int, int] | None = None,
    radius: int = 6,
    blob_intensity: float = 4.0,
    noise: float = 0.05,
    correlate: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(image (C,X,Y,Z) float32, seg (X,Y,Z) int)`` with known coupling.

    Args:
        class_channels: ``{label: primary_channel}``. Default ``{1: 0, 2: 1}``.
        radius: sphere radius in voxels for each planted class.
        blob_intensity: signal added to the primary channel inside the sphere.
        noise: std of Gaussian tissue noise inside the brain.
        correlate: in [0,1]; fraction of ``blob_intensity`` leaked into the next
            channel (a legitimate correlate to be discounted by conditional reliance).
        seed: RNG seed (deterministic).
    """
    if class_channels is None:
        class_channels = {1: 0, 2: 1}
    rng = np.random.default_rng(seed)

    image = np.zeros((n_channels, *shape), dtype=np.float32)
    seg = np.zeros(shape, dtype=np.int16)
    brain = _ellipsoid_mask(shape)

    # Baseline tissue signal in every channel within the brain.
    for c in range(n_channels):
        image[c][brain] = 1.0 + noise * rng.standard_normal(int(brain.sum()))

    # Clamp the sphere radius so it (and its placement margin) fit small test volumes.
    eff_radius = max(1, min(radius, min(shape) // 4))
    margin = eff_radius + 1
    for label, channel in class_channels.items():
        # Random center when there is room; otherwise place at the volume center.
        center = [int(rng.integers(margin, s - margin)) if s - margin > margin else s // 2
                  for s in shape]
        sphere = _sphere_mask(shape, center, eff_radius) & brain
        seg[sphere] = label
        image[channel][sphere] += blob_intensity
        if correlate > 0:
            secondary = (channel + 1) % n_channels
            image[secondary][sphere] += correlate * blob_intensity

    return image, seg


def save_case_nifti(case_dir: str | Path, image: np.ndarray, seg: np.ndarray) -> Path:
    """Write a synthetic case as BraTS-style NIfTI files (one per modality + seg)."""
    import nibabel as nib

    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    cid = case_dir.name
    affine = np.eye(4)
    for idx, modality in enumerate(CHANNEL_ORDER):
        suffix = MODALITY_SUFFIX[modality]
        nib.save(nib.Nifti1Image(image[idx], affine), case_dir / f"{cid}-{suffix}.nii")
    nib.save(nib.Nifti1Image(seg.astype(np.int16), affine), case_dir / f"{cid}-{SEG_SUFFIX}.nii")
    return case_dir


def generate_dataset(
    root: str | Path,
    n_cases: int = 8,
    prefix: str = "BraTS-GLI",
    seed: int = 0,
    **case_kwargs,
) -> list[Path]:
    """Generate ``n_cases`` synthetic cases on disk under ``root``; return case dirs."""
    root = Path(root)
    dirs: list[Path] = []
    for i in range(n_cases):
        cid = f"{prefix}-{i:05d}-000"
        image, seg = synthetic_case(seed=seed + i, **case_kwargs)
        dirs.append(save_case_nifti(root / cid, image, seg))
    return dirs
