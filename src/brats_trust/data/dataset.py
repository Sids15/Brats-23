"""Ablation-capable dataset/dataloader (roadmap S9, S3.1).

Builds 4-channel ``[FLAIR, T1, T1CE, T2]`` inputs (frozen order) and overlapping
``[WT, TC, ET]`` targets from BraTS-2023 cases, via a MONAI transform pipeline. The
*ablation* (modality intervention) is applied at evaluation time by the reliance/
fragility metrics, not here -- training always sees full modalities.

Notes:
- Channel order comes from ``constants.CHANNEL_ORDER`` -> file suffixes; the image key is
  a *list* of the four modality paths so MONAI stacks them in that exact order.
- Labels are converted from BraTS-2023 values ``{1:NCR, 2:ED, 3:ET}`` to the overlapping
  regions in ``constants.REGIONS`` (note: 2023 uses label 3 for ET, unlike MONAI's
  built-in BraTS converter which assumes the legacy label 4).
"""
from __future__ import annotations

from pathlib import Path

import torch
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    NormalizeIntensityd,
    RandFlipd,
    RandSpatialCropd,
    SpatialPadd,
)

from ..constants import CHANNEL_ORDER, MODALITY_SUFFIX, REGION_ORDER, REGIONS, SEG_SUFFIX


class ConvertBraTS2023Labelsd(MapTransform):
    """Map a single-channel BraTS-2023 seg into a 3-channel overlapping WT/TC/ET mask."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lbl = torch.as_tensor(d[key])
            if lbl.ndim == 4 and lbl.shape[0] == 1:  # drop channel dim if present
                lbl = lbl[0]
            channels = [
                torch.isin(lbl, torch.tensor(REGIONS[r], dtype=lbl.dtype)) for r in REGION_ORDER
            ]
            d[key] = torch.stack(channels, dim=0).to(torch.float32)
        return d


def case_to_dict(case_dir: str | Path) -> dict[str, object]:
    """Map a case directory to ``{'image': [4 modality paths], 'label': seg path}``."""
    case_dir = Path(case_dir)
    cid = case_dir.name
    image = [str(case_dir / f"{cid}-{MODALITY_SUFFIX[m]}.nii") for m in CHANNEL_ORDER]
    label = str(case_dir / f"{cid}-{SEG_SUFFIX}.nii")
    return {"image": image, "label": label}


def build_transforms(cfg, train: bool) -> Compose:
    """Compose the load/convert/crop/normalize (+ train patch & flip) pipeline."""
    keys = ["image", "label"]
    patch = tuple(cfg.train.patch_size)
    tfms: list = [
        LoadImaged(keys, ensure_channel_first=True, image_only=True),
        ConvertBraTS2023Labelsd("label"),
        CropForegroundd(keys, source_key="image", allow_smaller=True),
        NormalizeIntensityd("image", nonzero=True, channel_wise=True),
    ]
    if train:
        tfms += [
            SpatialPadd(keys, spatial_size=patch),       # guarantee >= patch before cropping
            RandSpatialCropd(keys, roi_size=patch, random_size=False),
            RandFlipd(keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys, prob=0.5, spatial_axis=2),
            EnsureTyped(keys),
        ]
    else:
        tfms += [EnsureTyped(keys)]
    return Compose(tfms)


def make_dataloader(
    case_dirs,
    cfg,
    train: bool,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Build a DataLoader over the given case directories.

    ``num_workers`` defaults to 0 (safe for tests / Windows); pass
    ``cfg.train.num_workers`` for real training.
    """
    data = [case_to_dict(d) for d in case_dirs]
    ds = Dataset(data, transform=build_transforms(cfg, train))
    bs = batch_size if batch_size is not None else cfg.train.batch_size
    return DataLoader(ds, batch_size=bs, shuffle=train, num_workers=num_workers)
