"""Ablation-capable dataset/dataloader.

The single most load-bearing data component (roadmap S9: 'ablation-capable
dataloader (mean-fill channel k) from day one'). It must be able to return a
sample with any subset of modality channels *intervened on* (mean-fill /
healthy-prior fill, NOT zero by default -- roadmap S3.1) while keeping the
frozen channel order intact.

STUB: interface only. To be implemented in Stage 0 on top of MONAI transforms.
"""
from __future__ import annotations

from pathlib import Path

from ..constants import CHANNEL_ORDER


class BraTSDataset:
    """4-channel [FLAIR, T1, T1CE, T2] volumes + overlapping WT/TC/ET labels.

    Args (planned):
        case_dirs: list of per-case directories (from splits).
        ablate: optional set/list of modality names or channel indices to fill.
        fill: 'mean' | 'healthy_prior' | 'zero' (roadmap S3.1; default mean).
        patch_size / transforms: standard MONAI pipeline.
    """

    def __init__(
        self,
        case_dirs: list[Path],
        ablate: list[str] | list[int] | None = None,
        fill: str = "mean",
    ) -> None:
        self.case_dirs = list(case_dirs)
        self.ablate = ablate
        self.fill = fill
        assert len(CHANNEL_ORDER) == 4
        # TODO(Stage 0): build MONAI LoadImage/Crop/Normalize/Patch transforms;
        # implement label one-hot into WT/TC/ET; implement channel intervention.

    def __len__(self) -> int:  # pragma: no cover - stub
        raise NotImplementedError("Stage 0")

    def __getitem__(self, idx: int):  # pragma: no cover - stub
        raise NotImplementedError("Stage 0")
