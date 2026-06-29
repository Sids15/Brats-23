"""Dataset preflight logic (roadmap S8/S9) — importable core; CLI in scripts/preflight.py.

Validates the cohort before any GPU time: per case, checks all 4 modalities + seg present,
consistent shapes & affines across modalities, seg labels subset of {1,2,3} (BraTS-2023),
and finite values. Emits a manifest + summary. Built/tested against the synthetic generator.
"""
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from .constants import LABELS, MODALITY_SUFFIX, SEG_SUFFIX
from .data import splits
from .logging_utils import write_json, write_tidy

VALID_LABELS = set(LABELS)  # {1, 2, 3}


def check_case(case_dir: Path) -> dict:
    """Validate one case; return a manifest row with an ``ok`` flag and any ``issues``."""
    cid = case_dir.name
    issues: list[str] = []

    if not splits.validate_case(case_dir):
        return {"case_id": cid, "ok": False, "issues": "missing_files", "shape": None,
                "labels": None, "has_nonfinite": None}

    shapes, affines = [], []
    has_nonfinite = False
    for modality, suffix in MODALITY_SUFFIX.items():
        img = nib.load(str(case_dir / f"{cid}-{suffix}.nii"))
        shapes.append(tuple(img.shape))
        affines.append(np.asarray(img.affine))
        if not np.isfinite(np.asanyarray(img.dataobj, dtype=np.float32)).all():
            has_nonfinite = True
            issues.append(f"nonfinite:{modality}")

    if len(set(shapes)) != 1:
        issues.append(f"shape_mismatch:{set(shapes)}")
    if not all(np.allclose(a, affines[0]) for a in affines[1:]):
        issues.append("affine_mismatch")

    seg = np.asanyarray(nib.load(str(case_dir / f"{cid}-{SEG_SUFFIX}.nii")).dataobj)
    present = set(np.unique(seg).astype(int).tolist())
    extra = present - VALID_LABELS - {0}
    if extra:
        issues.append(f"unexpected_labels:{sorted(extra)}")

    return {
        "case_id": cid,
        "ok": not issues,
        "issues": ";".join(issues),
        "shape": str(shapes[0]),
        "labels": str(sorted(present)),
        "has_nonfinite": has_nonfinite,
    }


def run_preflight(root: str | Path, out_dir: str | Path | None = None) -> dict:
    """Validate every discovered case; optionally write manifest + summary; return summary."""
    cases = splits.discover_cases(root)
    rows = [check_case(c) for c in cases]
    n_ok = sum(r["ok"] for r in rows)
    summary = {
        "root": str(root),
        "n_cases": len(rows),
        "n_ok": n_ok,
        "n_failed": len(rows) - n_ok,
        "failed_cases": [r["case_id"] for r in rows if not r["ok"]],
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_tidy(out_dir / "manifest", rows)
        write_json(out_dir / "preflight_summary.json", summary)
    return summary
