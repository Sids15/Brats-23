"""Patient-level train/val/test splits, saved to disk with zero leakage.

Roadmap S9: splits are keyed by *patient* so longitudinal timepoints
(``BraTS-GLI-XXXXX-000`` / ``-001``) never straddle the train/test boundary --
otherwise the same anatomy leaks across splits and inflates scores. The split is
deterministic in ``seed`` so it can be regenerated and cited exactly in the paper.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

from ..constants import MODALITY_SUFFIX, SEG_SUFFIX


def discover_cases(root: str | Path, case_glob: str = "BraTS-GLI-*") -> list[Path]:
    """Return per-case directories under ``root`` (sorted for determinism).

    Discovery is lenient (matches directories by name); use :func:`validate_case`
    to assert a case has all expected modality + segmentation files.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob(case_glob) if p.is_dir())


def validate_case(case_dir: Path) -> bool:
    """True iff ``case_dir`` contains all 4 modality files and the seg file."""
    cid = case_dir.name
    needed = [f"{cid}-{suf}.nii" for suf in MODALITY_SUFFIX.values()]
    needed.append(f"{cid}-{SEG_SUFFIX}.nii")
    return all((case_dir / fname).exists() for fname in needed)


def patient_id(case: str | Path) -> str:
    """'BraTS-GLI-00008-001' -> 'BraTS-GLI-00008' (drop the timepoint suffix)."""
    name = case.name if isinstance(case, Path) else case
    return name.rsplit("-", 1)[0]


def make_splits(
    root: str | Path,
    fractions: dict[str, float],
    seed: int,
    out_path: str | Path | None = None,
) -> dict[str, object]:
    """Build deterministic, patient-grouped splits and optionally persist them.

    Args:
        root: dataset root containing the per-case directories.
        fractions: e.g. ``{"train": 0.8, "val": 0.1, "test": 0.1}`` (must sum to 1).
        seed: RNG seed; same seed + same cases -> identical split.
        out_path: if given, write the split JSON there.

    Returns a dict with a ``meta`` block and one case-id list per split. Raises if a
    patient appears in more than one split (the leakage guarantee, checked explicitly).
    """
    total = sum(fractions.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {total}")

    cases = discover_cases(root)
    case_names = [c.name for c in cases]

    # Group case ids by patient, then split at the patient level.
    by_patient: dict[str, list[str]] = {}
    for name in case_names:
        by_patient.setdefault(patient_id(name), []).append(name)

    patients = sorted(by_patient)
    random.Random(seed).shuffle(patients)

    n = len(patients)
    n_train = round(fractions.get("train", 0.0) * n)
    n_val = round(fractions.get("val", 0.0) * n)
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    bounds = {
        "train": patients[:n_train],
        "val": patients[n_train : n_train + n_val],
        "test": patients[n_train + n_val :],
    }

    splits: dict[str, list[str]] = {
        name: sorted(c for p in plist for c in by_patient[p]) for name, plist in bounds.items()
    }

    # Explicit leakage guard: no patient may appear in two splits.
    seen: dict[str, str] = {}
    for split_name, plist in bounds.items():
        for p in plist:
            if p in seen:
                raise RuntimeError(f"patient {p} in both {seen[p]} and {split_name}")
            seen[p] = split_name

    result: dict[str, object] = {
        "meta": {
            "seed": seed,
            "fractions": fractions,
            "n_patients": n,
            "n_cases": len(case_names),
            "root": str(root),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        **splits,
    }

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def load_splits(path: str | Path) -> dict[str, object]:
    """Load a previously saved splits JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
