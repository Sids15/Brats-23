"""Tests for the dataset preflight (built/tested on synthetic; run on real data via CLI)."""
from __future__ import annotations

import nibabel as nib
import numpy as np

from brats_trust import preflight
from brats_trust.data import synthetic


def test_preflight_passes_clean_synthetic_cohort(tmp_path):
    synthetic.generate_dataset(tmp_path, n_cases=4, shape=(16, 16, 16), seed=0)
    summary = preflight.run_preflight(tmp_path, out_dir=tmp_path / "report")
    assert summary["n_cases"] == 4
    assert summary["n_failed"] == 0
    assert (tmp_path / "report" / "manifest.csv").exists()


def test_preflight_flags_bad_label_and_missing_files(tmp_path):
    dirs = synthetic.generate_dataset(tmp_path, n_cases=2, shape=(16, 16, 16), seed=0)

    # Inject an out-of-range label (99) into the first case's seg.
    bad = dirs[0]
    seg_path = bad / f"{bad.name}-seg.nii"
    seg = np.asanyarray(nib.load(str(seg_path)).dataobj).astype(np.int16)
    seg[0, 0, 0] = 99
    nib.save(nib.Nifti1Image(seg, np.eye(4)), str(seg_path))

    # Remove a modality file from the second case.
    (dirs[1] / f"{dirs[1].name}-t1c.nii").unlink()

    summary = preflight.run_preflight(tmp_path)
    assert summary["n_failed"] == 2
    assert set(summary["failed_cases"]) == {d.name for d in dirs}
