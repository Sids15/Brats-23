"""Tests for the torch-free data layer: splits, preprocessing, synthetic generator."""
from __future__ import annotations

import numpy as np

from brats_trust.data import preprocess, splits, synthetic


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def _make_fake_cohort(root, n_patients=10, longitudinal_every=3):
    """Create empty case dirs; every Nth patient gets a second timepoint (-001)."""
    dirs = []
    for i in range(n_patients):
        for tp in range(2 if i % longitudinal_every == 0 else 1):
            d = root / f"BraTS-GLI-{i:05d}-{tp:03d}"
            d.mkdir(parents=True)
            dirs.append(d)
    return dirs


def test_make_splits_is_deterministic_and_leakage_free(tmp_path):
    _make_fake_cohort(tmp_path, n_patients=10)
    fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
    a = splits.make_splits(tmp_path, fractions, seed=42)
    b = splits.make_splits(tmp_path, fractions, seed=42)
    assert a["train"] == b["train"] and a["test"] == b["test"]  # deterministic

    # No patient id appears in more than one split (the leakage guarantee).
    patients_per_split = {
        s: {splits.patient_id(c) for c in a[s]} for s in ("train", "val", "test")
    }
    assert patients_per_split["train"].isdisjoint(patients_per_split["test"])
    assert patients_per_split["train"].isdisjoint(patients_per_split["val"])
    assert patients_per_split["val"].isdisjoint(patients_per_split["test"])

    # Every case is assigned exactly once.
    all_cases = a["train"] + a["val"] + a["test"]
    assert len(all_cases) == len(set(all_cases)) == a["meta"]["n_cases"]


def test_splits_persist_to_disk(tmp_path):
    _make_fake_cohort(tmp_path, n_patients=6)
    out = tmp_path / "splits.json"
    splits.make_splits(tmp_path, {"train": 0.5, "val": 0.25, "test": 0.25}, seed=1, out_path=out)
    loaded = splits.load_splits(out)
    assert loaded["meta"]["seed"] == 1


def test_validate_case(tmp_path):
    case = tmp_path / "BraTS-GLI-00000-000"
    img, seg = synthetic.synthetic_case(shape=(8, 8, 8))
    synthetic.save_case_nifti(case, img, seg)
    assert splits.validate_case(case)
    (case / "BraTS-GLI-00000-000-t1c.nii").unlink()
    assert not splits.validate_case(case)


# --------------------------------------------------------------------------- #
# Preprocess
# --------------------------------------------------------------------------- #
def test_brain_crop_removes_background(tmp_path):
    image = np.zeros((2, 6, 6, 6), dtype=np.float32)
    image[:, 2:4, 2:4, 2:4] = 3.0
    label = np.zeros((6, 6, 6), dtype=np.int16)
    label[2:4, 2:4, 2:4] = 1
    cropped_img, cropped_lbl = preprocess.brain_crop(image, label)
    assert cropped_img.shape == (2, 2, 2, 2)
    assert cropped_lbl.shape == (2, 2, 2)


def test_per_channel_zscore_normalizes_brain(tmp_path):
    rng = np.random.default_rng(0)
    image = np.zeros((1, 8, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[1:7, 1:7, 1:7] = True
    image[0][mask] = 5.0 + 2.0 * rng.standard_normal(int(mask.sum()))
    z = preprocess.per_channel_zscore(image)
    assert abs(z[0][mask].mean()) < 1e-5
    assert abs(z[0][mask].std() - 1.0) < 1e-2
    assert (z[0][~mask] == 0).all()  # background untouched


def test_intervene_mean_and_zero():
    image = np.zeros((2, 5, 5, 5), dtype=np.float32)
    image[:, 1:4, 1:4, 1:4] = np.arange(1, 28).reshape(3, 3, 3)
    mask = preprocess.brain_mask(image)

    mean_filled = preprocess.intervene(image, channel=0, fill="mean")
    expected = image[0][mask].mean()
    assert np.allclose(mean_filled[0][mask], expected)  # spatial info destroyed
    assert (mean_filled[0][~mask] == 0).all()
    assert np.array_equal(mean_filled[1], image[1])     # other channel untouched

    zero_filled = preprocess.intervene(image, channel=0, fill="zero")
    assert (zero_filled[0] == 0).all()


# --------------------------------------------------------------------------- #
# Synthetic
# --------------------------------------------------------------------------- #
def test_synthetic_case_plants_class_signal_in_designated_channel():
    img, seg = synthetic.synthetic_case(
        shape=(32, 32, 32), class_channels={1: 2, 2: 0}, seed=1, correlate=0.0
    )
    assert img.shape == (4, 32, 32, 32)
    assert set(np.unique(seg)) >= {0, 1, 2}
    # Class 1 lives in channel 2: its signal there exceeds its signal in channel 0.
    assert img[2][seg == 1].mean() > img[0][seg == 1].mean()


def test_generate_dataset_round_trips(tmp_path):
    import nibabel as nib

    dirs = synthetic.generate_dataset(tmp_path, n_cases=3, shape=(16, 16, 16), seed=0)
    assert len(dirs) == 3
    assert all(splits.validate_case(d) for d in dirs)
    vol = nib.load(str(dirs[0] / f"{dirs[0].name}-t1c.nii")).get_fdata()
    assert vol.shape == (16, 16, 16)
