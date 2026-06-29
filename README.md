# BraTS-Trust

Intervention-based **conditional modality-reliance** ("faithfulness") measurement
for 3D brain-tumor segmentation. Implements the design frozen in
[`brain_tumor_roadmap_v4_final.md`](./brain_tumor_roadmap_v4_final.md).

**One-line thesis:** a model can be *right for the wrong reasons* — produce a
correct mask while its decision is not sensitive to the information *uniquely*
carried by the modality that physically images that class — and that mismatch
can be measured, characterized across receptive-field conditions, and shown to
have a consequence.

> Status: **skeleton**. Config, constants, and the physics key are real; data /
> models / metrics are interface stubs. See "Build order" below.

## Dataset

ASNR-MICCAI **BraTS-2023 Adult Glioma (GLI)**, pre-treatment — the single
cohort the roadmap anchors on (S8). 1251 cases, each a directory:

```
BraTS-GLI-XXXXX-YYY/
  BraTS-GLI-XXXXX-YYY-t2f.nii   # FLAIR
  BraTS-GLI-XXXXX-YYY-t1n.nii   # T1
  BraTS-GLI-XXXXX-YYY-t1c.nii   # T1CE
  BraTS-GLI-XXXXX-YYY-t2w.nii   # T2
  BraTS-GLI-XXXXX-YYY-seg.nii   # labels: 1=NCR, 2=ED, 3=ET
```

Files are uncompressed `.nii`. Frozen channel order is `[FLAIR, T1, T1CE, T2]`
(`constants.CHANNEL_ORDER`); evaluation regions are overlapping `WT / TC / ET`.
Set the data path in `configs/default.yaml` (`data.root`) per machine.

## Layout

```
configs/default.yaml          # frozen S9 global protocol (override-able)
src/brats_trust/
  constants.py                # frozen channel order, labels, regions
  config.py                   # YAML loader w/ deep-merge overrides
  physics_answer_key.json     # documents physics expectation; NEVER a penalty
  data/   splits, dataset (ablation-capable, mean-fill), preprocess  [stubs]
  models/ shared U-Net scaffold + pluggable block                   [stubs]
  metrics/ reliance (primary), fragility (consequence)              [stubs]
scripts/make_splits.py
tests/test_smoke.py
```

## Conventions & docs

- [`rules.md`](./rules.md) — operating rules (env, naming, logging, git, security).
- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — system architecture + run-dir schema
  (also seeds the paper's Methods/Reproducibility sections).

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows Git Bash
# install torch matching your CUDA from https://pytorch.org first, then:
pip install -e .
pytest -q                                            # skeleton smoke tests
```

Training target: RTX 4500 Ada (24 GB). `train.amp` is on; `patch_size`
`128^3` fits comfortably.

## Build order (roadmap S12)

0. Protocol, ablation dataloader, scaffold, metric, physics key  ← *skeleton done; logic next*
1. Synthetic sanity check (S3.5)
2. **Probe 3 RF sweep — DECISIVE** (does effective RF predict faithfulness?)
3. If yes → Probe 1/2 + Tier-A anchors
4. Reliance matrices, comparative fragility, XAI-fails, statistics
5. Characterize ERF↔faithfulness + consequence
6. Optional: modality-dropout fix, probing, gate
7. Toolkit + Docker; write-up (Fork A: MICCAI/MIDL/iMIMIC)

## Non-negotiables (so they don't drift)

- Intervention fill is **mean / healthy-prior, not zero** (S3.1).
- The physics key **documents and informs**; it is never a score or penalty (CUT LIST).
- Faithfulness (reasoning) and fragility (robustness) are **two separate axes** (S3).
- One cohort only; no multi-task pooling (S8, CUT LIST).
