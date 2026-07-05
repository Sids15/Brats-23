# BraTS-Trust

Intervention-based **conditional modality-reliance** ("faithfulness") measurement
for 3D brain-tumor segmentation. Implements the design frozen in
[`brain_tumor_roadmap_v4_final.md`](./brain_tumor_roadmap_v4_final.md).

**One-line thesis:** a model can be *right for the wrong reasons* — produce a
correct mask while its decision is not sensitive to the information *uniquely*
carried by the modality that physically images that class — and that mismatch
can be measured, characterized across receptive-field conditions, and shown to
have a consequence.

> Status: Stages 0–1 complete and verified on GPU — the full test suite passes,
> dataset preflight is clean on all 1251 cases, and the real-data pipeline smoke
> (train → inference → reliance → fragility) runs end-to-end. The receptive-field
> sweep (Stage 2) is next. See "Build order" below.

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
  data/    splits, dataset (ablation-capable, mean-fill), preprocess, synthetic
  models/  base + one module per architecture: unet3d (scaffold/RF-sweep), dynunet, unetr, swin_unetr, segmamba
  metrics/ reliance (primary), fragility (consequence), ERF, faithfulness, stats, XAI
scripts/   preflight, train, evaluate, sweeps (probe1/probe3), fix, figures, security_audit
tests/     full suite: data, metrics, pipeline, architectures, figures, stage4/6, ...
```

## Conventions & docs

- [`rules.md`](./rules.md) — operating rules (env, naming, logging, git, security).
- [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md) — system architecture + run-dir schema
  (also seeds the paper's Methods/Reproducibility sections).
- [`docs/PIPELINE.md`](./docs/PIPELINE.md) — end-to-end pipeline diagram + Methods prose
  (preprocess → splits → matched-protocol training → metrics → figures).
- [`docs/RUNBOOK.md`](./docs/RUNBOOK.md) — stage-by-stage commands, expected results, and
  what to send back if a result looks wrong.

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate    # Windows Git Bash
# install torch matching your CUDA from https://pytorch.org first, then:
pip install -e .
pytest -q                                            # full test suite
```

Training target: RTX 4500 Ada (24 GB). `train.amp` is on; `patch_size`
`128^3` fits comfortably.

## Running

The whole pipeline is verified end-to-end on CPU with synthetic data, so on the GPU
machine it's plug-and-play. Set `data.root` in `configs/default.yaml`, then:

```bash
python scripts/preflight.py                       # 1. validate the dataset (run first)
python scripts/train.py --name baseline_seed42    # 2. train (CUDA + AMP auto-detected)
python scripts/evaluate.py --checkpoint runs/<run>/best_model.pt   # 3. paper-ready results

python scripts/run_synthetic_check.py             # S3.5 sanity check (CPU, no real data)
```

Each run writes a timestamped `runs/<ts>__<name>/` with config snapshot, env, curves, and
tidy `results/` (per-case + aggregate metrics, reliance matrix, fragility) — see
`docs/ARCHITECTURE.md`.

## Build order (roadmap S12)

0. Protocol, ablation dataloader, scaffold, metric, physics key  ← *done (CPU-verified)*
1. Synthetic sanity check (S3.5)  ← *done: `scripts/run_synthetic_check.py` passes*
2. **Probe 3 RF sweep — DECISIVE** (does effective RF predict faithfulness?)  ← *next, on GPU*
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
