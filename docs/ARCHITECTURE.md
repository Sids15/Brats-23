# Architecture — BraTS-Trust

This document describes the system as built, and is written to double as scaffolding for
the paper's *Methods* and *Reproducibility* sections. It tracks the frozen design in
[`../brain_tumor_roadmap_v4_final.md`](../brain_tumor_roadmap_v4_final.md); section
references like "S3.1" point there.

## 1. Research goal (one paragraph)
A segmentation model can be **right for the wrong reasons**: produce a correct mask while
its decision is not sensitive to the information *uniquely* carried by the modality that
physically images that class, leaning instead on correlated modalities in a way that
breaks under perturbation. BraTS-Trust **measures** this with an intervention-based,
physics-grounded, *conditional* reliance metric (S3.1), **characterizes** how it varies
with effective receptive field (S4.1), and **demonstrates the consequence** via comparative
missing-modality fragility (S3.3). Contribution = methodology + controlled characterization
+ demonstrated consequence (+ an optional fix). Cohort: **BraTS-2023 adult glioma (GLI)**.

## 2. Data & frozen conventions
- **Cohort:** ASNR-MICCAI BraTS-2023-GLI, pre-treatment, 1251 cases. One case per
  directory `BraTS-GLI-XXXXX-YYY/` with uncompressed `.nii` files. Patient id is the
  `BraTS-GLI-XXXXX` prefix (timepoints `-000/-001` belong to the same patient → grouped
  in splits to prevent leakage, S9).
- **Channel order (frozen, `constants.CHANNEL_ORDER`):** `[FLAIR, T1, T1CE, T2]` mapping
  to file suffixes `t2f, t1n, t1c, t2w`. Modalities are indexed by *position*; the
  ablation dataloader and every metric rely on this never changing.
- **Labels (`constants.LABELS`, BraTS-2023 seg values):** `1=NCR, 2=ED, 3=ET`.
- **Regions (`constants.REGIONS`, overlapping output channels):**
  `WT={1,2,3}, TC={1,3}, ET={3}`.
- **Preprocessing (S9):** brain-crop to nonzero bbox, per-channel z-score over nonzero
  voxels. BraTS is already 1 mm isotropic and co-registered.
- **Physics answer key (`physics_answer_key.json`):** documents which modality physically
  images each class (ET→T1CE, TC→T1CE, WT→FLAIR) and the *legitimate* correlates. It
  **informs** the conditional test and is **never** a score or penalty (CUT LIST).

## 3. Module map
```
src/brats_trust/
  constants.py        Frozen channel order, labels, regions, suffix map.
  config.py           YAML loader; default.yaml (S9 protocol) + deep-merged overrides.
                      Config is attribute-accessible and round-trips to dict for logging.
  logging_utils.py    Reproducibility + research-grade run logging (see S5 below).
  physics_answer_key.json   Physics expectation (documentation artifact).
  engine.py           Train loop + sliding-window inference (AMP on CUDA).     [done]
  pipeline.py         evaluate_and_log: writes all paper-ready tidy outputs.   [done]
  preflight.py        Dataset validation core (shapes/affines/labels/finite).  [done]
  data/
    splits.py         Patient-level, leakage-free train/val/test splits.       [done]
    preprocess.py     Brain-crop, per-channel z-score, modality intervention.  [done]
    synthetic.py      S3.5 calibration generator (known class->channel coupling).[done]
    dataset.py        Ablation-capable MONAI loader (4ch in, WT/TC/ET out).    [done]
  models/
    scaffold.py       Shared 3D U-Net; pluggable conv/dwsep block (S4 Probe 3).[done]
  metrics/
    stats.py          Bootstrap CIs, effect sizes, Holm correction (S3, S4.2). [done]
    segmentation.py   Per-region Dice + HD95 + sens/spec (BraTS-2023).         [done]
    reliance.py       Conditional intervention reliance, per-case + aggregate. [done]
    fragility.py      CONSEQUENCE: comparative missing-modality fragility (S3.3).[done]
scripts/
  train.py            Train the scaffold (CUDA or CPU).
  evaluate.py         Evaluate a checkpoint -> all tidy results.
  preflight.py        Validate the real dataset before training (run first).
  run_synthetic_check.py  S3.5 end-to-end sanity check (CPU, no real data).
  make_splits.py      CLI wrapper around data.splits.
  security_audit.py   Pre-commit secret/PII scan (enforced gate).
configs/default.yaml  The frozen S9 global protocol.
```
`[done]` = implemented + tested. The whole chain (load -> train -> infer -> reliance ->
fragility -> tidy outputs) is verified end-to-end on CPU with synthetic data
(`scripts/run_synthetic_check.py`, `tests/test_pipeline.py`); the GPU only adds scale.

## 4. Configuration
`configs/default.yaml` encodes the non-negotiable S9 protocol (cohort, channel order,
preprocessing, regions, training/inference settings, ablation fill, seed budgets, stats).
`config.load_config(*overrides)` deep-merges experiment YAMLs over it, so an experiment
records exactly which knobs it changed. The resolved config is snapshotted into each run.

## 5. Logging & reproducibility (`logging_utils.py`)
The spine of "paper-ready outputs". `setup_run(name, cfg)` creates:
```
runs/<UTC-timestamp>__<name>/
  config.yaml        resolved config actually used
  env.json           python/OS/host, package versions, GPU (name+VRAM), CUDA,
                     git commit + dirty flag, seeds, command line
  run.log            full ISO-timestamped console log
  metrics.jsonl/.csv per-step training/val curves (convergence diagnostics, S4.2)
  results/
    per_case_metrics.{csv,jsonl}   case × region × {dice, hd95, sens, spec,
                                   lesionwise_dice, lesionwise_hd95}   (BraTS-2023)
    aggregate_metrics.{csv,jsonl}  region × {n, mean, std, median, IQR, 95% CI}
    reliance_matrix.{csv,jsonl}    region × modality × {score, fill=mean|zero}  (S3.1)
    fragility.{csv,jsonl}          removed-modality drop + leaned-on vs physics gap (S3.3)
    erf.{csv,jsonl}                variant × effective receptive field           (S4.1)
    stats.json                     effect sizes, CIs, corrected p-values      (S3/S4.2)
  run_summary.json   timings, GPU count, GPU-hours (from finalize())
```
Design principles: **artifacts are written eagerly** (config/env at start, metrics flushed
per row) so a crashed run still yields a usable record; **per-case values are always
saved** so CIs/effect sizes/box-plots regenerate from disk; **tidy schemas** (the
`*_COLUMNS` constants) keep cross-run aggregation trivial. Reproducibility primitives:
`set_seed` (random/numpy/torch + deterministic cuDNN) and `get_git_revision` (commit +
dirty flag).

## 6. Measurement stack (how metrics map to the run dir)
- **Conditional reliance (PRIMARY, S3.1)** → `results/reliance_matrix.*`. Per class ×
  modality, prediction change under single-modality intervention (**mean-fill**, with a
  zero-fill sensitivity row) *conditional on the others present*. Honest proxy for the
  unique-information quantity; the rigorous conditional-MI/PID version is Fork-B (S10).
- **Comparative fragility (CONSEQUENCE, S3.3)** → `results/fragility.*`. Compare Dice drop
  from removing the leaned-on vs the physics-correct modality; the *gap* (not a generic
  drop) is the evidence for "wrong reasons".
- **Segmentation quality** → `per_case_metrics.*` + `aggregate_metrics.*` (Dice + HD95 per
  WT/TC/ET, BraTS-2023 lesion-wise reserved).
- **ERF↔faithfulness (S4.1)** → `erf.*` + `stats.json` (Spearman). The money figure.
Faithfulness (reasoning, S3.1) and fragility (robustness, S3.3) are reported as **two
separate axes**, never collapsed into a single "trust score" (CUT LIST).

## 7. Experiment lifecycle
`load_config(overrides)` → `setup_run(name, cfg, set_global_seed=…)` → build splits/
dataloader/model → train, logging curves via `ctx.metrics.log(...)` → evaluate, writing
tidy results via `write_tidy(ctx.result("per_case_metrics"), rows, PER_CASE_COLUMNS)` →
`ctx.finalize()`. Aggregation/plotting scripts read `runs/*/results/*.jsonl` across seeds.

## 8. Build order (S12)
0. Protocol, ablation dataloader, scaffold, metric, physics key  ← *foundation + tooling done*
1. Synthetic sanity check (S3.5)
2. **Probe 3 RF sweep — DECISIVE** (does effective RF predict faithfulness?)
3. If yes → Probe 1/2 + Tier-A anchors
4. Reliance matrices, comparative fragility, XAI-fails, statistics
5. Characterize ERF↔faithfulness + consequence
6. Optional: modality-dropout fix, probing, gate
7. Toolkit + Docker; write-up (Fork A: MICCAI/MIDL/iMIMIC)
