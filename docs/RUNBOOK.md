# RUNBOOK — stage-by-stage operating manual

How to run BraTS-Trust on the GPU machine, what a healthy result looks like, and what to
send back if it doesn't. Stages map to `brain_tumor_roadmap_v4_final.md` §12.

## Setup (once, on the GPU machine)
```bash
git clone <repo> && cd Brats-23
python -m venv .venv && . .venv/Scripts/activate      # (Linux: source .venv/bin/activate)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA build for your driver
pip install -r requirements.txt && pip install -e .
pytest -q                                              # expect: all tests pass
```
Then set `data.root` in `configs/default.yaml` to your BraTS path
(e.g. `D:/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData`).

> **Tags:** [CPU-verified] = already proven on synthetic data here. [verify on GPU] =
> code complete but first real confirmation happens on your machine.

---

## Stage 0 — Foundation  [CPU-verified]
Nothing to run; it's the plumbing (config, splits, dataloader, model, metrics, logging).
**Healthy:** `pytest -q` is green.

## Stage 1 — Synthetic sanity check (§3.5)  [CPU-verified]
```bash
python scripts/run_synthetic_check.py
```
**Expected:** trains a tiny model on synthetic data and prints
`PASS: ET top reliance = T1CE`. The reliance for ET should be highest on **T1CE** (the
channel the signal was planted in), others much lower.
**If it FAILs:** send me the printed reliance table + `results/reliance_matrix.csv`. Likely
causes: too few epochs (raise `--epochs`) or a metric regression.

## Stage 1b — Dataset preflight (run FIRST on real data, §8/§9)
```bash
python scripts/preflight.py
```
**Expected:** JSON summary with `"n_cases": 1251`, `"n_failed": 0`, and
`outputs/preflight/manifest.csv`. Every case has 4 modalities + seg, consistent
shapes/affines, labels ⊆ {1,2,3}, no NaNs.
**If `n_failed` > 0:** send me `outputs/preflight/manifest.csv` (the `issues` column names
the problem per case). We decide whether to exclude or fix those cases.

## Stage 2 — Receptive-field sweep, THE DECISIVE STEP (§4, §4.1)  [orchestration CPU-verified]
First a baseline sanity train, then the sweep + analysis.
```bash
# (optional) one baseline model end-to-end on real data:
python scripts/train.py --name baseline_seed42
python scripts/evaluate.py --checkpoint runs/<that_run>/best_model.pt

# the sweep: 3 RF variants x >=5 seeds (set in configs/default.yaml: seeds_confirmatory)
python scripts/run_probe3.py --config configs/default.yaml --out outputs/probe3
python scripts/analyze_probe3.py --summary outputs/probe3/probe3_summary.jsonl
```
**Expected (baseline):** val Dice climbs into a sensible range (WT highest, ET lowest;
roughly WT ~0.85+, TC ~0.8, ET ~0.7 for a converged small U-Net — exact values vary).
**Expected (sweep):** `outputs/probe3/probe3_summary.csv` with one row per variant×seed
(ERF, faithfulness, val Dice), `erf_faithfulness_stats.json` (Spearman rho + p), and
`erf_vs_faithfulness.png`. The roadmap's hypothesis: **larger ERF → lower faithfulness**
(negative rho). A clean trend is the headline; a null (rho≈0, high p) is still a valid
finding — we'd soften "wrong reasons" to "reliance profiling" (§5).
**What to send me to decide next:** `probe3_summary.csv` + `erf_faithfulness_stats.json` +
the PNG. From the sign/strength of rho we choose: proceed to Stage 3 (trend holds) or
report the focused negative result (null).
**If a variant won't converge:** send its `runs/<run>/metrics.csv` (loss/Dice curves); we
apply the §4.2 protocol (report matched + tuned, exclude non-converged under preset rules).

---

## Stages 3–7 — appended as the code lands
(3 = other architectures + anchors, 4 = full reliance/fragility/XAI + stats, 5 =
characterize, 6 = optional fix, 7 = toolkit/Docker/write-up.) Each will get the same
Command / Expected / If-it-fails block here as it's built.
