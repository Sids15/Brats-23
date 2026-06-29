# RUNBOOK — stage-by-stage operating manual

How to run BraTS-Trust on the GPU machine, what a healthy result looks like, and what to
send back if it doesn't. Stages map to `brain_tumor_roadmap_v4_final.md` §12.

# ============================================================================
# TEST SEQUENCE — run these 4 steps in order on the RTX 4500 Ada (Windows).
# After each step, paste back exactly what the "PASTE BACK" line asks for.
# ============================================================================

## Step 0 — Setup (once)
Run in **PowerShell** from where you want the repo:
```powershell
git clone -b dev https://github.com/Sids15/Brats-23.git
cd Brats-23
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .
pytest -q
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Notes: if `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope Process RemoteSigned`
first, or use `.venv\Scripts\activate.bat` in cmd. The `cu121` wheel works on the 4500 Ada;
if your driver is newer you can use `cu124` instead (see https://pytorch.org).
**EXPECT:** `pytest` ends with `43 passed, 1 skipped`; the last line prints
`True NVIDIA RTX 4500 Ada Generation`.
**PASTE BACK:** the `pytest` summary line **and** that `True NVIDIA ...` line.

## Step 1 — Pipeline check, NO data needed
```powershell
python scripts/run_synthetic_check.py
```
**EXPECT:** ends with `PASS: ET top reliance = T1CE` and a small reliance table.
**PASTE BACK:** the reliance table + the PASS/FAIL line.

## Step 2 — Dataset preflight (first touch of the real data)
Confirm `data.root` in `configs/default.yaml` is your BraTS path (currently
`D:/brats/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData` — edit it if different), then:
```powershell
python scripts/preflight.py
```
**EXPECT:** JSON with `"n_cases": 1251`, `"n_failed": 0`.
**PASTE BACK:** the whole JSON summary. If `n_failed` > 0, also send `outputs\preflight\manifest.csv`.

## Step 3 — Pipeline smoke on REAL data (the important one)
Trains on 24 cases for 3 epochs at the production `128^3` patch / `batch_size: 2`, then
evaluates (exercises preprocessing + dataloader + training + inference + reliance + fragility).
```powershell
python scripts/train.py --config configs/smoke.yaml --name smoke --limit 24

# grab the newest smoke run dir, then evaluate it:
$run = (Get-ChildItem runs -Directory -Filter *__smoke | Sort-Object Name)[-1].FullName
python scripts/evaluate.py --config configs/smoke.yaml --checkpoint "$run\best_model.pt" --split val
```
**Batch size:** `2` fits 24 GB with AMP for our scaffold + DynUNet. If you hit **CUDA OOM**,
edit `configs/smoke.yaml` → `train.batch_size: 1` (or `patch_size: [96, 96, 96]` and
`inference.roi_size: [96, 96, 96]`) and rerun; tell me which you used.
**EXPECT:** a training progress bar; `runs\<ts>__smoke\` containing `metrics.csv` (loss
trending down, a val `mean_dice`), `best_model.pt`, `env.json`; and the evaluate run's
`results\` with `reliance_matrix.csv` (12 rows) + `fragility.csv` (6 rows). Dice will be low
— that's expected (3 epochs / 24 cases); we're testing plumbing + GPU memory, not accuracy.
**PASTE BACK:** `runs\<ts>__smoke\metrics.csv`, `runs\<ts>__smoke\env.json`, and any
error/OOM traceback.

**If Steps 0–3 are all clean → I green-light full training + the Stage 2 sweep (below).**

---

> **Tags:** [CPU-verified] = already proven on synthetic data here. [verify on GPU] =
> code complete; first real confirmation happens on your machine.

---

> The 4 numbered steps above ARE the pre-flight (foundation, synthetic sanity check,
> preflight, real-data smoke). Once they're green, continue with the stages below.

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

## Stage 3 — Architecture sweep: Probe 1 + Tier-A anchors (§4 Probe 1, §5)  [MONAI anchors CPU-verified; SegMamba GPU-only]
Compares architectures (CNN / transformer / Mamba) under the *matched* protocol.
```bash
# one-time, for the Mamba arm (CUDA build required):
pip install mamba-ssm causal-conv1d

python scripts/run_probe1.py --config configs/default.yaml --out outputs/probe1
```
Models: `unet3d` (our scaffold), `dynunet` (nnU-Net architecture), `unetr`, `swin_unetr`,
`segmamba`. Restrict with `--models unet3d dynunet unetr`.
**Expected:** `outputs/probe1/probe1_summary.csv` — one row per architecture×seed with
faithfulness (WT/TC/ET + overall), ERF, and val Dice. If `mamba-ssm` is missing you'll see
`SKIP segmamba: ...` and the others still run.
**Caveat to keep in the writeup (§4):** an architecture swap changes receptive field +
optimization + inductive bias at once — report as "contribution under matched protocol",
never "causes".
**If `swin_unetr` errors on size:** it needs inputs divisible by 32 and ≥64³ (our 128³
patch is fine). **If a model OOMs on the GPU:** lower `train.batch_size` or `model.features`;
send me the error. **What to send me:** `probe1_summary.csv` (+ any failing run's
`metrics.csv`).

---

## Stage 4 — Full measurement: statistics + XAI-fails (§3, §3.4, §4.2)  [CPU-verified]
After the sweeps, aggregate across seeds and run the saliency check.
```bash
# pool seeds of one variant -> tables + significance (shell expands the glob):
python scripts/aggregate.py --runs runs/probe3_rf_small_seed* --out outputs/agg_rf_small

# saliency blindness check on a trained model:
python scripts/run_xai_check.py --checkpoint runs/<run>/best_model.pt
```
**Expected (aggregate):** under `--out`, `aggregate_segmentation.csv` (Dice/HD95 mean+CI per
region), `reliance_matrix.csv` (reliance averaged across seeds + CI), and
`fragility_gap.csv` with a `mean_gap` and **Holm-corrected `p_holm`** per region. A
**positive gap with small `p_holm`** = removing the leaned-on modality hurts more than
removing the physics one → consequential unfaithfulness (§3.3).
**Expected (XAI):** `mean_saliency_cosine` is **high (≈0.8–1.0)** even after T1CE is
mean-filled → saliency barely moves → it's blind to the modality shortcut, justifying the
intervention-first stack (§3.4). `results/msfi.csv` reports the saliency share on the
physics modality (convergent check).
**What to send me:** `outputs/agg_*/summary.json` + `fragility_gap.csv`, and the XAI
`mean_saliency_cosine`. If the fragility gap is **not** positive/significant, that's the
signal to soften "wrong reasons" → "modality-reliance profiling" (§5) — a valid outcome.

---

## Stage 5 — Characterize + summary figure (§4.1, §5, §11)  [CPU-verified]
Pull the headline numbers together into one panel.
```bash
python scripts/analyze_probe3.py --summary outputs/probe3/probe3_summary.jsonl   # the curve + Spearman
python scripts/make_summary_figure.py \
    --reliance outputs/agg_rf_small/reliance_matrix.jsonl \
    --probe3   outputs/probe3/probe3_summary.jsonl \
    --fragility outputs/agg_rf_small/fragility_gap.jsonl \
    --xai      runs/<xai_run>/results/xai_fails.jsonl \
    --out      outputs/summary_figure.png
```
**Expected:** `outputs/summary_figure.png` — a 2×2 panel (reliance heatmap · ERF↔faithfulness
scatter with Spearman ρ · fragility gap · saliency-cosine histogram). Any panel whose input
is omitted shows a "no data" placeholder, so you can build it up as stages finish.
**Reading the result (§5):** a clean **negative ERF↔faithfulness trend** is the headline; a
**positive fragility gap** confirms the consequence; **high XAI cosine** confirms saliency is
blind. If the ERF trend is null (ρ≈0), report it honestly as a focused negative result and
soften "wrong reasons" → "modality-reliance profiling".
**What to send me:** `summary_figure.png` + `erf_faithfulness_stats.json`.

---

## Stage 6 — Optional modality-dropout fix (§6)  [CPU-verified]
Trains baseline vs the fix (randomly mean-fill a modality in the last 20% of training) and
compares. Enable via `configs/default.yaml` → `train.modality_dropout.enabled: true`, or use:
```bash
python scripts/run_fix.py --config configs/default.yaml --out outputs/fix
```
**Expected:** `outputs/fix/fix_comparison.json` with `delta_faithfulness` and
`delta_val_dice`. The roadmap's evidence bar (§6): **faithfulness improves** (`delta_faithfulness
> 0`) while **Dice holds** (`delta_val_dice ≈ 0` or better); also check the fragility gap
shrinks via `scripts/aggregate.py` on the two run sets. We report what we measure — the fix
may just even out reliance.
**What to send me:** `fix_comparison.json` (+ aggregated `fragility_gap.csv` for both).

## Stage 7 — Toolkit + write-up — appended as the code lands
(Packaging the repo as a reusable toolkit + Docker/MLCube; the paper draft.) Added here with
Command / Expected / If-it-fails when built.
