# Project Rules — BraTS-Trust

Authoritative operating conventions for this repo. The goal: the code **and** the git
history should together form a reproducible, paper-ready record of the whole research
process. Read this before contributing.

## 1. Environment
- **Install only into the project `.venv`** — never globally. Create with
  `python -m venv .venv`; activate `./.venv/Scripts/activate` (Windows).
- Heavy CUDA-specific deps (`torch`, `monai`) are installed on the **training machine**
  matching its CUDA build (see https://pytorch.org); tooling-only deps here.
- Pin versions in `pyproject.toml`; exact resolved versions are recorded per run in
  `env.json` (`logging_utils.capture_environment`).
- Current dev tooling: `pyyaml`, `pytest`, `ruff`.

## 2. Naming
- **Cross-check every data/entity name against the canonical directory listing**
  `C:\Users\Kensyi15\Downloads\dir.md` before using it (BraTS case ids, modality
  suffixes `t1c/t1n/t2f/t2w/seg`, etc.).
- The channel order `[FLAIR, T1, T1CE, T2]` and label/region definitions are **frozen**
  in `src/brats_trust/constants.py`. Index modalities by position; never reorder.
- Run names: short, lowercase, descriptive (e.g. `probe3_rf_k5_seed42`). Run dirs are
  auto-prefixed with a UTC timestamp by `setup_run`.
- Branches: `dev` for ongoing work; merge to `main` at milestones; feature branches
  `stage0-dataloader`, `probe3-sweep`, etc. when useful.

## 3. Code quality (no AI slop)
- Type hints on public functions; **docstrings explain *why*, not just *what***.
- No filler comments, no dead code, no restating the obvious. Match surrounding style.
- Reuse existing utilities (config, constants, logging) instead of re-implementing.
- Lint before committing: `ruff check src scripts tests`.
- Tests stay import-light (torch-free where possible) so they run anywhere.

## 4. Logging & outputs (paper-ready, BraTS-2023)
- **Every experiment goes through `logging_utils.setup_run`** — no ad-hoc `print` for
  results. This guarantees a complete run directory (see `docs/ARCHITECTURE.md`).
- Capture *everything the paper could need*, the first time:
  reproducibility (`config.yaml`, `env.json` with git hash + dirty flag + GPU + seeds),
  training curves (`metrics.jsonl/.csv`), and tidy results under `results/`
  (`per_case_metrics`, `aggregate_metrics`, `reliance_matrix`, `fragility`, `erf`,
  `stats.json`). Use the canonical `*_COLUMNS` schemas in `logging_utils`.
- Always save **per-case** values (not just aggregates) so CIs, effect sizes, and box
  plots can be regenerated. Report Dice + HD95 per WT/TC/ET (+ lesion-wise slots).
- Set seeds via `set_seed`; record them. ≥3 seeds exploratory, ≥5 confirmatory (S4.2).

## 5. Documentation
- Keep `docs/ARCHITECTURE.md` current with the system; it seeds the paper's methods.
- Significant design decisions and their rationale belong in docs/commit messages, not
  only in code.

## 6. Git workflow
- **Commit frequently** in small, logical chunks — the history is part of the record.
- **Run the security audit before every commit:** `python scripts/security_audit.py`
  (staged) — it must exit 0 (no HIGH findings). Use `--all` for a full sweep.
- **Every commit carries the co-author trailer:**
  ```
  Co-Authored-By: rishabhahuja12 <rishabhahuja961@gmail.com>
  ```
- Never commit data, checkpoints, or run outputs (`.gitignore` covers `data/`, `*.nii`,
  `runs/`, `checkpoints/`, `.venv/`).

## 7. Security
- No secrets, API keys, tokens, or credentials in code or config — ever.
- No absolute user paths (`C:\Users\<name>\…`) committed; machine paths live in config
  `data.root` and are overridden per host.
- The security audit (`scripts/security_audit.py`) is the enforced gate; HIGH findings
  (private keys, cloud keys, secret literals) block commits.
