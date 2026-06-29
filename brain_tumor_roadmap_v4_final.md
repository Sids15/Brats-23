# BraTS-Trust — Faithful Modality Reliance (Paper 1)
## Master Roadmap v4 — FINAL BUILD DOC (design frozen)

> This supersedes both my v3 and the uploaded "deep-research v4" report. It keeps **v3's science** as the frozen core, adds the **execution layer** (timeline, compute, packaging, ERF engineering) the research report got right, folds in the **accepted late additions** (modality-dropout fix, XAI-fails check, physics key as artifact, faithfulness/fragility as two axes), and **permanently excludes** the three errors that keep returning. After this: build, don't revise.

---

## CUT LIST — do not re-add (these have hitched a ride three times)
- **Trust Score** (`Dice − λ1·Fragility − λ2·(1−Faithfulness) − …`): unprincipled weighted sum with free λ's; reviewer catnip; unnecessary for a diagnostic paper. Report faithfulness and fragility as separate interpretable quantities.
- **Physics-key-as-penalty** (L2 distance to a reliance key): this *is* the naive "faithful = matches T1CE" definition already rejected; it punishes legitimate correlated inference. The key documents and *informs* the test; it never *scores* it.
- **Multi-task expansion** (adult + peds + meningioma + mets leaderboard): scope creep, and the tasks don't share label schemas or even modality counts (meningioma-RT is single-modality T1c; post-treatment adds resection cavity). One cohort only.
- **Naive single-modality ablation as a wrongness test**, and **uncaveated zero-fill**.

---

## 0. Thesis
A model can be **right for the wrong reasons**: produce a correct mask while its decision is *not* sensitive to the information **uniquely** carried by the modality that physically images that class's signal — leaning instead on correlated modalities in a way that **breaks under perturbation**. We define and measure this with an intervention-based, physics-grounded, *conditional* reliance metric, characterize how it varies across controlled architectural conditions (receptive field), and demonstrate the consequence.

Contribution = **measurement methodology + controlled characterization + demonstrated consequence (+ a simple fix)**. Not a new architecture; not "explanations exist"; not (this version) a causal theory of inductive bias.

---

## 1. Conditional faithfulness (the load-bearing definition — never weaken it)
**Faithful** = sensitivity consistent with acquisition physics **after controlling for correlated modalities** — the decision responds to information *uniquely* available in the signal-bearing modality, not recoverable from the others. A model legitimately inferring enhancement from edema/mass-effect/morphology is **not** cheating; low single-modality dependence alone is **not** wrongness.

**Wrong reasons (operational)** = correct Dice for a class **+** the decision does not use the unique signal of the physics-correct modality **+** a **measured consequence** (§3.3). Without the consequence, language softens to **"modality-reliance profiling."**

The intervention-based measures in §3 are honest **proxies** for the conditional/unique-information quantity. The fully rigorous version (conditional MI / partial information decomposition) is the Fork-B project (§10), not a bolt-on.

---

## 2. What's adopted vs contributed
**Adopt & cite (never claim):** MSFI (modality faithfulness metric, BraTS, classification); cross-modal architectures w/ ablations; missing-modality robustness; shortcut learning (classification).
**Contribute:** intervention-based **conditional** reliance for *segmentation*, as the dependent variable across **controlled** RF conditions, with a **comparative fragility consequence**, an **ERF↔faithfulness** relationship, and a packaged **toolkit**.

---

## 3. Measurement stack

**3.1 PRIMARY — conditional reliance (intervention).** Per class (ET/TC/WT) × modality, measure prediction change under intervention on a modality **conditional on the others present**. Report the per-class × per-modality **reliance matrix**. Intervention = **mean-fill / healthy-tissue-prior fill, not zeros**; report sensitivity to the fill choice (zero-fill is OOD and partly measures shock, not reliance).

**3.2 SECONDARY — MSFI (adopted).** Saliency-quality check only: does a saliency method localize modality-specific features at all? Not the main result.

**3.3 CONSEQUENCE — comparative missing-modality fragility (MANDATORY for "wrong reasons").** Remove the modality the model **wrongly leans on** vs the **physics-correct** one; compare Dice drop. **If leaning on the correlate makes it break *harder*, the unfaithful reliance is consequential.** This is the proof, no human study. Preserve the *comparative* structure — a generic "drop when missing" number does not prove wrongness.

**3.4 XAI-fails (cheap, appendix).** One model: run Grad-CAM, mean-fill T1CE, run again; show the heatmap barely changes → saliency is blind to modality-level shortcuts. Justifies the intervention-first stack; doubly supported by MSFI's published finding. ~1–2 days with `pytorch-grad-cam`.

**3.5 CALIBRATION — sanity check only (demoted).** 2-channel synthetic (sphere of class 1 in A, class 2 in B), tiny U-Net, verify the metric flags A as essential for class 1. ~3 days. Scope explicitly as "the metric isn't broken," **not** proof of real-MRI correctness, **not** a lead figure.

**Two named axes (concept kept, fill-based operationalization rejected):**
- *Faithfulness* = correctness of reasoning (§3.1, conditional intervention).
- *Fragility* = robustness to deployment (a scanner/modality missing).
Distinguish them by the **question each answers**, and keep §3.3's comparative link — not by "mean-fill vs zero-fill," which a reviewer will call the same operation twice.

All comparisons: **effect sizes + CIs + multiple-comparison-corrected p-values**, across seeds.

---

## 4. Tier B — the core (controlled scaffold + probes)
Single shared U-Net scaffold (matched stages/widths/skips/patch/optimizer/schedule/init/data/seeds); one pluggable block.

| Probe | Variable | Status | Note |
|---|---|---|---|
| **Probe 3 — RF sweep** | Effective receptive field *within* conv family (kernel/depth), via **depthwise-separable / bottleneck convs so params & FLOPs stay fixed** | **PRIMARY** | The one genuinely controllable swap → the **ERF↔faithfulness curve** (§4.1) |
| **Probe 1 — Mechanism swap** | Encoder: conv vs attention vs Mamba | **SECONDARY, caveated** | Changes RF + optimization + inductive bias + normalization *simultaneously*; never single-variable. Word as "isolates contribution under matched optimization," **never** "localizes" / "causes" |
| **Probe 2 — Decoder swap** | Decoder only, identical conv encoder | **CONTROL** | If reliance barely moves, effect sits in the encoder |

**4.1 Money figure — ERF vs faithfulness.** Empirically measure effective RF (gradient-based) per variant; plot faithfulness vs effective RF; Spearman test. A clean "larger ERF → lower faithful reliance" curve is the memorable result — reported as a controlled empirical relationship, **not** a causal law. If null, report the null honestly (still publishable as a focused negative result).

**4.2 Rigor protocol (every probe).** Match optimizer/schedule/init/data/seeds; if a variant needs different hyperparameters to converge fairly, report **both** matched and tuned runs; exclude non-converged runs under **pre-registered** criteria; report convergence curves + measured ERF. **≥5 seeds for the confirmatory Probe 3 sweep**; ≥3 elsewhere.

---

## 5. Tier A — minimal context (NOT the result)
Three off-the-shelf anchors, global protocol, no tuning: one CNN (nnU-Net/ResUNet), one transformer (UNETR/Swin-UNETR), one Mamba (SegMamba). Show the phenomenon exists in real models; draw **no** localization conclusions from them. Keep tiny.

---

## 6. Optional — only if it earns decisive triangulation
- **Modality-dropout fix** (the *acceptable* fix, optional, last). Last 20% of training / fine-tune: randomly mean-fill one modality per batch (~25%). **Evidence = the §3.3 fragility gap shrinks AND §3.1 faithfulness improves while full-modality Dice holds.** Do **not** claim it "forces faithful representations" by fiat — it's a robustness trick that may just even out reliance; *measure* whether faithfulness actually improves. (Audit+fix materially helps at MICCAI/MIDL.)
- **Linear concept probing** — 1-layer probe on frozen bottleneck features ("is T1CE present?"). Triangulation only; limit stated: decodable ≠ used.
- **Readable fusion gate** — keep *iff* its learned reliance cross-checks §3.1 (agree = convergent validity; disagree = a finding). Else cut.

---

## 7. Rejected ideas (logged so they don't return)
Sequence-permutation "Mamba shortcut" audit (false premise — CNNs/transformers aren't channel-order invariant; permuting degrades all models). Gradient-penalty fix (hardcodes the too-strong prior). A "theorem" for wrong-reasons (near-tautological; doesn't establish spurious-vs-legitimate). "Which inductive bias *causes* faithful reliance" (causal claim requiring isolation §4 says is impossible).

---

## 8. Dataset (scoped — read before downloading anything)
- **Anchor on ONE coherent multimodal cohort: adult glioma, pre-treatment, the standard four sequences (T1, T1CE, T2, FLAIR), WT/TC/ET labels.** This is the clean, most-studied setting and the only one where the multimodal-reliance premise holds uniformly.
- **Do NOT pool across tumor types or label schemas.** Meningioma-RT is single-modality (T1c); post-treatment adult glioma uses a different sub-region scheme (adds resection cavity). Mixing reintroduces the cross-cohort domain shift deferred to Paper 2.
- If you want "2023–26": scope it to the **adult-glioma** cohort across those releases **only** with the pre-treatment WT/TC/ET schema held fixed, excluding the post-treatment/resection-cavity subset. Verify the **actual current tasks on Synapse** (BraTS-Lighthouse 2025 is the current challenge; the "BraTS 2026" task list in the research report was garbled).
- Other tumor types / years = **optional external-validity check at the very end**, never the spine.

---

## 9. Global protocol (non-negotiable)
Single adult-glioma cohort; patient-level splits saved to disk, zero leakage; brain-crop + per-channel z-score; fixed channel order `[FLAIR, T1, T1CE, T2]` frozen (ablation indexes by position); WT/TC/ET overlapping channels; same patch size, DiceCE loss, augmentation, inference (sliding window + Gaussian stitch + TTA + CCA); ablation-capable dataloader (mean-fill channel k) from day one; `physics_answer_key.json` committed as documentation that *informs* the conditional test (not a penalty). Seeds: ≥3 exploration, ≥5 confirmatory.

---

## 10. Audience fork (decide deliberately)
- **(A) SHIP IT — MICCAI / MIDL / iMIMIC.** Rigorous evaluation-diagnosis-plus-fix methodology is a legitimate primary contribution there. Finishable. This is the default.
- **(B) Aim higher (NeurIPS/ICML)** only by building the real ML contribution: information-theoretic conditional-faithfulness (conditional MI / PID on 3D volumes). ~Doubles the project; needs real theory, not assertion.
- **Trap:** chasing B's prestige with A's effort via a thin theorem + a "which inductive bias" title. Reviewers price it exactly. Pick A or commit to B.

---

## 11. Execution layer (what v3 lacked)

**Packaging / reproducibility.** Public repo: dataloaders, scaffold, metrics, synthetic check. Release as a **BraTS-Trust toolkit** others apply to their own models (a community resource → citations). Docker/MLCube image: model + data in → segmentation + Dice + conditional faithfulness + comparative fragility out. (Packaging, not the contribution.)

**Compute estimate.** One 3D U-Net on adult glioma ≈ 10–20 GPU-h; Probe 3 sweep (≥5 seeds × ~3 RF variants) ≈ 50–80 GPU-h; anchors + metrics + fix ≈ another ~40 GPU-h. Order ~150–200 GPU-h total. Feasible; no hardware limit on your side.

**Indicative timeline.**
- Wk 1–2: dataset access + preprocessing (adult glioma only); commit physics key + protocol.
- Wk 3–4: implement + unit-test §3 metric; build ablation dataloader; **synthetic sanity check** (§3.5).
- Wk 5–8: **Probe 3 RF sweep** on adult glioma, ≥5 seeds, ERF + convergence diagnostics → the decisive figure.
- Wk 9–10: if §4.1 holds, add Probe 1/2 + the 3 Tier-A anchors; run full reliance matrices + §3.3 comparative fragility + §3.4 XAI-fails.
- Wk 11–12: optional modality-dropout fix; statistics, effect sizes, CIs.
- Wk 13–14: toolkit + Docker; summary figure; draft.

**Summary figure (the poster child).** One dense panel: reliance matrix · ERF↔faithfulness curve · XAI-failure example · comparative-fragility (and fix) effect.

---

## 12. Stages (build order) + the standing call
0. Protocol, ablation dataloader, scaffold spec, metric implementation, physics-key JSON.
1. Synthetic sanity check (small).
2. **Probe 3 RF sweep — DECISIVE.** Does effective RF predict faithfulness? Run this before building anything else out.
3. If yes → Probe 1/2 + Tier-A anchors.
4. Full measurement: reliance matrices, comparative fragility, XAI-fails; statistics.
5. Characterize ERF↔faithfulness + consequence; soften "wrong reasons" → "reliance profiling" if §3.3 is weak.
6. Optional: modality-dropout fix, probing, gate — each only if it earns triangulation.
7. Toolkit + Docker; write-up; submit (Fork A).

**Standing recommendation:** the design is frozen. The only question left that planning cannot answer is **Stage 2** — whether receptive field actually predicts faithfulness. Build it. If it holds, everything else attaches to it; if it doesn't, you've saved yourself from building on sand. Stop revising; start generating the result.
