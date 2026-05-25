# Thesis writing guide

**Audience:** external writing agent · **Author:** Roger Esteve Sanchez (MATT, UPC-ETSETB) · **Advisor:** Dr Francisco Javier Hernando Pericas  

**Deliverable:** LaTeX thesis (`main.tex`, new `introduction.tex`, existing `state_of_the_art.tex`). **Do not** re-run experiments or invent metrics.

**One-line brief:** Empirical study of **modality, fusion, audio–text alignment, and Wav2Vec layer pooling** on three clinical speech tasks—not a comparison of two codebases.

---

## Document map

| § | Topic |
|---|--------|
| **A** | How to use this guide |
| **B** | Core narrative (mandatory framing) |
| **C** | Tasks and datasets |
| **D** | Two implementations (CogniAligned vs SER) |
| **E** | Alignment strategies and attribution |
| **F** | Experiment programme (steps A → Align×Wv) |
| **G** | Project status (done / to write / stale files) |
| **H** | CSV reference and pitfalls |
| **I** | Secondary reference documents |
| **J** | Mapping to `main.tex` |
| **K** | Section-by-section writing spec |
| **L** | Minimum thesis tables (T0–T7) |
| **M** | Do not include (strict) |
| **N** | `thesis_context/` bundle |
| **O** | Gaps (flag user; do not invent) |
| **P** | Positioning sentence (abstract / conclusions) |
| **Q** | Example tables for advisor review |

---

## A. How to use this document

| Step | Read | Purpose |
|------|------|---------|
| 1 | **§B–D** | What the thesis claims; tasks; two stacks |
| 2 | **§E–F** | Alignment attribution; experiment phases and CSVs |
| 3 | **§G** | What exists vs missing; **stale** log files |
| 4 | **§N** + `thesis_context/02_results/` | Upload bundle; **only** source for numbers |
| 5 | **§H, §J, §K** | Metrics, LaTeX map, prose outline |
| 6 | **§M, §O, §Q** | Forbidden content; blockers; example tables |

**Numbers:** `thesis_context/02_results/*.csv` (mirrors repo-root `*_ablation*.csv`, `step_b_results.csv`).

**Never use for ablation metrics:** `cogni_*_experiments.md`, `ser_*_experiments.md`, `experiment_results.csv` (legacy 108-job sweep), `alignment_pilot.csv` (superseded), tables in `AD_Disease_Odyssey.pdf` (outdated).

**Paths:** Canonical repo `/home/usuaris/veussd/roger.esteve.sanchez/`; portable copy `thesis_context/`. After editing this file at repo root, run `cp doc_guide.md thesis_context/doc_guide.md`.

---

## B. Core narrative (mandatory framing)

### B.1 Subject of the thesis

An **empirical study of technical strategies** for multimodal clinical speech ML:

1. **Modality** — audio, text, both, optional handcrafted acoustic features (AF)  
2. **Fusion** — seven seq2seq fusion families  
3. **Alignment** — word→token (literature) vs two **token→token** schemes (**this work**)  
4. **Wav2Vec pooling** — final layer vs learnable weighted sum of layers  

Two codebases (**CogniAligned**, **SER**) exist so fusion and Wav2Vec trends can be **replicated** under a second stack. The **unit of comparison is the technique**, not the repository.

**Research question (informal):** Which strategies matter for AD-related speech tasks, alone and in combination, and what combined architecture would merge the best choices?

### B.2 What the thesis is not

- A bake-off of “CogniAligned vs SER.”  
- A claim that both stacks are equivalent (different text encoders, alignment, training length).  
- A single deployed “final model”—experiments **decompose** choices; Discussion proposes a **hybrid** not fully implemented in one repo.

### B.3 How to phrase results

| Avoid | Prefer |
|-------|--------|
| “CogniAligned beat SER on ADReSSo.” | “Mamba fusion reached higher test accuracy on ADReSSo in the CogniAligned stack; SER showed the same fusion ranking directionally.” |
| “We compare two models.” | “We ablate fusion, alignment, and Wav2Vec pooling; SER replicates fusion and layer pooling only.” |
| “The best model is …” | “The strongest **combination of techniques** in our grid was …” |

**Tables:** Primary columns = dataset, **technique**, metrics. Add `implementation` only for replication rows (Steps B, WvLayer).

### B.4 Hybrid architecture (Discussion only)

Propose a **target system** the thesis does not train end-to-end:

| Component | Typical evidence in this work |
|-----------|-------------------------------|
| Multimodal input | Step A (A+T; +AF if gain justifies complexity) |
| Fusion | Step B (task-dependent; often mamba / bicross / crossgated) |
| Alignment | Tier A + Align×Wv — word→token baseline + best **token→token** row |
| Wav2Vec | WvLayer — weighted layers when val/test gain clear |
| Engineering | CogniAligned-style **aligned `.pt` precompute** + SER-style **embedding cache** |

*Example:* “A practical system would precompute token-aligned tensors, apply learnable Wav2Vec layer weights, and train fusion on cached features—no single codebase here implements every top-ranked choice together.”

---

## C. Clinical tasks and datasets

| Task | Cohort | Language | Labels | Held-out test |
|------|--------|----------|--------|---------------|
| AD vs CN | **ADReSSo** (Cookie Theft) | English | Binary | **Yes** — `test-dist` → report `test_acc` |
| Amyloid proxy | **SPIN / WAB_samples** | Spanish | Binary | **No** — 5-fold CV val only |
| PPA variant | **WAB_samples** | Spanish | 3-class | **No** — 5-fold CV val only |

- Amyloid and PPA share the WAB tree → `run_preprocessing_wab.py`.  
- ADReSSo is the only ADReSS-style external test; Spanish tasks: **never** invent a test split.  
- Cross-lingual EN vs ES: Discussion, not an apology.

**Naming:** Write **ADReSSo** in prose; `main.tex` §4.1 title is `\subsection{AD Classification on ADReSS}` (challenge naming)—keep consistent in text.

---

## D. Two implementations

| | **CogniAligned** (`CogniAligned/`) | **SER** (`ad-detection/`) |
|--|-----------------------------------|---------------------------|
| Role | Full pipeline; **only** stack with token alignment | Replicate **fusion** and **Wav2Vec pooling** |
| Audio | Wav2Vec2 XLSR-300M, frozen | Same family |
| Text | DistilBERT, frozen | ModernBERT, frozen |
| Layout | Token-aligned precomputed `.pt` | Segment-level (no subword alignment) |
| AF (OpenSMILE eGeMAPS) | Step A and optional configs | Not in main ablation grid |
| Fusion code | `modules/model.py` | `scripts/poolings.py` |
| Training | 5-fold CV; lr `2e-5`; max 200 epochs; patience 15 | 5-fold CV; lr `5e-5`; 25 epochs |
| Speed | Precompute once | `cache/ser_embeddings`, `ser_embeddings_last` |

**Fairness:** SER originally ran encoders every batch; cached embeddings were added later (`experiments_plan.md`). Report inference times from final CSVs (cached). Cross-stack numbers are **directional**, not identical reproduction.

**Names:** **CogniAlign** = published method (cite Ortiz et al.). **CogniAligned** = your codebase.

---

## E. Alignment strategies (Introduction + Methodology)

| Strategy | `alignment` in CSV | Preprocess suffix | Proposed by | In thesis |
|----------|-------------------|-------------------|-------------|-----------|
| Word → token | `word_token` | — | **CogniAlign** (`ortiz2025cognialign`) | Literature baseline |
| Token → token, uniform | `token_token_uniform` | `_tokalign` | **This thesis** | Novel |
| Token → token, proportional | `token_token_char` | `_tokalignchar` | **This thesis** | Novel |

Prose label **proportional** = `token_token_char` only.

**Code:** `thesis_context/04_code/preprocessembeddings.py`; CLI `run_preprocessing.py`, `run_preprocessing_wab.py`.

**Not evaluated:** `none` / `_noalign` — future work at most.

**SER:** No alignment axis; rows use `alignment=segment_level` in Align×Wv CSV only.

---

## F. Experiment programme

### F.1 Controlled defaults (Steps B, C, WvLayer, Align×Wv)

Unless a phase says otherwise:

| Setting | Value |
|---------|--------|
| Modalities | Audio + text (**no AF**) |
| Alignment (CogniAligned) | `word_token` in Step B; uniform/char in Tier A / Align×Wv |
| Wav2Vec in precompute | **Final layer** in B and Tier A; **weighted** additionally in WvLayer and Align×Wv |
| CV | 5-fold stratified per dataset |
| CogniAligned lr | `2e-5` |
| SER lr | `5e-5` |

Step **A** alone varies modality (including A4 with AF) on a **minimal** encoder (`n_layers=1`, `n_heads=1`, mean pool)—not comparable fusion capacity to Step B.

### F.2 Phases

| Step | Question | CSV | Rows (approx.) | CogniAligned | SER |
|------|----------|-----|----------------|--------------|-----|
| **A** | Modality contribution? | `modality_ablation.csv` | 12 | Yes | No |
| **B** | Best fusion family? | `step_b_results.csv` | 42 (7×2×3) | Yes | Yes |
| **C** (Tier A) | Alignment + fusion (bicross **or** mamba)? | `alignment_tier_a.csv` | 18 (3×2×3) | Yes | No |
| **WvLayer** | Final vs weighted Wav2Vec? | `wav2vec_layer_ablation.csv` | 36 (3 fusions × 2 layers × 2 impl × 3 tasks) | Yes | Yes† |
| **Align×Wv** | Alignment × **weighted** Wav2Vec @ **bicross**? | `alignment_wv2vec_ablation.csv` | 36 (3 align × 2 Wv × 3 tasks × 2 impl, partial reuse) | Yes | Wv only |

† Some SER WvLayer cells marked `incomplete` in CSV—do not quote missing val metrics.

**Dependencies:** A → motivates multimodal use. B → fixed fusion comparators for Tier A (bicross, mamba) and Align×Wv (bicross). WvLayer can be read independently. Align×Wv = alignment story + weighted layers.

**Reuse:** `note=reused_stepB` / `reused_wv_layer` means metrics copied from an earlier trained cell—valid for reporting; footnote in thesis.

### F.3 Fusion names (CSV ↔ prose)

| Prose | CogniAligned `arch` | SER `ser_method` |
|-------|---------------------|------------------|
| Self-attention | `selfattn` | `SelfAttention` |
| Multi-head self-attention | `mhselfattn` | `MultiHeadAttention` |
| Transformer + PE | `transformerpe` | `TransformerPosEnc` |
| Cross-attention | `cross` | `CrossAttention` |
| Gated cross-attention | `crossgated` | `GatedCrossAttention` |
| Bidirectional cross-attention | `bicross` | `BidirectionalCrossAttention` |
| Mamba | `mamba` | `MambaSeqToSeq` |

Orchestrators (repo root): `run_modality_ablation.py`, `run_step_b.py`, `run_alignment_tier_a.py`, `run_wav2vec_layer_ablation.py`, `run_alignment_wv2vec_ablation.py`.

---

## G. Project status

### Done
- Steps A, B, C (Tier A), WvLayer, Align×Wv complete; CSVs in `02_results/`.  
- `state_of_the_art.tex` drafted (~378 lines).  
- `main.tex` skeleton; SOTA included.

### To write
- `introduction.tex` (missing; required by `main.tex`).  
- Abstract; Methodology; Experiments; Discussion; Conclusions.  
- Replace five `[Figure: …]` placeholders in SOTA.  
- Title, date, cover in `main.tex`.

### Stale (templates only)
- `CogniAligned/cogni_*_experiments.md`, `ad-detection/ser_*_experiments.md` (~2026-05-11).  
- No ablation rows; see `thesis_context/06_experiment_notes/`.

---

## H. CSV reference

### Metrics
- `val_acc_mean` — mean of per-fold best validation accuracy.  
- `best_fold_acc` — best single fold.  
- `val_inference_ms_per_sample`, `test_inference_ms_per_sample`.  
- `test_acc` — **ADReSSo only** (official test-dist).  
- Spanish decimals: `*_es.csv` (`;` separator).

### Per-file rules

| File | Use | Pitfall |
|------|-----|---------|
| `modality_ablation.csv` | Step A; cells A1–A4 | CogniAligned only; small encoder |
| `step_b_results.csv` | Step B fusion | Primary fusion table; `model` + `arch` / `ser_method` |
| `alignment_tier_a.csv` | Step C | Includes **mamba**; not in Align×Wv |
| `wav2vec_layer_ablation.csv` | WvLayer | `wav2vec_layers` = `final` \| `weighted`; check `incomplete` |
| `alignment_wv2vec_ablation.csv` | Align×Wv @ bicross | **SER `test_acc`:** verify via `step_b_results.csv` or `wav2vec_layer_ablation.csv` if suspicious |
| `experiment_results.csv` | — | **Do not use** |
| `alignment_pilot.csv` | — | Appendix only |

### Reporting checklist
1. State step (A/B/C/WvLayer/Align×Wv) and controlled defaults (§F.1).  
2. ADReSSo: report `test_acc` + `val_acc_mean`; Spanish tasks: val only.  
3. Alignment tables: **provenance** column (CogniAlign vs this thesis).  
4. Footnote reuse and incomplete SER cells.  
5. No significance tests unless added later (§O).

---

## I. Secondary references

| Document | Use | Avoid |
|----------|-----|-------|
| `thesis_context/03_pdfs/AD_Disease_Odyssey.pdf` | Motivation, prior wording | Current metrics |
| `thesis_context/03_pdfs/Master_Thesis_Lucas_Takanori.pdf` | UPC structure/tone | Technical content |
| `thesis_context/00_planning/thesis_ablation_plan.md` | Phase definitions | — |
| `thesis_context/00_planning/experiments_plan.md` | SER cache, fairness | — |
| `state_of_the_art.tex` | Literature chapter | Your result numbers |

---

## J. Mapping to `main.tex`

| LaTeX | Source | Guide |
|-------|--------|-------|
| Abstract | `main.tex` | §P; headline numbers from §H |
| §1 Introduction | **`introduction.tex`** (create) | §K.1 |
| §2 State of the Art | `state_of_the_art.tex` | §K.2 |
| §3 Methodology | new prose in `main.tex` or `methodology.tex` | §K.3; add §D, §E, strategy catalogue (T0) |
| §4 Experiments | see below | §K.4 |
| §5 Discussion | | §K.5 |
| §6 Budget | brief | school requirement |
| §7 Environmental impact | brief | school requirement |
| §8 Conclusions and Future Work | | §K.6 |
| Appendices | full CSVs, hyperparameters | §L |

### §4 order (do not reorder)

| `main.tex` subsection | Content |
|-----------------------|---------|
| §4.1 AD Classification on ADReSS | Task summary; best bundle; **`test_acc`**; vs literature |
| §4.2 Amyloid-β on SPIN | Val only; best bundle |
| §4.3 PPA variant | Val only; confusion matrix or limitation |
| §4.4 Ablation Studies | Steps A → B → WvLayer → C + Align×Wv (technique-first) |
| §4.5 Error Analysis | Prepared work or explicit future work |

**Inside §4.4 suggested order:** Modality → Fusion → Wav2Vec layers → Alignment (Tier A then Align×Wv). End each block with SER replication paragraph where applicable (B, WvLayer only).

---

## K. Section writing specifications

### K.1 Introduction

1. Clinical motivation (AD, amyloid, PPA).  
2. Speech as biomarker (pointer to Odyssey).  
3. Gap: entangled design choices; limited cross-lingual ablations.  
4. Approach: staged steps A–Align×Wv; SER for replication.  
5. **Contributions (numbered):**  
   - Modality on three tasks (Step A).  
   - Seven fusion families × two implementations (Step B).  
   - **Two token→token alignments** vs CogniAlign word→token (Steps C, Align×Wv).  
   - Wav2Vec final vs weighted (WvLayer).  
   - Recommended hybrid architecture (Discussion).  
6. Thesis outline.  
7. Optional: Interspeech 2025 extension.

### K.2 State of the Art

- Replace/delete five figure placeholders.  
- Trim HuBERT/WavLM detail (~30%); trained encoder = **XLSR-300M** only.  
- **No CSV numbers** in SOTA.  
- Add bridge: CogniAlign word→token → this thesis extends with uniform/proportional + joint ablations.

### K.3 Methodology

Follow `main.tex` subsections; additionally:

- **Two implementations** (§D) and **fair comparison** caveat.  
- **Alignment** (§E) with algorithms for uniform/proportional.  
- **Strategy catalogue** (table T0).  
- Step A encoder ≠ Step B fusion capacity (§F.1).

### K.4 Experiments

- **§4.1–4.3:** One “best bundle” table per task (T6 style)—derived from full CSV scan, not a single arbitrary row.  
- **§4.4:** Hypothesis → table → interpretation per step; cite CSV filenames.  
- **Alignment:** Tier A (bicross **and** mamba) **then** Align×Wv (weighted, bicross only).  
- **§4.5:** Do not fabricate confusion matrices.

### K.5 Discussion

1. **Technique synthesis first** (T6 / §B.4)—trade-offs, not one “winner run.”  
2. Prior work (Odyssey outdated results; ADReSSo benchmarks).  
3. Cross-lingual EN/ES.  
4. Clinical deployment.  
5. Limitations: N, labels, ASR, frozen encoders, stack differences, no multiple-testing correction.  
6. Threats to validity.

### K.6 Conclusions

Answer contributions cautiously; hybrid system as **future work** with named strategies.

---

## L. Minimum tables (T0–T7)

| ID | Content | Primary source |
|----|---------|----------------|
| T0 | Strategy × where evaluated | §F.2 |
| T1 | Datasets and protocol | §C |
| T2 | Modality ablation | `modality_ablation.csv` |
| T3 | Fusion ablation | `step_b_results.csv` |
| T4 | Wav2Vec final vs weighted | `wav2vec_layer_ablation.csv` |
| T5 | Alignment (+ weighted @ bicross) | `alignment_tier_a.csv`, `alignment_wv2vec_ablation.csv` |
| T6 | Best technique bundle per task | Derived from all CSVs |
| T7 | Params and inference time | Step B + WvLayer CSVs |

**Example layouts:** §Q (subset rows; advisor review).

---

## M. Do not include

- Two-codebase bake-off framing.  
- SER **alignment** comparisons (N/A).  
- `experiment_results.csv`, `alignment_pilot.csv`, Odyssey result tables.  
- HuBERT/WavLM as **trained** encoders in this work.  
- `none` alignment as evaluated.  
- Identical cross-stack claims without encoder/training caveats.  
- Ablation numbers from `*_experiments.md`.  
- Quoting incomplete SER cells without marking them missing.

---

## N. `thesis_context/` bundle

**Path:** `/home/usuaris/veussd/roger.esteve.sanchez/thesis_context/`

| Folder | Contents |
|--------|----------|
| `doc_guide.md` | Copy of this file (sync from root) |
| `README.md` | Short index |
| `00_planning/` | Ablation + experiments plans |
| `01_latex/` | `main.tex`, `state_of_the_art.tex`, bib snippets |
| `02_results/` | **All ablation CSVs** |
| `03_pdfs/` | Odyssey, Lucas thesis |
| `04_code/` | Preprocess + SER embedding scripts |
| `05_orchestrators/` | `run_*.py` |
| `06_experiment_notes/` | Stale `.md` logs |
| `07_configs/` | Sample YAML |

Missing from bundle: full code trees, checkpoints, `introduction.tex`, `tools/*.tex`, `bibliography.bib` (§O).

---

## O. Gaps (flag user; do not invent)

1. `introduction.tex` missing.  
2. `tools/packages.tex`, `bibliography.bib` may be missing.  
3. No official amyloid/PPA test set.  
4. Error analysis not prepared.  
5. No statistical tests on fold means.  
6. SOTA figure placeholders remain.  
7. Stale `*_experiments.md` not backfilled.  
8. Verify SER `test_acc` in `alignment_wv2vec_ablation.csv` against `wav2vec_layer_ablation.csv` / `step_b_results.csv`.

---

## P. Positioning sentence

*This thesis empirically compares technical strategies for multimodal clinical speech classification—modality mix, fusion architecture, Wav2Vec layer pooling, and audio–text alignment—on English and Spanish cohorts. Building on CogniAlign’s word-to-token alignment, we propose and evaluate two token-to-token schemes (uniform and proportional), and synthesize the results into a recommended combined architecture that no single codebase in this work fully implements.*

---

## Q. Appendix — example thesis tables (advisor review)

**Purpose:** Show **layout and columns** for T0–T7 (§L). Subset of rows; **not** final thesis tables. T6 is **illustrative**—final bundles must come from a full CSV review.

**Numbers:** From repo CSVs (May 2026). Accuracies = metric × 100, one decimal.

---

### T0 — Strategy catalogue

| Strategy | CogniAligned | SER | Step |
|----------|:------------:|:---:|------|
| Modality (A/T/A+T/+AF) | ✓ | — | A |
| Fusion (7 families) | ✓ | ✓ | B |
| Word→token alignment | ✓ | — | C, Align×Wv |
| Token→token uniform / proportional | ✓ | — | C, Align×Wv |
| Wav2Vec final vs weighted | ✓ | ✓ | WvLayer, Align×Wv |
| Align × weighted Wv @ bicross | ✓ | Wv only | Align×Wv |

---

### T1 — Datasets

| Task | Cohort | Lang. | Classes | Test |
|------|--------|-------|---------|------|
| AD vs CN | ADReSSo | EN | 2 | Official `test_acc` |
| Amyloid | WAB/SPIN | ES | 2 | CV val only |
| PPA | WAB | ES | 3 | CV val only |

---

### T2 — Modality (ADReSSo, Step A)

| Cell | Modality | Val % | Test % |
|------|----------|-------|--------|
| A1 | Audio | 70.8 | 66.2 |
| A2 | Text | 86.1 | 80.3 |
| A3 | A+T (mean) | 84.8 | 78.9 |
| A4 | A+T+AF | 86.7 | 81.7 |

*Note:* Step A uses a 1-layer encoder—not Step B fusion capacity.

---

### T3 — Fusion subset (ADReSSo, Step B; A+T, no AF)

| Fusion | Impl. | Val % | Test % |
|--------|-------|-------|--------|
| Bicross | CogniAligned | 84.8 | 78.9 |
| Bicross | SER | 85.2 | 74.6 |
| Mamba | CogniAligned | 88.4 | **83.1** |
| Mamba | SER | 83.6 | 74.6 |
| Crossgated | CogniAligned | 86.6 | 81.7 |
| Crossgated | SER | 82.5 | 76.1 |

---

### T4 — Wav2Vec layers (ADReSSo, bicross, Step WvLayer)

| Pooling | Impl. | Val % | Test % |
|---------|-------|-------|--------|
| Final | CogniAligned | 84.8 | 78.9 |
| Weighted | CogniAligned | 88.5 | 78.9 |
| Final | SER | — | 87.3* |
| Weighted | SER | 85.2 | 74.6 |

\*SER final: val missing in CSV (`incomplete`)—do not report val without check.

---

### T5 — Alignment (ADReSSo)

**Tier A — bicross, final Wav2Vec in precompute**

| Alignment | Source | Val % | Test % |
|-----------|--------|-------|--------|
| Word→token | CogniAlign | 84.8 | 78.9 |
| Uniform | This thesis | 84.2 | 76.1 |
| Proportional | This thesis | 84.8 | 78.9 |

**Tier A — mamba (same alignments; include in thesis, not only bicross)**

| Alignment | Source | Val % | Test % |
|-----------|--------|-------|--------|
| Word→token | CogniAlign | 88.4 | **83.1** |
| Proportional | This thesis | 88.5 | **83.1** |

**Align×Wv — bicross + weighted Wav2Vec**

| Alignment | Val % | Test % |
|-----------|-------|--------|
| Word→token | 88.5 | 78.9 |
| Uniform | 87.2 | 77.5 |
| Proportional | 87.2 | **80.3** |

*Interpretation:* Highest **test** on ADReSSo in grid shown: mamba + word/proportional (**83.1%**); weighted bicross + proportional (**80.3%**). Discussion must reconcile val vs test and complexity—not pick one cell silently.

---

### T6 — Illustrative bundles (verify before thesis)

| Task | Modality | Fusion | Alignment | Wav2Vec | Val % | Test % |
|------|----------|--------|-----------|---------|-------|--------|
| ADReSSo | A+T | Mamba | word→token | final* | 88.4 | 83.1 |
| ADReSSo | A+T | Bicross | proportional | weighted | 87.2 | 80.3 |
| Amyloid | A+T | Bicross | uniform | — | 85.1 | — |
| PPA | A+T | Bicross | word→token | final | 69.7 | — |

\*Step B/WvLayer precompute; not the weighted Align×Wv row.

---

### T7 — Compute (ADReSSo examples)

| Setting | Impl. | Trainable params (M) | Val ms | Test ms |
|---------|-------|----------------------|--------|---------|
| Fusion bicross | CogniAligned | 21.5 | 0.59 | 1.04 |
| Fusion bicross | SER | 0.61 | 1.18 | 236.6 |
| Fusion mamba | CogniAligned | 11.1 | 0.96 | 1.14 |
| Modality text (A2) | CogniAligned | 7.3 | 0.20 | 0.72 |

SER params low: fusion head only; embeddings precomputed.

---

### Pre-thesis table checklist

- [ ] Full CSVs exported to appendix where space allows  
- [ ] ADReSSo: always `test_acc` when citing “best”  
- [ ] Spanish tasks: no test column  
- [ ] Footnotes: `reused_stepB`, incomplete SER  
- [ ] Alignment provenance on every alignment table  
- [ ] T6 derived from systematic scan, not §Q alone  

---

*Guide v3: consistency pass—fixed §A/§G cross-refs, Step B defaults (no AF), Tier A mamba in T5, T0 source, main.tex §4 titles, row counts, SER CSV caveats.*
