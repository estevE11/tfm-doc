# Revisor improvements — tracking list

Tracker for the revisor's feedback. **Nothing fixed yet** — this is the to-do list with
context, verification, and open questions for each point. Source data verified against
`thesis_context/` (CSVs, configs, experiment notes).

Document files: `main.tex` (cover, abstract, methodology, budget, env, conclusions),
`introduction.tex`, `state_of_the_art.tex`, `result.tex`.

Legend: **[verified]** checked against data/code · **[needs decision]** my input needed ·
**[may need re-run]** experiment data missing.

---

## A. Results section

### A1. Final results must go at the END, not stated early
> "Results final results have to go at the end (currently we have two sections for dataset where we state the final results)."

- **Current state:** `result.tex` opens with three per-task subsections — `4.1 ADReSSo`
  (`ssec:res_adresso`), `4.2 Amyloid` (`ssec:res_amyloid`), `4.3 PPA` (`ssec:res_ppa`) —
  and each one announces its **headline/best number up front** (88.16 / 83.90 / 74.95).
  The consolidated `Table tab:res_best` ("Best technique bundle per task") then appears
  much later, at the end of the Ablation Studies block.
- **Reading of the complaint:** headline results are presented twice and too early — once
  in the per-task intros, once in `tab:res_best`. Revisor wants the final/best numbers
  pulled to the end so the section *builds up* to them through the ablations instead of
  spoiling them.
- **Tie-in:** overlaps with A4/A5 (per-task vs per-experiment structure) and item D
  (move best-config narrative to the end). Best handled together as one Results rewrite.
- Affected: `result.tex` (whole structure).

### A2. Table layout — rename ADReSSo→AD, add a dataset header row — ✅ DONE
> Implemented in `result.tex`: all 6 column-task tables (`tab:res_modality`, `tab:res_acoustic`,
> `tab:res_fusion`, `tab:res_wav2vec_pool`, `tab:res_alignment`, `tab:res_encoder`) now have a
> top dataset row (`\multicolumn{2}{c}{SPIN}` + `ADReSSo`) with `\cmidrule`s, task row renamed
> ADReSSo→AD. `tab:res_best` (task-as-row) row relabelled "AD (ADReSSo, ref.)". Fusion bar-chart
> legend ADReSSo→AD. Compiles clean (51pp, no errors). Pending Phase-0 decisions don't affect this.

> "change adresso in tables to AD then add another row above for the datasets eg. SPIN, ADReSSO, then below Amyloid, PPA, AD"

- **Goal:** two-level column header. Top row = dataset (SPIN | ADReSSo). Second row =
  task (Amyloid, PPA | AD). So the *task* column currently labelled "ADReSSo" becomes
  "AD", and the dataset name moves to a grouping header above.
- **Current columns** in every results table: `Amyloid | PPA | ADReSSo`. Note Amyloid+PPA
  are the SPIN/Spanish tasks (bolded), ADReSSo is the English reference (not bolded).
  The new grouping makes that SPIN-vs-ADReSSo split explicit.
- **Tables to change** (all in `result.tex`): `tab:res_modality`, `tab:res_acoustic`,
  `tab:res_fusion`, `tab:res_wav2vec_pool`, `tab:res_alignment`, `tab:res_encoder`,
  `tab:res_best`. Also `tab:datasets` in `main.tex` already lists tasks row-wise — keep
  consistent.
- **LaTeX:** needs `\multicolumn{2}{c}{SPIN}` + `\cmidrule` over the two Spanish columns.
- Low risk, purely cosmetic, but touches ~7 tables.

### A3. Ablation per TASK, not per experiment
> "professor told me to make ablation per task instead of per experiment (im not too sure about that)" — *revisor's own uncertainty noted.*

- **Current:** `4.4 Ablation Studies` is organised **per axis/experiment**
  (`Modality`, `Handcrafted Features`, `Fusion`, `Layer Pooling+Mixer`, `Alignment`,
  `Feature Extractors`, `Unified-Model Confirmation`) — each its own `\subsubsection`,
  each with a cross-task table.
- **Proposed:** reorganise so each *task* (AD, Amyloid, PPA) gets its narrative and the
  axes are discussed within it.
- **[needs decision] — TENSION:** revisor flagged their own doubt. The data is structured
  cross-task: every results table (`tab:res_*`) has one column per task, and the central
  thesis message is *"the best axis choice is task-dependent"*, which reads best with the
  axis as the row and tasks side-by-side for comparison. A pure per-task reorg would
  **break the side-by-side axis comparison** and force duplicating each table 3×.
- **HYBRID PROPOSAL (recommended):** two-layer structure —
  1. **Ablation walk (axis-driven, cross-task) — the body.** Keep the cross-task tables
     exactly as they are (one column per task), but **drop the per-experiment subsection
     headers** (satisfies A4): the modality → fusion → pooling/mixer → alignment → encoder
     findings flow as continuous prose, each paragraph still comparing all three tasks so
     the "task-dependent" message survives. This is where the comparison lives.
  2. **Per-task synthesis (task-driven) — the end.** Close with a short paragraph per task
     (AD, Amyloid, PPA), each one *pulling together* that task's winning stack and its
     headline number, culminating in `tab:res_best`. This is the **per-task** view the
     revisor wants, and it lands the **final results at the end** (satisfies A1/A5).
  - Net: revisor gets per-task framing *where it pays off* (the conclusions/best configs),
    we keep the cross-task tables *where the evidence lives* (the ablations), no table is
    duplicated 3×, and the headline numbers move from the top (current per-task intros) to
    the bottom. Resolves A1+A3+A4+A5 in one structure.
  - Still worth a one-line confirmation with the revisor that this counts as "per task".
- **CONCRETE PROPOSED LAYOUT for `result.tex`:**
  - *Intro paragraph* — keep as is (conventions: 5-fold CV, seeds 43/3407, ADReSSo
    test-dist only, point estimates / no significance testing, stable-recipe note).
  - **REMOVE** the current top-of-section per-task subsections `4.1 ADReSSo` /
    `4.2 Amyloid` / `4.3 PPA` (these are what spoil the headline numbers early).
  - **§ Ablation findings** (axis-driven, cross-task, flowing prose, *no per-experiment
    `\subsubsection` headers* — they arise naturally):
    | Topic in prose | Tables / figures placed here |
    |---|---|
    | Modality + handcrafted features | `tab:res_modality`, `tab:res_acoustic` |
    | Fusion (incl. SER cross-stack reproduction) | `tab:res_fusion`, `fig:res_fusion_bar`; `fig:res_compute` **minimised** (A6) |
    | Wav2Vec pooling + layer mixer | `tab:res_wav2vec_pool`, `fig:res_layer_mixer` |
    | Audio–text alignment | `tab:res_alignment` |
    | Feature extractors (encoder swaps) | `tab:res_encoder` |
  - **§ Per-task results** (the END — this is the "per task" + "final results at end" part):
    one paragraph each for **AD (ADReSSo)**, **Amyloid (SPIN)**, **PPA (SPIN)**, each
    pulling together that task's winning stack and headline number (val for all; val+test
    for AD). The unified-model confirmation note folds in here. Single table: **`tab:res_best`**
    ("Best technique bundle per task") lands here, at the bottom.
  - **§ Error analysis** — stays last (currently a stub → cross-ref future work).
  - **Summary of table moves:** all `tab:res_*` ablation tables stay in the body
    (unchanged columns, pending A2 header reformat); only **`tab:res_best` moves to the
    end**; the headline prose currently in 4.1–4.3 is **deleted from the top** and
    **rewritten into the per-task closing paragraphs**.
- Affected: `result.tex`.

### A4. Don't give each ablation experiment its own subsection — let them arise naturally
> "told me also to not make a subsection per each ablation experiments, just have to come up naturally in the results section"

- **Current:** 7 `\subsubsection`s under Ablation Studies (see A3).
- **Proposed:** drop the per-experiment headers; fold the ablations into flowing prose.
- Compatible with A1/A3/A5 — do as one rewrite. Keep the tables (they carry the numbers),
  just remove the heading scaffolding and let the text bridge them.
- Affected: `result.tex`.

### A5. (= A1 restated) final results at the end, not the beginning
> "with final results at the end not at the beginning"

- Duplicate of A1. Single fix.

### A6. Accuracy-vs-time content must be minimal — ✅ DONE
> Removed `fig:res_compute` entirely; condensed the inference-cost paragraph in the Fusion
> discussion to a single clause (cached embeddings → all families sub-ms → accuracy drives the
> choice). SER cross-stack *reproduction* point kept (that's the important part, not the timing).
> Budget/Environmental-Impact cache framing left untouched.

> "the accuracy/time section has to be minimal, not really important in AD detection"

- **Current:** `Figure fig:res_compute` (accuracy vs per-sample inference time scatter) +
  a full paragraph in the Fusion subsection (lines ~120–153 of `result.tex`) about the
  CogniAligned-vs-SER >2-orders-of-magnitude inference gap from the embedding cache.
- **Proposed:** cut to a sentence or footnote; drop or shrink `fig:res_compute`.
- **Note:** the compute angle is also used in `Budget` and `Environmental Impact`
  ("precompute-and-cache keeps GPU cost small") — those can keep the framing; only the
  *Results* treatment shrinks. Don't delete the cache argument entirely, just demote it.
- Affected: `result.tex` (mainly), cross-ref check in `main.tex` Budget/Env.

---

## B. Handcrafted features in the modality ablation

> "Handcrafted features ablation is missing, modality ablations should include them, if no results found you tell me and then we repeat the experiments."

**[verified] — partly missing. Here is exactly what exists:**

1. **A dedicated handcrafted-feature ablation DOES exist** — it is `Table tab:res_acoustic`
   ("Handcrafted Acoustic Features", `4.4`). Data: `res_model_acoustic_features.csv`
   (ADReSSo: `no_af` / `af_token` / `af_concat`) and
   `res_model_spanish_acoustic_features.csv` (amyloid, PPA: `no_af` / `af_concat`).
   All doc numbers reconcile to the CSV seed-means (e.g. amyloid no_af 78.60, af_concat
   80.68; PPA 72.52 → 70.25; ADReSSo 86.94 → 87.54). So "ablation is missing" is **not**
   true in the absolute — it exists as its own table.

2. **What IS missing:** the **modality table** (`tab:res_modality`) has no `A+T+AF` row.
   The `res_model` modality block (`MOD` in `res_model_spin_ablation_summary.csv`) contains
   only `audio_only`, `text_only`, `concat`, `mean` — **no acoustic-feature cell.** So the
   revisor is right that the *modality* study omits handcrafted features.

3. **[may need re-run] — the catch.** A clean `A+T+AF` row for the modality table can't just
   be lifted from the acoustic table, because they use **different heads/recipes**:
   - Modality study (`tab:res_modality`) = deliberately **minimal head** (1 attn layer,
     mean pool) — a floor ranking of inputs.
   - Acoustic table (`tab:res_acoustic`) = **full bidirectional cross-attention** recipe.
   Mixing them is apples-to-oranges. To add `A+T+AF` to the modality table *consistently*
   (same minimal head), the run is **not in the data** → must be re-run for the 3 tasks
   × 2 seeds.
   - **However**, the **legacy CogniAligned** modality grid *does* have it:
     `modality_ablation.csv` row `A4_av_mean_af` (`A+T+AF`, mean fusion, minimal head) for
     all 3 tasks (e.g. ADReSSo 0.8666, amyloid 0.8088, PPA 0.6928). If the revisor accepts
     legacy CogniAligned numbers for the modality floor, **no re-run needed** — just add the
     row and label the stack. If they want it under `res_model`, **re-run required.**
- **Decision for you:** legacy CogniAligned A+T+AF cell (data ready) vs new res_model
  A+T+AF run (clean but needs compute). Flagging per your instruction.
- Affected: `result.tex` (`tab:res_modality`, modality prose).

---

## C. Fusion short names must be explained — ✅ DONE
> Added a full abbreviation legend to the `fig:res_fusion_bar` caption (Concat, Mean, Self-attn,
> MH self-attn, Transf.+PE, Cross, Gated cross, Bi-cross, Mamba → full names). `fig:res_compute`
> (the other user of these abbrevs) was removed under A6, so no second legend needed.

> "Fusion short names have to be explained."

- **Where:** `Figure fig:res_fusion_bar` x-axis and `fig:res_compute` use abbreviations:
  `Concat`, `Mean`, `Self-attn`, `Transf.+PE`, `MH self-attn`, `Cross`, `Gated cross`,
  `Bi-cross`, `Mamba`. Tables `tab:res_fusion` / `tab:fusion` spell them out, but the
  **figures don't**, and the short forms are never mapped to the full names in one place.
- **Canonical mapping** (from `res_model_fusion_sweep.csv` internal keys → display):
  | short | full | csv key |
  |---|---|---|
  | Concat | Concatenation (baseline) | `concat` |
  | Mean | Element-wise mean (baseline) | `mean` |
  | Self-attn | Self-attention | `self_attention` |
  | MH self-attn | Multi-head self-attention | `multi_head_self_attention` |
  | Transf.+PE | Transformer + positional encoding | `transformer_positional` |
  | Cross | Cross-attention | `cross_attention` |
  | Gated cross | Gated cross-attention | `gated_cross_attention` |
  | Bi-cross | Bidirectional cross-attention | `bidirectional_cross_attention` |
  | Mamba | Mamba (selective state-space) | `mamba` |
- **Fix:** add this legend to the fusion figure caption, or define abbreviations on first
  use. `tab:fusion` in `main.tex` already groups them — could cite it from the figure.
- Affected: `result.tex` (figure captions).

---

## D. `res_model` needs a backronym (must stay "R.E.S.")

> "res_model is good, but we should make an excuse for the R.E.S. ... per example Reumatic Estimation System, but it has to make sense for the project. But it is mandatory that it stays as R.E.S."

- **[needs decision] — naming task.** Must expand to R.E.S. and fit the project (a single
  **configuration-driven unified model** that swaps modality / fusion / alignment / Wav2Vec
  pooling / layer-mixer / encoders from one config, consolidating the per-axis winners).
- ~~"Reconfigurable Encoder-fusion System"~~ — **rejected (round 1).**
- ~~"Reconfigurable Experimentation Stack" / "Representation Ensemble for Speech" /
  "Recommended Evaluation Stack" / "Robust Estimation System"~~ — **not liked (round 2).**
- **Round 3 candidates** (aiming for a name that *sounds like a real system* and ties to the
  clinical-speech / reproducibility themes — pick/refine):
  - **R**eproducible **E**valuation **S**uite — *(my recommendation)*: the thesis's whole
    point is that `res_model` **reproduces the per-axis trends in one unified codebase**
    after they appeared on two separate stacks. "Reproducible" names the model's actual job;
    "Suite" fits a config-driven harness running many experiments.
  - **R**econfigurable **E**mbedding **S**creener — clinical framing: a tool that fuses
    (re-configurable) audio+text **embeddings** to **screen** for neurodegenerative disease.
    Closest to the "sounds like a real clinical system" vibe of the revisor's own example.
  - **R**esidual **E**mbedding **S**ystem — if "res" is meant to echo *residual/representation*;
    plainest, leans on the literal `res_` prefix.
  - **R**eference **E**ncoder **S**tack — frames it as the consolidated *reference* model the
    thesis recommends.
- *Naming is a taste call* — these are directions, not a verdict. If none fit, tell me which
  **word per slot** you lean toward (R / E / S) and I'll build around it.
- Where it lands: define once in `main.tex §3.4` (`ssec:res_model`), then the `\texttt{res\_model}`
  occurrences across `main.tex` + `result.tex` can keep the mono name but gloss it as R.E.S.
- Affected: `main.tex` (definition), naming consistency throughout.

---

## E. State of the Art — reorder, fewer subsections

> "State of the art section: bad order. Biological motivation maybe suits better on
> introduction. Less subsections. Subsections (approx.) have to be: AD prediction before
> Deep learning, Acoustic features, AD prediction with DL, Datasets in general (not only
> the ones we use)."

- **Current `state_of_the_art.tex` subsections (8 + nested):**
  1. Biological motivation: amyloid pathology and overlapping syndromes
  2. Spoken language as a scalable digital biomarker
  3. From shallow classifiers to deep neural representations
     - Self-supervised acoustic encoders: Wav2Vec 2.0
     - Contextual text encoders
  4. Benchmark challenges and public corpora
  5. Automatic differentiation of PPA variants from connected speech
  6. Multimodal fusion: pooling, cross-attention, temporal alignment
     - Self-attention / Cross-attention+gated / Mamba / Cross-modal alignment
  7. Handcrafted acoustic descriptors alongside learned embeddings
  8. Synthesis and positioning
- **Revisor's target order (approx):**
  1. AD prediction (classical / before deep learning)
  2. Acoustic features
  3. AD prediction with deep learning
  4. Datasets in general (broaden beyond ADReSSo + SPIN)
- **Actions:**
  - **Move "Biological motivation" → `introduction.tex`.** The intro already opens with
    amyloid/PPA pathology (`introduction.tex` lines 5–7) — risk of **duplication**; will
    need merging, not just moving.
  - **Cut subsection count.** Collapse the nested fusion/encoder subsubsections into prose.
  - **Reorder** to classical-AD → acoustic features → deep-learning-AD → datasets.
  - **Broaden datasets:** current §4 only really covers ADReSSo + SPIN; revisor wants a
    general datasets overview (e.g. DementiaBank/Pitt, ADReSS(o), other-language cohorts).
    May need 1–2 new citations beyond `bibliography.bib` current set.
- **Open question:** where do the *fusion taxonomy* subsubsections (self-/cross-attn, Mamba,
  alignment) go? They currently pre-stage the methodology. If SOTA shrinks, decide whether
  they stay in SOTA (trimmed) or move wholesale to Methodology §"System modules" (see F).
- Affected: `state_of_the_art.tex` (major), `introduction.tex` (absorb bio motivation).

---

## F. Methodology — reorder and restructure into "System modules" + "System architecture" — ✅ DONE
> Implemented in `main.tex`. New Methodology order:
> Problem Formulation → Datasets & Preprocessing → **System Modules** (Audio Branch:
> Encoder+Layer Pooling [merged] · Text Branch: Encoder · Handcrafted Acoustic Features
> [moved up, before fusion, framed as the third input] · Cross-Modal Alignment and Fusion
> [merged] · Classification Heads) → **System Architecture** (overview + end-to-end figure
> [relocated here] + three-implementations; Unified Implementation \texttt{res\_model}
> [demoted to subsubsection]; Feature Extractors: Single-Factor Encoder Ablation) → Training
> Protocol → Ablation Protocol → Evaluation Metrics.
> All labels preserved (`sssec:pooling`, `sssec:feature_extractors`, `ssec:architecture`,
> `ssec:res_model`, `sssec:alignment`, `sssec:acoustic_encoder`); cross-refs in Problem
> Formulation retargeted (Wav2Vec variant → `sssec:acoustic_encoder`; consolidation →
> `ssec:res_model`). Re-review pass removed redundant transitions (duplicate "Wav2Vec-pooling
> axis", "placed on a common index", "aligned…aligned", "training schedule"). Compiles clean
> (51pp, exit 0, no undefined/multiply-defined).
> Note: revisor's "fusion maybe suits results better" → chose the fold-into-alignment option
> (a), not the move-to-results option; fusion taxonomy + `tab:fusion` stay in Methodology.

> "Methodology: bad order. Merge 'Acoustic Encoder: Choice and Configuration' with
> 'Wav2Vec Pooling and Layer Mixing'. Handcrafted acoustic features section to the top.
> 'Fusion module' section is weird — maybe fuse with 'Cross-modal alignment' or
> 'Classification heads', or it suits the results section better. Make fewer sections:
> a 'System modules' section explaining the parts + their options, and a 'System
> architecture' section explaining the general model that uses those modules."

- **Current `main.tex §3 System Architecture` subsubsections:**
  1. Acoustic Encoder: Choice and Configuration (`sssec:acoustic_encoder`)
  2. Wav2Vec Pooling and Layer Mixing (`sssec:pooling`)
  3. Text Encoder: Choice and Configuration
  4. Feature-Extractor Stacks (`sssec:feature_extractors`)
  5. Cross-Modal Alignment (`sssec:alignment`)
  6. Fusion Module
  7. Handcrafted Feature Extraction and Integration
  8. Classification Heads
- **Target structure (revisor):**
  - **"System modules"** — describe each component and the *options* available for it
    (encoders+pooling, alignment schemes, fusion families, heads, handcrafted features).
  - **"System architecture"** — the general model wiring that *uses* those modules
    (the two-stream pipeline + `Figure fig:system_architecture`).
- **Specific moves:**
  - **Merge** `Acoustic Encoder` + `Wav2Vec Pooling and Layer Mixing` into one module
    block (audio branch: encoder choice + how its layers are pooled/mixed). Natural fit —
    both are the audio path.
  - **"Fusion Module" → fold into Cross-Modal Alignment — DECIDED (user).** Combine into a
    single "how the two streams meet" module: alignment places the streams on a common
    index, then fusion combines them. Keep the seven-family catalogue + `tab:fusion` inside
    this merged block. (Fusion taxonomy stays in Methodology, not moved to Results.)
  - **Handcrafted acoustic features — order by model FLOW (user clarification).** AF is a
    side-channel, **but in the model's dataflow it is one of the inputs to the fusion
    module** (prepended as a token / concatenated before the head). So it must be explained
    **before** the fusion (merged alignment+fusion) block — i.e. introduced as an input
    alongside the audio/text streams, *not* literally first of all modules and *not* buried
    after fusion (current §7 position). Net ordering of the modules should follow the flow:
    inputs (audio branch incl. pooling/mixer, text branch, **handcrafted AF**) →
    alignment+fusion → classification head.
  - **Reorder** so the section reads modules-then-architecture, not the current
    encoder-first drill-down.
- **Redundancy to resolve while restructuring** (already noted in earlier review): the
  `res_model` description appears 3× (§3.3 intro paragraph, §3.4 subsection, §3.6 Ablation
  Protocol); the staged-programme description appears ~4× (intro, §3.1, §3.6, Results
  intro). Good moment to dedupe.
- Affected: `main.tex` (§3 major rewrite).

---

## Cross-cutting / sequencing notes

- **Do A1+A3+A4+A5 as ONE Results rewrite** — they all target the same per-task /
  per-experiment / final-results-at-end structure. Don't fix piecemeal.
- **Do E + F together** if the fusion taxonomy moves between SOTA and Methodology — decide
  its home once.
- **Resolved:** A3 hybrid (axis-driven ablation body + per-task synthesis at end);
  F-fusion (fold into Cross-Modal Alignment); F-handcrafted (order by model flow, before
  fusion).
- **Still open:** B (legacy CogniAligned AF cell vs re-run res_model — your call);
  D (pick a backronym — round-3 candidates, recommend "Reproducible Evaluation Suite", or
  give me your preferred R/E/S word per slot); A3 one-line confirm with revisor that the
  hybrid counts as "per task".
- **Re-run candidate:** only B (modality × A+T+AF under res_model minimal head) — and only
  if legacy CogniAligned numbers are not accepted. Everything else is writing/restructuring,
  no new experiments.
- **Pre-existing TODOs unrelated to this list** (don't lose them): Gantt still has 2019–2020
  template data (`introduction.tex §1.2`); four `[Figure to be added]` placeholders
  (3 fusion diagrams in `main.tex`, 1 Wav2Vec architecture in `state_of_the_art.tex`).
