# Thesis Ablation Plan

This file plans a **clean, sequenced** ablation built on top of the thesis sweep
already in `experiment_results.csv`. It is intentionally separate from
`experiments_plan.md` (which logged the original 108-job thesis sweep) so the
two records stay decoupled.

Three phases, in order of dependency:

- **Step A — Modality ablation** (cheap, establishes the floor)
- **Step B — Fusion ablation** (architecture comparison, reuses most existing runs)
- **Step C — Alignment ablation** (the new contribution)

All experiments use:

- 5-fold stratified CV per dataset (already-saved split files in
  `CogniAligned/.../splits/`).
- Datasets: **ADReSSo**, **Amyloid**, **PPA** (3 × everything below).
- Reported metrics: `val_acc_mean`, `best_fold_acc`, `inference_ms/sample`,
  total / trainable params, and (ADReSSo only) `test_acc` on `task1.csv`.
- Unique `experiment_tag` per row so checkpoints never overwrite each other
  (`models/<tag>/...` for SER, `logs/<task>_<tag>/...` for CogniAligned).
- W&B logging on the `CogniAligned` and `ad-multiclass-detection` projects,
  grouped by phase: `groupA_modality`, `groupB_fusion`, `groupC_alignment`.

---

## Step A — Modality ablation

**Goal:** quantify what each modality (and the AF token) contributes on top of a
minimal, identical fusion backbone. This is the "what does the data have to
give us at all" floor.

**Where it runs:** CogniAligned only. Reason: in `ad-detection` every
`seq_to_seq` layer hard-codes `torch.cat((speech, text), ...)`, so audio-only
or text-only requires patching the model. CogniAligned natively supports both
via `config.model.multimodality = false` (it dispatches to `MyTransformerEncoder`,
a plain `nn.TransformerEncoder` over one sequence).

**Architecture for all Step-A rows (fixed):** `MyTransformerEncoder` with
`n_layers=1`, `n_heads=1`, `hidden_size=768`, `intermediate_size=3072`,
`pooling=mean`. This is the smallest "self-attention encoder" we can run and
keeps Step A about *modalities*, not architectures.

| # | dataset × {ADReSSo, Amyloid, PPA} | audio | text | AF | multimodality | tag prefix |
|---|---|---|---|---|---|---|
| A1 | audio only, no AF | ✅ | ❌ | ❌ | false | `abl_a1_audio` |
| A2 | text only, no AF | ❌ | ✅ | ❌ | false | `abl_a2_text` |
| A3 | audio + text, mean fusion, no AF | ✅ | ✅ | ❌ | true (`fusion: mean`) | `abl_a3_av_mean` |
| A4 | audio + text, mean fusion + AF | ✅ | ✅ | ✅ | true (`fusion: mean`, `use_acoustic_features: true`) | `abl_a4_av_mean_af` |

**Total runs:** 4 cells × 3 datasets × 5 folds = **60 jobs**.

**Configs to author:** `CogniAligned/modules/configs/ablation/{A1..A4}/{adresso,amyloid,ppa}.yaml`.
Each one only overrides `model.multimodality`, `model.audio_model`,
`model.textual_model`, `model.fusion`, `model.use_acoustic_features`, and the
data paths. Everything else inherits from the existing `default.yaml`.

**What we expect to learn:** how much the second modality adds at all and how
much AF as a token contributes. Result feeds into how we interpret Step B/C.

---

## Step B — Fusion ablation (audio + text, no AF, both models)

**Goal:** isolate the fusion architecture's contribution at fixed inputs.

**Status:** **DONE.** Results in `step_b_results.csv` (42 rows = 7 archs ×
2 models × 3 datasets). Orchestrated by `run_step_b.py`.

What was done in this pass:
- Reused 30 existing thesis cells on disk (CogniAligned: 6/7 archs, SER: 4/7 archs across all datasets).
- Trained the 12 missing cells: CogniAligned `transformerpe` × 3, SER `transformerpe` × 3 × 5 folds, SER `gatedcrossattn` × 3 × 5, SER `mamba` × 3 × 5.
- Fixed two pre-existing SER bugs uncovered during retraining (`TransformerBlock`/`MultiHeadAttention`/`SelfAttention` forward signatures became polymorphic; `GatedCrossAttention` gate now accepts the 2·emb_in concat it was built for).
- Patched `adresso_classify/test_adresso.py` to time inference per sample, accept multi-fold checkpoints for ensembling, and emit a parseable `TEST_RESULT` summary line.
- Ran ADReSSo test inference for all 14 ADReSSo cells (7 archs × 2 models) using the trained checkpoints.

Architecture set (identical names for SER and CogniAligned where applicable):

| Method | SER `seq_to_seq_method` | CogniAligned `fusion` |
|---|---|---|
| Self-attention (1 head) | `SelfAttention` | `selfattn` (ElementWiseFusion, `n_heads=1`) |
| Multi-head self-attention | `MultiHeadAttention` (heads ∈ {2, 4}) | `mhselfattn` (`n_heads` ∈ {2, 4}) |
| Transformer w/ positional encoding | `TransformerPosEnc` | `transformerpe` |
| Cross-attention (unidirectional) | `CrossAttention` (= `CrossAttentionReduced`) | `cross` |
| Gated cross-attention | `GatedCrossAttention` | `crossgated` |
| Bidirectional cross-attention | `BidirectionalCrossAttention` | `bicross` |
| Mamba | `MambaSeqToSeq` | `mamba` |

**Shared settings:** lr=2e-5 (mamba also sweeps 1e-4 for SER and 1e-5 / 5e-5 for
CogniAligned, kept from the original sweep), AdamW + cosine schedule,
weight_decay=0.1, dropout=0.3, 25 epochs (SER) / 200 max + early-stop 15 epochs
(CogniAligned), 5 folds.

**No new training is required for cells that already populate
`experiment_results.csv`**. The "missing" check before launching is just:

```
rows missing test_acc on ADReSSo  → re-run ADReSSo test inference
rows missing best_fold_acc        → re-parse logs (no retrain)
```

**Total *new* runs for Step B:** 0 if the CSV is already complete, otherwise
only the missing cells get retrained.

---

## Step C — Alignment ablation (NEW)

**Goal:** answer the core question of the thesis — does CogniAligned's
**word-to-token** alignment actually matter, and can a finer **token-to-token**
alignment do better?

**Implementation status:** preprocessing now supports four modes via
`--alignment_mode` (added in `CogniAligned/modules/preprocess/run_preprocessing.py`).
The audio file's filename gets a mode-specific suffix so all variants coexist:

| `alignment_mode` | audio filename suffix | meaning |
|---|---|---|
| `word_token` *(current default)* | `` (none) | broadcast one mean per word to every subword |
| `token_token_uniform` | `_tokalign` | split each word's audio span uniformly across its subwords |
| `token_token_char` | `_tokalignchar` | split proportionally to subword character lengths |
| `none` | `_noalign` | temporal-mean-pool raw audio to `max_length`; no alignment |

Configs select a variant with `data.alignment_suffix: <suffix>`. The
dataloader (`CogniAligned/modules/dataset.py`) now reads that field
(default empty → backwards compatible).

### Preprocessing runs needed (one-off, per dataset)

```bash
# from CogniAligned/modules/preprocess
python run_preprocessing.py --alignment_mode token_token_uniform
python run_preprocessing.py --alignment_mode token_token_char
python run_preprocessing.py --alignment_mode none
# repeat with --test for the ADReSSo test-dist set so test inference works
python run_preprocessing.py --alignment_mode token_token_uniform --test
python run_preprocessing.py --alignment_mode token_token_char    --test
python run_preprocessing.py --alignment_mode none                --test
```

Whisper transcription is cached, so each rerun only redoes Step 2 (embeddings).
Amyloid and PPA have their own preprocessing scripts (`modules/{amyloid,ppa}/...`);
the same `alignment_mode` global needs to be set there. I'll add the flag to
those wrappers when we get to running Step C on those datasets (cheap mirror
of the same patch).

### Training matrix

To keep Step C interpretable, fix everything **except** alignment, with the
fusion family chosen to be representative across the alignment spectrum:

- `cross` — already best in `experiment_results.csv`; benefits most from token-level alignment.
- `bicross` — strongest published CogniAligned variant.
- `mamba` — sequence-model perspective; uses concat-along-time, so alignment shape matters.

For the `none` mode we can only meaningfully fuse with cross / bicross / gated-cross
/ mamba (everything else assumes aligned same-length sequences). Element-wise
fusions are skipped in the `none` row.

| # | alignment | fusion | datasets | tag prefix |
|---|---|---|---|---|
| C1 | word_token (current) | cross | A, Am, P | `abl_c1_wt_cross` |
| C2 | word_token (current) | bicross | A, Am, P | `abl_c2_wt_bicross` |
| C3 | word_token (current) | mamba | A, Am, P | `abl_c3_wt_mamba` |
| C4 | token_token_uniform | cross | A, Am, P | `abl_c4_tu_cross` |
| C5 | token_token_uniform | bicross | A, Am, P | `abl_c5_tu_bicross` |
| C6 | token_token_uniform | mamba | A, Am, P | `abl_c6_tu_mamba` |
| C7 | token_token_char | cross | A, Am, P | `abl_c7_tc_cross` |
| C8 | token_token_char | bicross | A, Am, P | `abl_c8_tc_bicross` |
| C9 | token_token_char | mamba | A, Am, P | `abl_c9_tc_mamba` |
| C10 | none | cross | A, Am, P | `abl_c10_no_cross` |
| C11 | none | bicross | A, Am, P | `abl_c11_no_bicross` |
| C12 | none | mamba | A, Am, P | `abl_c12_no_mamba` |

**Rows C1, C2, C3 are FREE** — they already exist in `experiment_results.csv`
under the `cross / bicross / mamba` CogniAligned rows. We just rename/tag them.

**Total *new* training runs for Step C:** 9 cells × 3 datasets × 5 folds =
**135 jobs**, plus 3 preprocessing passes × {train, test-dist} = 6 short CPU/GPU
preprocessing jobs.

### Pilot (current scope — user-approved)

**3 datasets × 3 alignment techniques × `cross` fusion** (CogniAligned only):

| Alignment | Code name | Preprocessing suffix |
|---|---|---|
| Word → token (CogniAligned original) | `word_token` | *(none)* — **reuse** existing `cross` checkpoints |
| Token → token uniform | `token_token_uniform` | `_tokalign` |
| Token → token proportional (char) | `token_token_char` | `_tokalignchar` |

Datasets: **ADReSSo**, **Amyloid**, **PPA**.

Orchestrator: `run_alignment_pilot.py` → `alignment_pilot.csv` (+ `_es.csv`).

**New work in pilot:**
- Preprocessing: ADReSSo train+test for uniform & char; WAB train for uniform & char (covers Amyloid+PPA).
- Training: **6 jobs** (2 alignments × 3 datasets; word_token skipped).
- Test inference: **3 ADReSSo** runs (all alignments; word_token can reuse Step B or re-test).

**Not in this pilot:** `none` alignment, bicross/mamba fusions (saved for full Step C matrix).

---

## Reporting

A new CSV `ablation_results.csv` will be appended to alongside the existing
`experiment_results.csv` with the columns:

```
phase, model, dataset, alignment, fusion, lr, total_params, trainable_params,
inference_ms_per_sample, val_acc_mean, best_fold_acc, test_acc, note
```

`phase ∈ {A, B, C}`, `alignment ∈ {word_token, token_token_uniform, token_token_char, none, n/a}`.
For Step-B rows reused from the original sweep, `alignment = word_token` (the
preprocessing they consumed) and `phase = B`.

---

## Execution order

1. **A1–A4 on ADReSSo** (12 jobs) — sanity-check the modality floor.
2. **A1–A4 on Amyloid + PPA** (48 jobs).
3. **C-pilot on ADReSSo** (15 training jobs + 3 preprocessing).
4. Triage pilot — if alignment doesn't move accuracy beyond noise, stop and
   write up Step C as a negative result.
5. Otherwise: extend C to Amyloid + PPA (120 more training jobs + 6 preprocessing).
6. ADReSSo test inference for everything new; refresh CSVs.

Steps 1–2 can launch right now without further code changes. Step 3 needs the
new preprocessing to be run first; everything for that is already wired in
`run_preprocessing.py --alignment_mode ...`.
