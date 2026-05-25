# ad-detection — SER / speech experiments (ADReSSo)

Speech-centred or speech-heavy runs under `ad-detection` for **ADReSSo-style** classification (e.g. HC vs AD). Logging **from now on**.

---

## Instructions for future agents (read first)

1. Read **Metric clarification** and **AGENT** before editing.
2. Add one **Run summary** row per training run; use **`max Val Acc`** from the **full** val-accuracy curve.
3. **`Experiment`** (1–7 words) is the **title** column, like P3’s “Experiment” column — **not** the same as **`Script & config`** (see **Table columns**).
4. Use **Changelog** for config-heavy or surprising results; copy the template below.

### How to run (SLURM — ADReSSo + acoustic features)

From `ad-detection/`: `sbatch adresso_classify/run_train_af.sh`. Logs: `wandb/slurm_adresso_train_af_<jobid>.txt`. Scripts exclude `veuc05,veuc01`; use `squeue` / `sacct` to confirm completion.

---

## Table columns: `Experiment` vs `Script & config`

| Column | Purpose |
| :--- | :--- |
| **`Experiment`** | **1–7 word** human summary (e.g. “HuBERT MFCCs no text”). |
| **`Script & config`** | `train_adresso.py`, yaml, CLI, git ref. |

---

## Workflow

1. Run (`adresso_classify/train_adresso.py` + config).
2. Record **validation accuracy per epoch**.
3. Append **Run summary** with **`Experiment`** + **`Script & config`** + metrics.
4. Add **Changelog** when useful.

---

## Metric clarification (human + agent — read this first)

**Primary score for comparing runs:** **`max Val Acc`** = **maximum validation accuracy over the entire training run** (any epoch).

- **`Val Acc last`** = validation accuracy at final epoch only — **not** the primary metric.
- **K-fold:** Compute max val acc per fold over epochs, then **`max Val Acc (mean)`** = mean of those maxima.
- **`epoch @ max`:** Epoch where val acc reaches **`max Val Acc`** (per fold or single run; state epoch indexing once in `Notes`).

**Secondary:** **`max macro-F1`** with the same best-over-epochs rule when logged.

---

## AGENT: how to auto-fill this log

1. From logs / W&B / CSV, extract validation accuracy for **each** epoch.
2. Set **`max Val Acc`** to the **maximum** of that sequence (never assume the final epoch is best).
3. Set **`epoch @ max`** to the argmax epoch.
4. Cross-validation: aggregate as **`max Val Acc (mean)`** over folds; list per-fold **`max Val Acc`** in `Notes` if spread is large.
5. If only a single “best checkpoint” line exists, map it to **`max Val Acc`** and cite the source in **`Notes`**.

---

## What this file is for

- **Repo:** `ad-detection/`
- **Likely entrypoint:** `adresso_classify/train_adresso.py`
- **Related:** `CogniAligned/cogni_adresso_experiments.md`

---

## How to append (field cheat-sheet)

| Field | Record |
| :--- | :--- |
| `ID` | Job id / run name |
| **`Experiment`** | **1–7 words** |
| `Script & config` | CLI + yaml + branch |
| `Split` | Official vs custom CV |
| **`max Val Acc (mean)`** | Best-over-epochs; mean over folds if CV |
| **`epoch @ max`** | As above |
| **`max macro-F1 (mean)`** | If logged |
| **`Val Acc last`** | Optional |
| `Notes` | Baseline comparisons, log path |

---

## Run summary (living table)

| ID | Date | **Experiment** | Script & config | Split | **max Val Acc (mean)** | **epoch @ max** | **max macro-F1 (mean)** | Val Acc last (opt.) | **Params** | **Infer. (ms/spl)** | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2538669 | 2026-05-05 | SER ADReSSo acoustic AF | `sbatch adresso_classify/run_train_af.sh` → `train_adresso.py` with ADReSSo `acoustic_features.csv` (script default paths) | segment-level val (log: Val Segment Acc) | **80.00%** | 15 | **0.7969** | 72.50% | `wandb/slurm_adresso_train_af_2538669.txt`; veuc09 | — || — | |
| 2553290–2553294 | 2026-05-09 | SER ADReSSo Mamba 5-fold CV | `sbatch adresso_classify/launch_cv_mamba.sh` → `run_train_mamba.sh` → `MambaSeqToSeq` n_blocks=4 d_state=16 d_conv=4 expand=2; acoustic AF; 25 epochs; plateau LR | 5-fold segment-level CV | **~77.63%** *(fold 1 still running on veuc09)* | per-fold ep 21, —, 15, 10, 23 | **~0.7152** | — | Folds 0/2/3/4 complete; fold 1 (2553291) in progress at ep 10/25 best=77.59%. Per-fold acc: 78.00%, —, 81.48%, 76.09%, 75.00%. Logs: `wandb/slurm_adresso_mamba_fold*_255329*.txt` | — || — | |
| 2569618–2569619–2569620–2569621–2569622 | 2026-05-10 | SelfAttention (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **83.98%** | — | **0.8355** | — | Per-fold acc: 78.00%, 81.03%, 88.89%, 86.96%, 85.00%; ~1041.8 batches/min. Logs: `wandb/slurm_*_2569618–2569619–2569620–2569621–2569622.txt` | — || — | |
| 2569633–2569634–2569635–2569636–2569637 | 2026-05-10 | MultiHeadAttention-2 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **82.79%** | — | **0.8216** | — | Per-fold acc: 72.00%, 72.41%, 92.59%, 86.96%, 90.00%; ~948.0 batches/min. Logs: `wandb/slurm_*_2569633–2569634–2569635–2569636–2569637.txt` | — || — | |
| 2569648–2569649–2569650–2569651–2569652 | 2026-05-10 | MultiHeadAttention-4 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **83.60%** | — | **0.8274** | — | Per-fold acc: 82.00%, 81.03%, 85.19%, 84.78%, 85.00%; ~820.7 batches/min. Logs: `wandb/slurm_*_2569648–2569649–2569650–2569651–2569652.txt` | — || — | |
| 2569663–2569664–2569665–2569666–2569667 | 2026-05-10 | CrossAttentionReduced-4 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **82.85%** | — | **0.8215** | — | Per-fold acc: 82.00%, 77.59%, 87.04%, 82.61%, 85.00%; ~1020.5 batches/min. Logs: `wandb/slurm_*_2569663–2569664–2569665–2569666–2569667.txt` | — || — | |
| 2569770–2569771–2569772–2569773–2569774 | 2026-05-10 | Mamba-50ep-lr5e-5 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **87.20%** | — | **0.8631** | — | Per-fold acc: 80.00%, 94.83%, 87.04%, 89.13%, 85.00%; ~782.9 batches/min. Logs: `wandb/slurm_*_2569770–2569771–2569772–2569773–2569774.txt` | — || — | |
| 2569770–2569771–2569772–2569773–2569774 | 2026-05-10 | Mamba-50ep-lr5e-5 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **87.20%** | — | **0.8631** | — | Per-fold acc: 80.00%, 94.83%, 87.04%, 89.13%, 85.00%; ~782.9 batches/min. Logs: `wandb/slurm_*_2569770–2569771–2569772–2569773–2569774.txt` | — || — | |
| 2569785–2569786–2569787–2569788–2569789 | 2026-05-10 | Mamba-50ep-lr1e-4 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **83.19%** | — | **0.8193** | — | Per-fold acc: 82.00%, 82.76%, 90.74%, 80.43%, 80.00%; ~832.8 batches/min. Logs: `wandb/slurm_*_2569785–2569786–2569787–2569788–2569789.txt` | — || — | |
---

## Changelog

Style: `example_experiment.md`. Append at **bottom**.

### Changelog template (copy per run)

```markdown
### Experiment <ID>: <1–7 word title>

**Status:** Complete ✓ | In progress | Failed  
**Results:** max Val Acc (mean) = … \| epoch @ max = … \| max macro-F1 (mean) = … \| Val Acc last = … (optional)

**Config:**
- Script, yaml, split definition:

**What changed:**

**Observations:**

**Artifacts:**
```

### Experiment 2538669: SER ADReSSo acoustic AF

**Status:** Complete ✓  
**Results:** max Val Acc = **80.00%** (Val Segment Acc) | epoch @ max = **15** | max macro-F1 = **0.7969** | Val Acc last = 72.50%

**Config:** `adresso_classify/run_train_af.sh` as committed; acoustic features path bundled for ADReSSo train split.

**Artifacts:** `/home/usuaris/veussd/roger.esteve.sanchez/ad-detection/wandb/slurm_adresso_train_af_2538669.txt`

### Experiments 2553290–2553294: SER ADReSSo Mamba 5-fold CV

**Status:** 4/5 folds complete ✓; fold 1 (2553291) in progress on veuc09 (slow node, ~1.35 batch/min)  
**Results (4 folds):** max Val Acc (mean) = **~77.63%** | per-fold epoch @ max: 21, —, 15, 10, 23 | macro-F1 (mean) = **~0.7152**

**Config:** `adresso_classify/run_train_mamba.sh`; `MambaSeqToSeq` (n_blocks=4, d_state=16, d_conv=4, expand=2); `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; 25 epochs; AdamW lr=5e-5 wd=0.01; plateau LR (factor=0.5, patience=10); warmup 5 epochs; batch 4.

**What changed vs 2538669:** Replaced `SelfAttention` seq-to-seq with pure-PyTorch `MambaSeqToSeq`; running 5-fold CV instead of single split.

**Observations:**
- Per-fold max Val Acc: **78.00%** (f0), *pending* (f1), **81.48%** (f2), **76.09%** (f3), **75.00%** (f4)
- Mamba **underperforms** SelfAttention baseline (80.00% single-split vs ~77.6% CV mean) by ~2–3 pp; however split mismatch makes direct comparison noisy — proper CV for SelfAttention baseline still missing.
- Fold 2 (veuc11) is the highest-accuracy fold despite being a slower node.
- Fold 3 converged fastest (ep 10) suggesting Mamba may need fewer epochs when data is well-distributed.

**Artifacts:** `wandb/slurm_adresso_mamba_fold{0–4}_255329{0–4}.txt`

---


### Experiment 2569618–2569619–2569620–2569621–2569622: SelfAttention (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.98%** | max macro-F1 (mean) = **0.8355** | **Params:** 611K | **Inference:** 0.75 ms/sample

**Config:** `run_train_generic.sh`; `SelfAttention`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 78.00%, 81.03%, 88.89%, 86.96%, 85.00%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569618–2569619–2569620–2569621–2569622.txt`


### Experiment 2569633–2569634–2569635–2569636–2569637: MultiHeadAttention-2 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **82.79%** | max macro-F1 (mean) = **0.8216** | **Params:** 1.14M | **Inference:** 1.59 ms/sample

**Config:** `run_train_generic.sh`; `MultiHeadAttention-2`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 72.00%, 72.41%, 92.59%, 86.96%, 90.00%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569633–2569634–2569635–2569636–2569637.txt`


### Experiment 2569648–2569649–2569650–2569651–2569652: MultiHeadAttention-4 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.60%** | max macro-F1 (mean) = **0.8274** | **Params:** 1.66M | **Inference:** 2.62 ms/sample

**Config:** `run_train_generic.sh`; `MultiHeadAttention-4`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 82.00%, 81.03%, 85.19%, 84.78%, 85.00%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569648–2569649–2569650–2569651–2569652.txt`


### Experiment 2569663–2569664–2569665–2569666–2569667: CrossAttentionReduced-4 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **82.85%** | max macro-F1 (mean) = **0.8215** | **Params:** 1.66M | **Inference:** 0.80 ms/sample

**Config:** `run_train_generic.sh`; `CrossAttentionReduced-4`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 82.00%, 77.59%, 87.04%, 82.61%, 85.00%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569663–2569664–2569665–2569666–2569667.txt`


### Experiment 2569770–2569771–2569772–2569773–2569774: Mamba-50ep-lr5e-5 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **87.20%** | max macro-F1 (mean) = **0.8631** | **Params:** 2.36M | **Inference:** 1.85 ms/sample

**Config:** `run_train_generic.sh`; `Mamba-50ep-lr5e-5`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 80.00%, 94.83%, 87.04%, 89.13%, 85.00%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569770–2569771–2569772–2569773–2569774.txt`


### Experiment 2569770–2569771–2569772–2569773–2569774: Mamba-50ep-lr5e-5 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **87.20%** | max macro-F1 (mean) = **0.8631** | **Params:** 2.36M | **Inference:** 1.85 ms/sample

**Config:** `run_train_generic.sh`; `Mamba-50ep-lr5e-5`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 80.00%, 94.83%, 87.04%, 89.13%, 85.00%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569770–2569771–2569772–2569773–2569774.txt`


### Experiment 2569785–2569786–2569787–2569788–2569789: Mamba-50ep-lr1e-4 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.19%** | max macro-F1 (mean) = **0.8193** | **Params:** 2.36M | **Inference:** 1.83 ms/sample

**Config:** `run_train_generic.sh`; `Mamba-50ep-lr1e-4`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 82.00%, 82.76%, 90.74%, 80.43%, 80.00%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569785–2569786–2569787–2569788–2569789.txt`

## Baseline hints (legacy — do not assume max-over-epochs)

`ad_classification/experiments.md` (~**60.5%** CV, ~**62–64%** peaks) and CogniAlign job **2398091** (**~87.8%**) are **different tasks/stacks** and may use **last-epoch** or other reporting — **recompute `max Val Acc` from curves** before entering them as comparable rows in this table.

---

## Open points

- [ ] ADReSSo official train/dev vs custom split.
- [ ] Link diarization / transcription artefacts if applicable.
