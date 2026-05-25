# ad-detection — SER / speech experiments (PPA)

**SER** = speech-centred or speech-heavy runs under `ad-detection` for **PPA** (or aligned cohorts). Logging **from now on**.

---

## Instructions for future agents (read first)

1. Read **Metric clarification** and **AGENT** before editing this file.
2. After each run: add one **Run summary** row; primary metric = **`max Val Acc`** over **all** validation epochs (see Metric clarification).
3. **`Experiment`** is a **1–7 word** title only; put yaml / script / git details in **`Script & config`** — not the same column (see **Table columns**).
4. Add a **Changelog** `###` entry when the run is non-trivial or you need reproducibility notes; use the template under **Changelog**.

### How to run (SLURM — WAB + acoustic features)

From `ad-detection/`: `sbatch ad_classification/run_train.sh` (uses `--acoustic_features_path` …/WAB_samples/acoustic_features.csv in the script). Logs: `wandb/slurm_ad_train_<jobid>.txt`. Scripts set `#SBATCH --exclude=veuc05,veuc01` so jobs avoid Maxwell-only GPUs; if a job still fails with `cudaErrorNoKernelImageForDevice`, add that node to `--exclude`. Check placement with `squeue -u $USER` and `sacct -j JOBID`.

---

## Table columns: `Experiment` vs `Script & config`

| Column | Purpose |
| :--- | :--- |
| **`Experiment`** | **1–7 words** summarising the idea of the run (like P3’s “Baseline LSTM”). |
| **`Script & config`** | Entry script path, yaml, CLI flags, branch/commit. |

---

## Workflow

1. Run experiment (e.g. `ad_classification/train_ad.py` or future PPA-specific entrypoint + config).
2. Collect **validation accuracy for every epoch** (or step where validation runs).
3. Append **Run summary**: **`Experiment`**, **`Script & config`**, then metrics.
4. Add **Changelog** from template when useful.

---

## Metric clarification (human + agent — read this first)

**Primary score for comparing runs:** **`max Val Acc`** = **maximum validation accuracy achieved in any epoch** during the run.

- **Do not** use validation accuracy at the **last epoch** as the headline metric unless it equals the max; if logged separately, column **`Val Acc last`**.
- **K-fold:** Per fold → max val acc over epochs → **`max Val Acc (mean)`** = mean of those maxima.
- **`epoch @ max`:** Epoch where validation accuracy attains **`max Val Acc`** (state 0-based vs 1-based once in `Notes`).

---

## AGENT: how to auto-fill this log

1. Parse training output / W&B / CSV for **`val_accuracy`** (or equivalent) **per epoch**.
2. Set **`max Val Acc`** = numerical **maximum** of that series (per fold, then aggregate as documented in the table header).
3. Set **`epoch @ max`** = epoch index at that maximum (per fold or for single split).
4. **Never** assume the last printed validation line is the best; always scan the full curve.
5. If the logger only stores “best checkpoint”, you may copy that value into **`max Val Acc`** and put “from best_ckpt callback” in **`Notes`**.

---

## What this file is for

- **Repo:** `ad-detection/`
- **Focus:** PPA-related speech pipeline.
- **Likely scripts:** `ad_classification/train_ad.py` (or dedicated PPA trainer).
- **Related:** CogniAligned PPA log: `CogniAligned/cogni_ppa_experiments.md` (multimodal; same **max Val Acc** convention).

---

## How to append (field cheat-sheet)

| Field | Record |
| :--- | :--- |
| `ID` | SLURM / W&B / git tag |
| **`Experiment`** | **1–7 words**; Changelog title |
| `Script & config` | CLI + yaml path + branch if relevant |
| `Split` | Patient-level? K-fold? |
| **`max Val Acc (mean)`** | Best-over-epochs; mean over folds if CV |
| **`epoch @ max`** | As above |
| **`max macro-F1 (mean)`** | If logged; same max-over-epochs rule |
| **`Val Acc last`** | Optional |
| `Notes` | Diarization, augmentation, label definition |

---

## Run summary (living table)

| ID | Date | **Experiment** | Script & config | Split | **max Val Acc (mean)** | **epoch @ max** | **max macro-F1 (mean)** | Val Acc last (opt.) | **Params** | **Infer. (ms/spl)** | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2538660 | 2026-05-05 | SER PPA WAB plus acoustic | `sbatch ad_classification/run_train.sh` → `train_ad.py`; `--acoustic_features_path` WAB_samples/acoustic_features.csv; Wav2Vec2 XLSR-300M + ModernBERT | default train/val from script | **92.86%** | 22 (epoch index in log) | **0.9214** | 92.86% (ep 24) | SLURM log: `wandb/slurm_ad_train_2538660.txt`; node veuc10 | — || — | |
| 2553310,2553312–2553314 | 2026-05-10 | SER PPA Mamba 4-fold CV | `sbatch ad_classification/launch_cv_mamba.sh` → `run_train_mamba.sh` → `MambaSeqToSeq` n_blocks=4; acoustic AF; 25 epochs; plateau LR | 5-fold CV (fold 1 resubmitted as 2563974 — pending) | **~69.49%** *(4/5 folds)* | per-fold ep 10, —, 27, 25, 23 | **~0.6541** | — | Per-fold acc: 58.33%, —, 67.24%, 78.57%, 73.81%. Fold 1 (2553311) SIGSEGV on veuc11; resubmitted as 2563974 excl. veuc01,05,11. Logs: `wandb/slurm_ppa_mamba_fold*_255331*.txt` | — || — | |
| 2569628–2569629–2569630–2569631–2569632 | 2026-05-10 | SelfAttention (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **71.39%** | — | **0.6101** | — | Per-fold acc: 58.33%, 70.00%, 65.52%, 82.14%, 80.95%; ~849.1 batches/min. Logs: `wandb/slurm_*_2569628–2569629–2569630–2569631–2569632.txt` | — || — | |
| 2569643–2569644–2569645–2569646–2569647 | 2026-05-10 | MultiHeadAttention-2 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **65.02%** | — | **0.5603** | — | Per-fold acc: 58.33%, 65.00%, 67.24%, 60.71%, 73.81%; ~906.4 batches/min. Logs: `wandb/slurm_*_2569643–2569644–2569645–2569646–2569647.txt` | — || — | |
| 2569658–2569659–2569660–2569661–2569662 | 2026-05-10 | MultiHeadAttention-4 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **68.64%** | — | **0.6032** | — | Per-fold acc: 61.11%, 72.50%, 65.52%, 67.86%, 76.19%; ~918.0 batches/min. Logs: `wandb/slurm_*_2569658–2569659–2569660–2569661–2569662.txt` | — || — | |
| 2569673–2569674–2569675–2569676–2569677 | 2026-05-10 | CrossAttentionReduced-4 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **69.50%** | — | **0.6004** | — | Per-fold acc: 58.33%, 82.50%, 63.79%, 71.43%, 71.43%; ~928.6 batches/min. Logs: `wandb/slurm_*_2569673–2569674–2569675–2569676–2569677.txt` | — || — | |
| 2569780–2569781–2569782–2569783–2569784 | 2026-05-10 | Mamba-50ep-lr5e-5 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **64.12%** | — | **0.5481** | — | Per-fold acc: 58.33%, N/A, 51.72%, 75.00%, 71.43%; ~684.6 batches/min. Logs: `wandb/slurm_*_2569780–2569781–2569782–2569783–2569784.txt` | — || — | |
| 2569795–2569796–2569797–2569798–2569799 | 2026-05-10 | Mamba-50ep-lr1e-4 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **67.98%** | — | **0.6072** | — | Per-fold acc: 50.00%, 87.50%, 65.52%, 60.71%, 76.19%; ~809.2 batches/min. Logs: `wandb/slurm_*_2569795–2569796–2569797–2569798–2569799.txt` | — || — | |
---

## Changelog

Style: `example_experiment.md` (repo root). Append new entries at **bottom**.

### Changelog template (copy per run)

```markdown
### Experiment <ID>: <1–7 word title>

**Status:** Complete ✓ | In progress | Failed  
**Results:** max Val Acc (mean) = … \| epoch @ max = … \| max macro-F1 (mean) = … \| Val Acc last = … (optional)

**Config:**
- Script, yaml, branch, CLI:

**What changed:**

**Observations:**

**Artifacts:** SLURM log, W&B, checkpoints
```

### Experiment 2538660: SER PPA WAB plus acoustic

**Status:** Complete ✓  
**Results:** max Val Acc = **92.86%** | epoch @ max = **22** | max macro-F1 = **0.9214** | Val Acc last = 92.86% (epoch 24)

**Config:**
- `ad_classification/run_train.sh` → `train_ad.py` with acoustic CSV on WAB_samples; 25 epochs, batch 4, weighted loss.

**What changed:** First logged “new era” SER PPA run with structured metrics (max over epochs).

**Artifacts:** `/home/usuaris/veussd/roger.esteve.sanchez/ad-detection/wandb/slurm_ad_train_2538660.txt`

### Experiments 2553310,2553312–2553314 (+2563974): SER PPA Mamba 5-fold CV

**Status:** 4/5 folds complete ✓; fold 1 resubmitted as 2563974 (pending, excluded veuc01,05,11)  
**Results (4 folds):** max Val Acc (mean) = **~69.49%** | per-fold epoch @ max: 10, —, 27, 25, 23 | macro-F1 (mean) = **~0.6541**

**Config:** `ad_classification/run_train_mamba.sh`; `MambaSeqToSeq` (n_blocks=4, d_state=16, d_conv=4, expand=2); `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; 25 epochs; AdamW lr=5e-5 wd=0.01; plateau LR; weighted loss; batch 4. Fold 1 (2553311) crashed with SIGSEGV on veuc11 after 3 min; resubmitted as 2563974 excluding veuc01,05,11.

**What changed vs 2538660:** MambaSeqToSeq replaces SelfAttention; 5-fold CV replaces single split.

**Observations:**
- Per-fold max Val Acc: **58.33%** (f0), *pending* (f1), **67.24%** (f2), **78.57%** (f3), **73.81%** (f4)
- Large fold variance (58%–79%) typical for small PPA dataset.
- Substantially underperforms SER SelfAttention (92.86% single-split). Note baseline is single-split vs CV, so direct comparison is approximate.
- Fold 3 (78.57%) approaches SelfAttention territory — Mamba learns well when the split is favourable.

**Artifacts:** `wandb/slurm_ppa_mamba_fold{0,2–4}_255331{0,2–4}.txt`; fold 1 → `wandb/slurm_ppa_mamba_fold1_2563974.txt` (pending)

---


### Experiment 2569628–2569629–2569630–2569631–2569632: SelfAttention (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **71.39%** | max macro-F1 (mean) = **0.6101** | **Params:** 612K | **Inference:** 0.79 ms/sample

**Config:** `run_train_generic.sh`; `SelfAttention`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 58.33%, 70.00%, 65.52%, 82.14%, 80.95%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569628–2569629–2569630–2569631–2569632.txt`


### Experiment 2569643–2569644–2569645–2569646–2569647: MultiHeadAttention-2 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **65.02%** | max macro-F1 (mean) = **0.5603** | **Params:** 1.14M | **Inference:** 1.59 ms/sample

**Config:** `run_train_generic.sh`; `MultiHeadAttention-2`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 58.33%, 65.00%, 67.24%, 60.71%, 73.81%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569643–2569644–2569645–2569646–2569647.txt`


### Experiment 2569658–2569659–2569660–2569661–2569662: MultiHeadAttention-4 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **68.64%** | max macro-F1 (mean) = **0.6032** | **Params:** 1.66M | **Inference:** 2.10 ms/sample

**Config:** `run_train_generic.sh`; `MultiHeadAttention-4`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 61.11%, 72.50%, 65.52%, 67.86%, 76.19%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569658–2569659–2569660–2569661–2569662.txt`


### Experiment 2569673–2569674–2569675–2569676–2569677: CrossAttentionReduced-4 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **69.50%** | max macro-F1 (mean) = **0.6004** | **Params:** 1.66M | **Inference:** 0.73 ms/sample

**Config:** `run_train_generic.sh`; `CrossAttentionReduced-4`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 58.33%, 82.50%, 63.79%, 71.43%, 71.43%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569673–2569674–2569675–2569676–2569677.txt`


### Experiment 2569780–2569781–2569782–2569783–2569784: Mamba-50ep-lr5e-5 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **64.12%** | max macro-F1 (mean) = **0.5481** | **Params:** 2.36M | **Inference:** 1.90 ms/sample

**Config:** `run_train_generic.sh`; `Mamba-50ep-lr5e-5`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 58.33%, N/A, 51.72%, 75.00%, 71.43%
- 4/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569780–2569781–2569782–2569783–2569784.txt`


### Experiment 2569795–2569796–2569797–2569798–2569799: Mamba-50ep-lr1e-4 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **67.98%** | max macro-F1 (mean) = **0.6072** | **Params:** 2.36M | **Inference:** 1.89 ms/sample

**Config:** `run_train_generic.sh`; `Mamba-50ep-lr1e-4`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 50.00%, 87.50%, 65.52%, 60.71%, 76.19%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569795–2569796–2569797–2569798–2569799.txt`

## Baseline hints (indirect — not SER-PPA-specific)

`ad_classification/experiments.md` reports historical accuracies (~**60.5%** CV-style, peaks ~**64%**) that may **not** follow the max-over-epochs convention — **do not copy into this table without recomputing `max Val Acc`**.

**Baseline:** job **2538660** (table above) is the first structured SER-PPA row in this file.

---

## Open points

- [ ] Canonical config YAML for PPA under `ad-detection`.
- [ ] Note audio pipeline: diarized patient-only vs full mix (`diarization_pipeline/`).
