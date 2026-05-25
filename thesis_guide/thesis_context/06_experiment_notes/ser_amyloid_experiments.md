# ad-detection — SER / speech experiments (Amyloid)

Speech-centred or speech-heavy runs under `ad-detection` for **amyloid** prediction. Logging **from now on**.

---

## Instructions for future agents (read first)

1. Read **Metric clarification** and **AGENT** before editing.
2. One **Run summary** row per run; headline = **`max Val Acc`** over all val epochs (not **`Val Acc last`** unless documented).
3. **`Experiment`** = **1–7 word** summary; **`Script & config`** = technical details — separate columns (see **Table columns**).
4. Add **Changelog** for runs worth reproducing or comparing; use template below.

### How to run (SLURM — amyloid + acoustic features)

From `ad-detection/`: `sbatch amyloid/run_train.sh` (pass optional fold: `sbatch amyloid/run_train.sh 0`). Logs under `wandb/slurm_amyloid_train_<jobid>.txt`. Scripts exclude `veuc05,veuc01` for GPU arch compatibility; monitor with `squeue` / `sacct`.

---

## Table columns: `Experiment` vs `Script & config`

| Column | Purpose |
| :--- | :--- |
| **`Experiment`** | Short title, **1–7 words** (e.g. “WavLM frozen linear probe”). |
| **`Script & config`** | `train_amyloid.py`, yaml path, CLI, git ref. |

---

## Workflow

1. Run (`amyloid/train_amyloid.py` or your actual entrypoint + config).
2. Capture **validation accuracy each epoch**.
3. Append **Run summary** with **`Experiment`** + **`Script & config`** + metrics.
4. Add **Changelog** when useful.

---

## Metric clarification (human + agent — read this first)

**Primary score for comparing runs:** **`max Val Acc`** = **maximum validation accuracy at any epoch** (not the last epoch).

- Optional: **`Val Acc last`** for the final epoch only.
- **K-fold:** Per-fold max over epochs → **`max Val Acc (mean)`**.
- **`epoch @ max`:** Epoch of best validation accuracy (document indexing convention in `Notes` once).

**Secondary:** **`max F1`** on validation, same max-over-epochs rule.

---

## AGENT: how to auto-fill this log

1. Build the time series of validation accuracy; **`max Val Acc` = max(series)**.
2. **`epoch @ max` = argmax(series)** (per fold if needed).
3. Do **not** use last-epoch validation accuracy as **`max Val Acc`** without verifying it is the global maximum.
4. For checkpoint “best model” callbacks, the logged “best val acc” **is** typically **`max Val Acc`** — confirm in code then note “from EarlyStopping/best_ckpt” in **`Notes`**.

---

## What this file is for

- **Repo:** `ad-detection/`
- **Likely entrypoint:** `amyloid/train_amyloid.py`
- **Related:** `CogniAligned/cogni_amyloid_experiments.md` (multimodal; same metric convention).

---

## How to append (field cheat-sheet)

| Field | Record |
| :--- | :--- |
| `ID` | Job id / run name |
| **`Experiment`** | **1–7 words** |
| `Script & config` | yaml + CLI + branch |
| `Labels / cohort` | Binary amyloid definition |
| **`max Val Acc (mean)`** | Best-over-epochs; mean over folds if CV |
| **`epoch @ max`** | As above |
| **`max F1 (mean)`** | Best-over-epochs if available |
| **`Val Acc last`** | Optional |
| `Notes` | Patient-level split, leakage checks, log path |

---

## Run summary (living table)

| ID | Date | **Experiment** | Script & config | Split | **max Val Acc (mean)** | **epoch @ max** | **max F1 (mean)** | Val Acc last (opt.) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2538679 | 2026-05-05 | SER amyloid acoustic no aug | `sbatch amyloid/run_train.sh` → `train_amyloid.py`; WAB `acoustic_features.csv`; `--augmentation_prob 0` (after job 2538668 CUDA OOM in speed perturb) | 152 train / 57 val segments | **72.41%** (single split) | 12 | **0.7124** | 65.52% | Log: `wandb/slurm_amyloid_train_2538679.txt`; veuc10 |
| 2538668 | 2026-05-05 | SER amyloid acoustic failed | same script with `--augmentation_prob 0.5` | — | — | — | — | — | **Failed:** CUDA OOM in `torchaudio.functional.resample` during speed perturb (rank0); see `wandb/slurm_amyloid_train_2538668.txt` |
| 2553305–2553309 | 2026-05-10 | SER amyloid Mamba 5-fold CV | `sbatch amyloid/launch_cv_mamba.sh` → `run_train_mamba.sh` → `MambaSeqToSeq` n_blocks=4; acoustic AF; 25 epochs; plateau LR; no augmentation | 5-fold CV | **69.58%** | per-fold ep 14, 7, 4, 26, 17 | **0.6313** | — | Per-fold acc: 67.65%, 73.53%, 66.67%, 65.91%, 74.14%. Underperforms SER SelfAttention (72.41%) and CogniAligned bicross (85.74%). Logs: `wandb/slurm_amyloid_mamba_fold*_255330*.txt` |
| 2569451–2569452–2569453–2569454–2569455 | 2026-05-10 | SelfAttention (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **62.38%** | — | **0.5776** | — | Per-fold acc: 52.94%, 64.71%, 71.43%, 65.91%, 56.90%; ~1004.7 batches/min. Logs: `wandb/slurm_*_2569451–2569452–2569453–2569454–2569455.txt` |
| 2569623–2569624–2569625–2569626–2569627 | 2026-05-10 | SelfAttention (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **67.18%** | — | **0.6583** | — | Per-fold acc: 61.76%, 73.53%, 71.43%, 63.64%, 65.52%; ~831.9 batches/min. Logs: `wandb/slurm_*_2569623–2569624–2569625–2569626–2569627.txt` |
| 2569638–2569639–2569640–2569641–2569642 | 2026-05-10 | MultiHeadAttention-2 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **71.00%** | — | **0.7031** | — | Per-fold acc: 61.76%, 88.24%, 66.67%, 65.91%, 72.41%; ~872.0 batches/min. Logs: `wandb/slurm_*_2569638–2569639–2569640–2569641–2569642.txt` |
| 2569653–2569654–2569655–2569656–2569657 | 2026-05-10 | MultiHeadAttention-4 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **70.30%** | — | **0.6773** | — | Per-fold acc: 73.53%, 73.53%, 66.67%, 63.64%, 74.14%; ~853.4 batches/min. Logs: `wandb/slurm_*_2569653–2569654–2569655–2569656–2569657.txt` |
| 2569668–2569669–2569670–2569671–2569672 | 2026-05-10 | CrossAttentionReduced-4 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **69.10%** | — | **0.6597** | — | Per-fold acc: 67.65%, 73.53%, 64.29%, 65.91%, 74.14%; ~999.8 batches/min. Logs: `wandb/slurm_*_2569668–2569669–2569670–2569671–2569672.txt` |
| 2569775–2569776–2569777–2569778–2569779 | 2026-05-10 | Mamba-50ep-lr5e-5 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **67.60%** | — | **0.6225** | — | Per-fold acc: 55.88%, 73.53%, 71.43%, 68.18%, 68.97%; ~722.1 batches/min. Logs: `wandb/slurm_*_2569775–2569776–2569777–2569778–2569779.txt` |
| 2569775–2569776–2569777–2569778–2569779 | 2026-05-10 | Mamba-50ep-lr5e-5 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **67.60%** | — | **0.6225** | — | Per-fold acc: 55.88%, 73.53%, 71.43%, 68.18%, 68.97%; ~722.1 batches/min. Logs: `wandb/slurm_*_2569775–2569776–2569777–2569778–2569779.txt` |
| 2569790–2569791–2569792–2569793–2569794 | 2026-05-10 | Mamba-50ep-lr1e-4 (cached emb) | `adresso_classify/run_train_generic.sh` → `train_adresso.py`; precomputed embeddings; 5-fold CV | 5-fold segment-level CV | **69.04%** | — | **0.6623** | — | Per-fold acc: 64.71%, 73.53%, 73.81%, 65.91%, 67.24%; ~772.4 batches/min. Logs: `wandb/slurm_*_2569790–2569791–2569792–2569793–2569794.txt` |
---

## Changelog

Style: `example_experiment.md`. Append at **bottom**.

### Changelog template (copy per run)

```markdown
### Experiment <ID>: <1–7 word title>

**Status:** Complete ✓ | In progress | Failed  
**Results:** max Val Acc (mean) = … \| epoch @ max = … \| max F1 (mean) = … \| Val Acc last = … (optional)

**Config:**
- Script, yaml, cohort labels:

**What changed:**

**Observations:**

**Artifacts:**
```

### Experiment 2538679: SER amyloid acoustic no aug

**Status:** Complete ✓  
**Results:** max Val Acc = **72.41%** | epoch @ max = **12** | max F1 = **0.7124** | Val Acc last = 65.52%

**Config:** `amyloid/run_train.sh` → pretrained ADReSSo fold1 checkpoint, WAB CSV + acoustic CSV; augmentation disabled in script to avoid OOM on ~11GB GPUs with 2 `torchrun` ranks.

**What changed vs 2538668:** `--augmentation_prob 0` replaces 0.5.

**Artifacts:** `wandb/slurm_amyloid_train_2538679.txt`

### Experiment 2538668: SER amyloid acoustic OOM

**Status:** Failed (exit 1)  
**Results:** n/a — crashed epoch 0 batch 0 during augmentation.

**Artifacts:** `wandb/slurm_amyloid_train_2538668.txt`

### Experiments 2553305–2553309: SER amyloid Mamba 5-fold CV

**Status:** Complete ✓ (all 5 folds)  
**Results:** max Val Acc (mean) = **69.58%** | per-fold epoch @ max: 14, 7, 4, 26, 17 | macro-F1 (mean) = **0.6313**

**Config:** `amyloid/run_train_mamba.sh`; `MambaSeqToSeq` (n_blocks=4, d_state=16, d_conv=4, expand=2); `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; augmentation_prob=0; 25 epochs; AdamW lr=5e-5 wd=0.01; plateau LR; warmup 5 epochs; batch 4. Original folds 0–3 (2553295–2553298) failed on veuc01 (Maxwell GPU); resubmitted with `--exclude=veuc01,veuc05`.

**What changed vs 2538679:** MambaSeqToSeq replaces SelfAttention; full 5-fold CV replaces single split.

**Observations:**
- Per-fold max Val Acc: **67.65%** (f0), **73.53%** (f1), **66.67%** (f2), **65.91%** (f3), **74.14%** (f4)
- Significantly underperforms both SER SelfAttention (72.41%) and CogniAligned bicross (85.74%).
- Fold 2 converged very early (ep 4) — possible premature convergence or overfitting.
- Fold 3 took 26 epochs — longer sequences/harder cases.
- Mamba needs either more epochs, better LR schedule, or higher n_blocks for this task.

**Artifacts:** `wandb/slurm_amyloid_mamba_fold{0–4}_255330{5–9}.txt`

---


### Experiment 2569451–2569452–2569453–2569454–2569455: SelfAttention (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **62.38%** | max macro-F1 (mean) = **0.5776** | **Params:** 611K | **Inference:** 0.81 ms/sample

**Config:** `run_train_generic.sh`; `SelfAttention`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 52.94%, 64.71%, 71.43%, 65.91%, 56.90%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569451–2569452–2569453–2569454–2569455.txt`


### Experiment 2569623–2569624–2569625–2569626–2569627: SelfAttention (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **67.18%** | max macro-F1 (mean) = **0.6583** | **Params:** 611K | **Inference:** 0.81 ms/sample

**Config:** `run_train_generic.sh`; `SelfAttention`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 61.76%, 73.53%, 71.43%, 63.64%, 65.52%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569623–2569624–2569625–2569626–2569627.txt`


### Experiment 2569638–2569639–2569640–2569641–2569642: MultiHeadAttention-2 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **71.00%** | max macro-F1 (mean) = **0.7031** | **Params:** 1.14M | **Inference:** 1.25 ms/sample

**Config:** `run_train_generic.sh`; `MultiHeadAttention-2`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 61.76%, 88.24%, 66.67%, 65.91%, 72.41%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569638–2569639–2569640–2569641–2569642.txt`


### Experiment 2569653–2569654–2569655–2569656–2569657: MultiHeadAttention-4 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **70.30%** | max macro-F1 (mean) = **0.6773** | **Params:** 1.66M | **Inference:** 2.50 ms/sample

**Config:** `run_train_generic.sh`; `MultiHeadAttention-4`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 73.53%, 73.53%, 66.67%, 63.64%, 74.14%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569653–2569654–2569655–2569656–2569657.txt`


### Experiment 2569668–2569669–2569670–2569671–2569672: CrossAttentionReduced-4 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **69.10%** | max macro-F1 (mean) = **0.6597** | **Params:** 1.66M | **Inference:** 0.74 ms/sample

**Config:** `run_train_generic.sh`; `CrossAttentionReduced-4`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 67.65%, 73.53%, 64.29%, 65.91%, 74.14%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569668–2569669–2569670–2569671–2569672.txt`


### Experiment 2569775–2569776–2569777–2569778–2569779: Mamba-50ep-lr5e-5 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **67.60%** | max macro-F1 (mean) = **0.6225** | **Params:** 2.36M | **Inference:** 1.89 ms/sample

**Config:** `run_train_generic.sh`; `Mamba-50ep-lr5e-5`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 55.88%, 73.53%, 71.43%, 68.18%, 68.97%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569775–2569776–2569777–2569778–2569779.txt`


### Experiment 2569775–2569776–2569777–2569778–2569779: Mamba-50ep-lr5e-5 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **67.60%** | max macro-F1 (mean) = **0.6225** | **Params:** 2.36M | **Inference:** 1.89 ms/sample

**Config:** `run_train_generic.sh`; `Mamba-50ep-lr5e-5`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 55.88%, 73.53%, 71.43%, 68.18%, 68.97%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569775–2569776–2569777–2569778–2569779.txt`


### Experiment 2569790–2569791–2569792–2569793–2569794: Mamba-50ep-lr1e-4 (cached emb)

**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **69.04%** | max macro-F1 (mean) = **0.6623** | **Params:** 2.36M | **Inference:** 2.01 ms/sample

**Config:** `run_train_generic.sh`; `Mamba-50ep-lr1e-4`; precomputed Wav2Vec2+ModernBERT embeddings; `LinearAdapter` 256; `AttentionPooling`; freeze_base_models_only; acoustic AF; AdamW lr=?; plateau LR; warmup 5 epochs; batch 4.

**Observations:**
- Per-fold max Val Acc: 64.71%, 73.53%, 73.81%, 65.91%, 67.24%
- 5/5 folds completed.

**Artifacts:** `ad-detection/wandb/slurm_*_2569790–2569791–2569792–2569793–2569794.txt`

## Baseline hints (CogniAligned — not `ad-detection` SER)

Legacy `modules/amyloid/EXPERIMENTS.md` figures (~**78.5%**, ~**83.9%** mean val acc) may **not** be max-over-epochs — recompute before treating as **`max Val Acc`**.

---

## Open points

- [ ] Speech-only vs multimodal in `train_amyloid.py`.
- [ ] Preprocessing: sample rate, chunking, VAD.
