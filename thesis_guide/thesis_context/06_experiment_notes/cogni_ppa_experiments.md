# CogniAligned — PPA experiment log

Structured logging **from now on**. Older material below is copied from `modules/ppa/EXPERIMENTS.md` for continuity; treat job IDs and numbers as **possibly outdated**.

---

## Instructions for future agents (read first)

1. Read **Metric clarification** and **AGENT: how to auto-fill** before editing anything.
2. For each finished training run: add **one row** to **Run summary**; use **`max Val Acc`** (best over **all** validation epochs), not the last epoch unless you also log **`Val Acc last`**.
3. **`Experiment`** and **`Config`** are **different** (see **Table columns** below). Never put a long path in **`Experiment`**; keep the title to **1–7 words**.
4. For non-trivial runs, add a **`###` block under Changelog** (copy the template there). Link SLURM log, W&B, or checkpoint paths in that block.
5. Do not delete legacy sections without explicit user instruction; add new content at the top of the living table / bottom of Changelog.

### How to run (SLURM — CogniAligned PPA + acoustic)

From `CogniAligned/`: `sbatch slurm/ppa_train_stratified.sh`. SLURM stdout: `logs/slurm/ppa_strat_<jobid>.txt`. Same `#SBATCH --exclude=veuc05,veuc01` pattern as other Cogni jobs; `mkdir -p logs/slurm` is handled in the script.

---

## Table columns: `Experiment` vs `Config / branch`

Same idea as P3’s table in `example_experiment.md` (`| # | Experiment | …`):

| Column | Purpose |
| :--- | :--- |
| **`Experiment`** | **Short human title**, **1–7 words** (e.g. “Lower dropout acoustic on”). This is **not** the config path. |
| **`Config / branch`** | **Technical pointer**: YAML path, git branch or commit, main CLI flags, or notebook/cell reference. |

---

## Workflow

1. Run training (e.g. SLURM → `modules/main.py` + config under `modules/configs/ppa/`).
2. From the training log / W&B / CSV, read **validation accuracy per epoch** (not only the last line).
3. Append **one row** to **Run summary** (fill **`Experiment`** + **`Config / branch`** separately).
4. Add a **Changelog** entry for runs worth documenting (config deltas, failures, interesting fold behaviour). Use the template under **Changelog**.

---

## Metric clarification (human + agent — read this first)

**Primary score for comparing runs:** **max Val Acc** = the **maximum validation accuracy achieved in any epoch** during that training run.

- **Do not** use validation accuracy from the **last epoch** unless it happens to equal the max; if you log both, label the last-epoch column **`Val Acc last`** explicitly.
- **K-fold / multiple folds:** For each fold, compute the max validation accuracy over epochs for **that** fold. Then report:
  - **`max Val Acc (mean)`** = mean of those per-fold maxima (headline aggregate), and optionally **min / max** across folds in `Notes`.
- **Epoch column:** **`epoch @ max`** = the epoch index (1-based or 0-based — **state which** in `Notes` once per file) where validation accuracy first hit **`max Val Acc`** (or use “best checkpoint” epoch from your logger).

**Secondary metrics** (PPA is imbalanced): log **`max macro-F1`** with the **same rule** (best value over epochs on the validation split) when available. If your logger only stores F1 at the same step as accuracy, note that in `Notes`.

---

## AGENT: how to auto-fill this log

When writing or parsing experiment results for this project:

1. **Never** assume the reported validation accuracy is the **final** epoch unless the source explicitly says so.
2. Extract **`max Val Acc`** = `max(validation_accuracy[e] for e in epochs)` (or equivalent from W&B / CSV).
3. Extract **`epoch @ max`** from the argmax of that series (per fold if applicable).
4. If the experiment is **cross-validation**, fill **`max Val Acc (mean)`** as the mean of per-fold maxima; put per-fold maxima in `Notes` or under **Changelog**.
5. If legacy rows only have “mean val acc” without max-vs-last documentation, put **`?`** in **`epoch @ max`** and add “legacy / metric unclear” in **`Notes`**.

---

## What this file is for

- **Project:** CogniAligned (multimodal: audio + text [+ acoustic features]).
- **Task:** PPA subtype classification (e.g. `lvPPA` / `nfPPA` / `svPPA`).
- **Typical entrypoint:** `modules/main.py` with a PPA config under `modules/configs/ppa/`.

---

## How to append a new run (field cheat-sheet)

| Field | What to record |
| :--- | :--- |
| `ID` | Run name or SLURM job id |
| **`Experiment`** | **1–7 words**; same phrase as Changelog title |
| `Config / branch` | Path to yaml, git ref, key CLI overrides |
| `Split` | e.g. StratifiedGroupKFold, 5 folds |
| **`max Val Acc (mean)`** | Mean over folds of (max val acc per fold); single split = that max |
| **`epoch @ max`** | Epoch at best val acc (per-fold list or mean — state in Notes) |
| **`max macro-F1 (mean)`** | Same “best over epochs” rule on val, then mean over folds if applicable |
| **`Val Acc last`** | Optional: val acc at final epoch (for overfitting diagnostics) |
| `Notes` | What changed vs previous row; link to log / W&B |

---

## Run summary (living table)

| ID | Date | **Experiment** | Config / branch | Split | **max Val Acc (mean)** | **epoch @ max** | **max macro-F1 (mean)** | Val Acc last (opt.) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2538670 | 2026-05-05 | Stratified bicross plus AF | `slurm/ppa_train_stratified.sh` → `modules/ppa/main.py --config modules/configs/ppa/balanced_noweights.yaml`; WAB `acoustic_features.csv` | 5-fold stratified | **73.89%** | per-fold W&B `best_epoch` 18, 42, 15, 60, 43 | **0.6938** | — | SLURM: `logs/slurm/ppa_strat_2538670.txt`; per-fold max val acc: 79.49%, 76.92%, 63.42%, 84.62%, 65.00% |
| 2553275 | 2026-05-09 | Mamba fusion 5-fold CV | `slurm/train_mamba_ppa.sh` → `modules/configs/ppa/mamba.yaml`; `fusion: mamba`; d_state=16, d_conv=4, expand=2 | 5-fold stratified | **75.35%** | per-fold best_epoch 57, 45, 70, 44, 21 | **0.6905** | — | SLURM: `logs/slurm/cogni_mamba_ppa_2553275.txt`; per-fold acc: 84.61%, 76.92%, 68.29%, 76.92%, 70.00% |
| 2565648 | 2026-05-10 | Cross PPA | see config/cross_ppa.yaml | `slurm/train_cross_ppa.sh` | audio+text+AF | 5-fold CV | **70.80%** | 70, 9, 17, 9, 8 | **0.6399** | — | — | — | `slurm/cross_ppa_2565648.txt`; per-fold acc: 87.18%, 64.10%, 65.85%, 74.36%, 62.50% |
| 2565649 | 2026-05-10 | BiCross PPA | see config/bicross_ppa.yaml | `slurm/train_bicross_ppa.sh` | audio+text+AF | 5-fold CV | **71.77%** | 33, 27, 10, 12, 36 | **0.6618** | — | — | — | `slurm/bicross_ppa_2565649.txt`; per-fold acc: 76.92%, 71.80%, 68.29%, 74.36%, 67.50% |
| 2565650 | 2026-05-10 | Mamba PPA | see config/mamba_ppa.yaml | `slurm/train_mamba_ppa.sh` | audio+text+AF | 5-fold CV | **72.79%** | 32, 89, 31, 15, 47 | **0.6287** | — | — | — | `slurm/mamba_ppa_2565650.txt`; per-fold acc: 76.92%, 71.80%, 65.85%, 74.36%, 75.00% |
| 2565959 | 2026-05-10 | Mamba PPA lr1e-5 | see config/mamba_ppa_lr1e5.yaml | `slurm/train_mamba_ppa_lr1e5.sh` | audio+text+AF | 5-fold CV | **74.21%** | 20, 29, 78, 7, 24 | **0.6341** | — | — | — | `slurm/mamba_ppa_lr1e5_2565959.txt`; per-fold acc: 74.36%, 74.36%, 80.49%, 74.36%, 67.50% |
| 2565960 | 2026-05-10 | Mamba PPA lr5e-5 | see config/mamba_ppa_lr5e5.yaml | `slurm/train_mamba_ppa_lr5e5.sh` | audio+text+AF | 5-fold CV | **74.21%** | 20, 29, 78, 7, 24 | **0.6341** | — | — | — | `slurm/mamba_ppa_lr5e5_2565960.txt`; per-fold acc: 74.36%, 74.36%, 80.49%, 74.36%, 67.50% |
| 2565648 | 2026-05-10 | Cross PPA | see config/cross_ppa.yaml | `slurm/train_cross_ppa.sh` | audio+text+AF | 5-fold CV | **70.80%** | 70, 9, 17, 9, 8 | **0.6399** | — | — | — | `slurm/cross_ppa_2565648.txt`; per-fold acc: 87.18%, 64.10%, 65.85%, 74.36%, 62.50% |
| 2565649 | 2026-05-10 | BiCross PPA | see config/bicross_ppa.yaml | `slurm/train_bicross_ppa.sh` | audio+text+AF | 5-fold CV | **71.77%** | 33, 27, 10, 12, 36 | **0.6618** | — | — | — | `slurm/bicross_ppa_2565649.txt`; per-fold acc: 76.92%, 71.80%, 68.29%, 74.36%, 67.50% |
| 2565650 | 2026-05-10 | Mamba PPA | see config/mamba_ppa.yaml | `slurm/train_mamba_ppa.sh` | audio+text+AF | 5-fold CV | **72.79%** | 32, 89, 31, 15, 47 | **0.6287** | — | — | — | `slurm/mamba_ppa_2565650.txt`; per-fold acc: 76.92%, 71.80%, 65.85%, 74.36%, 75.00% |
| 2565648 | 2026-05-10 | Cross PPA | see config/cross_ppa.yaml | `slurm/train_cross_ppa.sh` | audio+text+AF | 5-fold CV | **70.80%** | 70, 9, 17, 9, 8 | **0.6399** | — | — | — | `slurm/cross_ppa_2565648.txt`; per-fold acc: 87.18%, 64.10%, 65.85%, 74.36%, 62.50% |
| 2565649 | 2026-05-10 | BiCross PPA | see config/bicross_ppa.yaml | `slurm/train_bicross_ppa.sh` | audio+text+AF | 5-fold CV | **71.77%** | 33, 27, 10, 12, 36 | **0.6618** | — | — | — | `slurm/bicross_ppa_2565649.txt`; per-fold acc: 76.92%, 71.80%, 68.29%, 74.36%, 67.50% |
| 2565650 | 2026-05-10 | Mamba PPA | see config/mamba_ppa.yaml | `slurm/train_mamba_ppa.sh` | audio+text+AF | 5-fold CV | **72.79%** | 32, 89, 31, 15, 47 | **0.6287** | — | — | — | `slurm/mamba_ppa_2565650.txt`; per-fold acc: 76.92%, 71.80%, 65.85%, 74.36%, 75.00% |
| 2569849 | 2026-05-10 | Mamba PPA lr1e-5 | see config/mamba_ppa_lr1e5.yaml | `slurm/train_mamba_ppa_lr1e5.sh` | audio+text+AF | 5-fold CV | **72.79%** | 32, 89, 31, 15, 47 | **0.6287** | — | — | — | `slurm/mamba_ppa_lr1e5_2569849.txt`; per-fold acc: 76.92%, 71.80%, 65.85%, 74.36%, 75.00% |
| 2569850 | 2026-05-10 | Mamba PPA lr5e-5 | see config/mamba_ppa_lr5e5.yaml | `slurm/train_mamba_ppa_lr5e5.sh` | audio+text+AF | 5-fold CV | **77.71%** | 29, 28, 45, 29, 23 | **0.6633** | — | — | — | `slurm/mamba_ppa_lr5e5_2569850.txt`; per-fold acc: 74.36%, 74.36%, 85.37%, 79.49%, 75.00% |
---

## Changelog

Long-form entries per run (style: `example_experiment.md` in repo root). **Order:** add newer `###` sections **below** older ones (or **above**—pick one convention for this file and stick to it; default: **newest at bottom**).

### Changelog template (copy per run)

```markdown
### Experiment <ID>: <same title as Experiment column — 1–7 words>

**Status:** Complete ✓ | In progress | Failed  
**Results:** max Val Acc (mean) = … \| epoch @ max = … \| max macro-F1 (mean) = … \| Val Acc last = … (optional)

**Config:**
- Config file / branch / commit:
- Split, seed, device:

**What changed:** (vs previous baseline run)

**Observations:** (fold spread, collapse, loss curve)

**Artifacts:** SLURM log path, W&B run, checkpoints
```

### Experiment 2538670: Stratified bicross plus AF
25386702538670
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **73.89%** | epoch @ max = per-fold 18, 42, 15, 60, 43 | max macro-F1 (mean) = **0.6938**

**Config:** `balanced_noweights.yaml`, bicross encoder, weighted sampler; 55-dim acoustic features from `WAB_samples/acoustic_features.csv`.

**Artifacts:** `/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm/ppa_strat_2538670.txt`

### Experiment 2553275: Mamba fusion 5-fold CV
25532752553275
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **75.35%** | epoch @ max = per-fold 57, 45, 70, 44, 21 | max macro-F1 (mean) = **0.6905** | early stopping per fold at 82, 70, 95, 69, 46

**Config:** `modules/configs/ppa/mamba.yaml`; `fusion: mamba`; `mamba_d_state: 16`, `mamba_d_conv: 4`, `mamba_expand: 2`; n_layers=6; hidden_size=768; lr=2e-5; cosine LR with 20-epoch warmup.

**What changed:** Swapped bicross encoder for pure-PyTorch `MambaFusionEncoder`; acoustic feature token fused via self-attention.

**Observations:**
- Per-fold max Val Acc: 84.61%, 76.92%, 68.29%, 76.92%, 70.00%
- Mean **75.35%** vs bicross **73.89%** — Mamba edges out bicross by +1.5% on PPA; macro-F1 comparable (0.6905 vs 0.6938). PPA is a harder 3-class task with high fold variance; fold 2 remains the weak point.

**Artifacts:** `logs/slurm/cogni_mamba_ppa_2553275.txt` · W&B project `CogniAligned-PPA`

---

## Provisional baseline (legacy — metrics may be last-epoch, not max)

From archived `modules/ppa/EXPERIMENTS.md` (not maintained here). **Re-log with `max Val Acc` when you have per-epoch curves.**

| Legacy ref | Description | Val Acc (legacy) | macro-F1 (legacy) | Comment |
| :--- | :--- | :--- | :--- | :--- |
| Exp 1 | “Baseline (fast)” | ~47–68% across folds | ~0.21–0.56 | Many folds collapsed to majority class |
| Exp 8 | + 55 acoustic features | ~70.3% | ~0.643 | Mixed across folds |
| **Exp 9** | Reduced underfitting, Job **2398099** | **~71.3%** | **~0.647** | **Interim baseline** until superseded |

---


### Experiment 2565648: Cross PPA
25656482565648
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **70.80%** | epoch @ max = per-fold 70, 9, 17, 9, 8 | max macro-F1 (mean) = **0.6399** | **Params:** 7.29M | **Inference:** 0.20 ms/splnfig:** `slurm/train_cross_ppa.sh`; fusion=cross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 87.18%, 64.10%, 65.85%, 74.36%, 62.50%

**Artifacts:** `CogniAligned/logs/slurm/cross_ppa_2565648.txt`


### Experiment 2565649: BiCross PPA
25656492565649
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **71.77%** | epoch @ max = per-fold 33, 27, 10, 12, 36 | max macro-F1 (mean) = **0.6618** | **Params:** 14.37M | **Inference:** 0.62 ms/splnfig:** `slurm/train_bicross_ppa.sh`; fusion=bicross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 76.92%, 71.80%, 68.29%, 74.36%, 67.50%

**Artifacts:** `CogniAligned/logs/slurm/bicross_ppa_2565649.txt`


### Experiment 2565650: Mamba PPA
25656502565650
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **72.79%** | epoch @ max = per-fold 32, 89, 31, 15, 47 | max macro-F1 (mean) = **0.6287** | **Params:** 4.02M | **Inference:** 0.31 ms/splnfig:** `slurm/train_mamba_ppa.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 76.92%, 71.80%, 65.85%, 74.36%, 75.00%

**Artifacts:** `CogniAligned/logs/slurm/mamba_ppa_2565650.txt`


### Experiment 2565959: Mamba PPA lr1e-5
25659592565959
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **74.21%** | epoch @ max = per-fold 20, 29, 78, 7, 24 | max macro-F1 (mean) = **0.6341** | **Params:** 4.02M | **Inference:** 0.44 ms/splnfig:** `slurm/train_mamba_ppa_lr1e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 74.36%, 74.36%, 80.49%, 74.36%, 67.50%

**Artifacts:** `CogniAligned/logs/slurm/mamba_ppa_lr1e5_2565959.txt`


### Experiment 2565960: Mamba PPA lr5e-5
25659602565960
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **74.21%** | epoch @ max = per-fold 20, 29, 78, 7, 24 | max macro-F1 (mean) = **0.6341** | **Params:** 4.02M | **Inference:** 0.32 ms/splnfig:** `slurm/train_mamba_ppa_lr5e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 74.36%, 74.36%, 80.49%, 74.36%, 67.50%

**Artifacts:** `CogniAligned/logs/slurm/mamba_ppa_lr5e5_2565960.txt`


### Experiment 2565648: Cross PPA
25656482565648
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **70.80%** | epoch @ max = per-fold 70, 9, 17, 9, 8 | max macro-F1 (mean) = **0.6399** | **Params:** 7.29M | **Inference:** 0.20 ms/splnfig:** `slurm/train_cross_ppa.sh`; fusion=cross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 87.18%, 64.10%, 65.85%, 74.36%, 62.50%

**Artifacts:** `CogniAligned/logs/slurm/cross_ppa_2565648.txt`


### Experiment 2565649: BiCross PPA
25656492565649
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **71.77%** | epoch @ max = per-fold 33, 27, 10, 12, 36 | max macro-F1 (mean) = **0.6618** | **Params:** 14.37M | **Inference:** 0.62 ms/splnfig:** `slurm/train_bicross_ppa.sh`; fusion=bicross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 76.92%, 71.80%, 68.29%, 74.36%, 67.50%

**Artifacts:** `CogniAligned/logs/slurm/bicross_ppa_2565649.txt`


### Experiment 2565650: Mamba PPA
25656502565650
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **72.79%** | epoch @ max = per-fold 32, 89, 31, 15, 47 | max macro-F1 (mean) = **0.6287** | **Params:** 4.02M | **Inference:** 0.31 ms/splnfig:** `slurm/train_mamba_ppa.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 76.92%, 71.80%, 65.85%, 74.36%, 75.00%

**Artifacts:** `CogniAligned/logs/slurm/mamba_ppa_2565650.txt`


### Experiment 2565648: Cross PPA
25656482565648
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **70.80%** | epoch @ max = per-fold 70, 9, 17, 9, 8 | max macro-F1 (mean) = **0.6399** | **Params:** 7.29M | **Inference:** 0.20 ms/splnfig:** `slurm/train_cross_ppa.sh`; fusion=cross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 87.18%, 64.10%, 65.85%, 74.36%, 62.50%

**Artifacts:** `CogniAligned/logs/slurm/cross_ppa_2565648.txt`


### Experiment 2565649: BiCross PPA
25656492565649
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **71.77%** | epoch @ max = per-fold 33, 27, 10, 12, 36 | max macro-F1 (mean) = **0.6618** | **Params:** 14.37M | **Inference:** 0.62 ms/splnfig:** `slurm/train_bicross_ppa.sh`; fusion=bicross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 76.92%, 71.80%, 68.29%, 74.36%, 67.50%

**Artifacts:** `CogniAligned/logs/slurm/bicross_ppa_2565649.txt`


### Experiment 2565650: Mamba PPA
25656502565650
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **72.79%** | epoch @ max = per-fold 32, 89, 31, 15, 47 | max macro-F1 (mean) = **0.6287** | **Params:** 4.02M | **Inference:** 0.31 ms/splnfig:** `slurm/train_mamba_ppa.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 76.92%, 71.80%, 65.85%, 74.36%, 75.00%

**Artifacts:** `CogniAligned/logs/slurm/mamba_ppa_2565650.txt`


### Experiment 2569849: Mamba PPA lr1e-5
25698492569849
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **72.79%** | epoch @ max = per-fold 32, 89, 31, 15, 47 | max macro-F1 (mean) = **0.6287** | **Params:** 4.02M | **Inference:** 0.44 ms/splnfig:** `slurm/train_mamba_ppa_lr1e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 76.92%, 71.80%, 65.85%, 74.36%, 75.00%

**Artifacts:** `CogniAligned/logs/slurm/mamba_ppa_lr1e5_2569849.txt`


### Experiment 2569850: Mamba PPA lr5e-5
25698502569850
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **77.71%** | epoch @ max = per-fold 29, 28, 45, 29, 23 | max macro-F1 (mean) = **0.6633** | **Params:** 4.02M | **Inference:** 0.32 ms/splnfig:** `slurm/train_mamba_ppa_lr5e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 74.36%, 74.36%, 85.37%, 79.49%, 75.00%

**Artifacts:** `CogniAligned/logs/slurm/mamba_ppa_lr5e5_2569850.txt`

## Archive — narrative (legacy, may be stale)

1. **Early folds:** Strong fold-to-fold variance; some folds predicted almost only majority class.
2. **Balanced sampling / no loss weights:** Training dynamics vs val fold distribution issues.
3. **Stratified group splits:** Stabilise class counts per fold.
4. **Acoustic features:** 55-dim vectors; Exp 8–9 in legacy table.

Full detail: `modules/ppa/EXPERIMENTS.md`.
