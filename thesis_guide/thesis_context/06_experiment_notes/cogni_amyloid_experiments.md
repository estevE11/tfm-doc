# CogniAligned — Amyloid experiment log

Structured logging **from now on**. Older material is copied from `modules/amyloid/EXPERIMENTS.md`; treat as **possibly outdated**.

---

## Instructions for future agents (read first)

1. Read **Metric clarification** and **AGENT** sections before editing.
2. Each run: one **Run summary** row; headline metric = **`max Val Acc`** over all val epochs (not last epoch only).
3. **`Experiment`** = **1–7 word** title; **`Config / branch`** = yaml / git / CLI — **different columns** (see **Table columns**).
4. Add a **Changelog** `###` block for runs you would want to reproduce or debug; use the template under **Changelog**.

---

## Table columns: `Experiment` vs `Config / branch`

| Column | Purpose |
| :--- | :--- |
| **`Experiment`** | Short title, **1–7 words** (e.g. “Higher LR less weight decay”). |
| **`Config / branch`** | `default.yaml`, branch name, important flags. |

---

## Workflow

1. Run training (`modules/main.py` + `modules/configs/amyloid/default.yaml` or variant).
2. Pull **validation accuracy per epoch** from logs / W&B / CSV.
3. Append one row to **Run summary** with **`Experiment`** + **`Config / branch`** + metrics.
4. Append **Changelog** from template when useful.

### How to run (SLURM — CogniAligned amyloid + acoustic)

From `CogniAligned/`: `sbatch slurm/amyloid_train_af.sh`. Log: `logs/slurm/cogni_amyloid_af_<jobid>.txt`. Uses `modules/configs/amyloid/default.yaml` with acoustic features; `#SBATCH --exclude=veuc05,veuc01` avoids Maxwell-only GPUs. Check `squeue` / `sacct` until `COMPLETED`.

---

## Metric clarification (human + agent — read this first)

**Primary score for comparing runs:** **`max Val Acc`** = **maximum validation accuracy across all epochs** (not the last epoch).

- If you also log final-epoch accuracy, use column **`Val Acc last`** and never confuse it with **`max Val Acc`**.
- **K-fold:** Per fold, take max val acc over epochs; report **`max Val Acc (mean)`** = mean of those fold maxima; put per-fold maxima in `Notes` if useful.
- **`epoch @ max`:** Epoch where val acc first hits the global max (per fold or single run — state convention in `Notes` once).

**Secondary:** **`max F1`** on validation with the same max-over-epochs rule (binary task).

---

## AGENT: how to auto-fill this log

1. Compute **`max Val Acc`** from the validation accuracy time series, **not** from the final row only.
2. Record **`epoch @ max`** via argmax on that series (per fold if CV).
3. For CV, **`max Val Acc (mean)`** = mean of per-fold maxima.
4. Legacy tables without per-epoch data: use **`?`** for `epoch @ max` and note “legacy metric definition unclear” in **`Notes`**.

---

## What this file is for

- **Project:** CogniAligned.
- **Task:** Binary amyloid (positive vs negative).
- **Typical entrypoint:** `modules/main.py` + amyloid config.

---

## How to append (field cheat-sheet)

| Field | Record |
| :--- | :--- |
| `ID` | Job id / run name |
| **`Experiment`** | **1–7 words**; matches Changelog heading |
| `Config / branch` | yaml path, git ref, CLI |
| `Split` | StratifiedGroupKFold etc. |
| **`max Val Acc (mean)`** | Best-over-epochs, then mean over folds if CV |
| **`epoch @ max`** | As defined above |
| **`max F1 (mean)`** | Best-over-epochs val F1, mean over folds if CV |
| **`Val Acc last`** | Optional final-epoch val acc |
| `Notes` | Hypothesis, leakage controls, log path |

---

## Run summary (living table)

| ID | Date | **Experiment** | Config / branch | Split | **max Val Acc (mean)** | **epoch @ max** | **max F1 (mean)** | Val Acc last (opt.) | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2538663 | 2026-05-05 | CogniAlign amyloid plus AF | `slurm/amyloid_train_af.sh` → `modules/amyloid/main.py --config modules/configs/amyloid/default.yaml` | 5-fold CV | **85.74%** | per-fold W&B `best_epoch` 23, 59, 21, 24, 25 | **0.8537** | — | Log: `logs/slurm/cogni_amyloid_af_2538663.txt`; per-fold max val acc: 85.29%, 93.94%, 80.00%, 87.10%, 82.35% |
| 2553274 | 2026-05-09 | Mamba fusion 5-fold CV | `slurm/train_mamba_amyloid.sh` → `modules/configs/amyloid/default_mamba.yaml`; `fusion: mamba`; d_state=16, d_conv=4, expand=2 | 5-fold CV | **82.80%** | per-fold best_epoch 31, 44, 15, 49, 22 | **0.8229** | — | Log: `logs/slurm/cogni_mamba_amyloid_2553274.txt`; per-fold acc: 85.29%, 81.82%, 77.14%, 90.32%, 79.41% |
| 2565645 | 2026-05-10 | Cross Amyloid | see config/cross_amyloid.yaml | `slurm/train_cross_amyloid.sh` | audio+text+AF | 5-fold CV | **83.90%** | 21, 18, 20, 22, 30 | **0.8354** | — | — | — | `slurm/cross_amyloid_2565645.txt`; per-fold acc: 85.29%, 81.82%, 80.00%, 87.10%, 85.29% |
| 2565646 | 2026-05-10 | BiCross Amyloid | see config/bicross_amyloid.yaml | `slurm/train_bicross_amyloid.sh` | audio+text+AF | 5-fold CV | **85.68%** | 23, 66, 19, 24, 46 | **0.8526** | — | — | — | `slurm/bicross_amyloid_2565646.txt`; per-fold acc: 85.29%, 93.94%, 80.00%, 83.87%, 85.29% |
| 2565647 | 2026-05-10 | Mamba Amyloid | see config/mamba_amyloid.yaml | `slurm/train_mamba_amyloid.sh` | audio+text+AF | 5-fold CV | **83.37%** | 22, 47, 28, 44, 37 | **0.8311** | — | — | — | `slurm/mamba_amyloid_2565647.txt`; per-fold acc: 82.35%, 81.82%, 80.00%, 90.32%, 82.35% |
| 2565957 | 2026-05-10 | Mamba Amyloid lr1e-5 | see config/mamba_amyloid_lr1e5.yaml | `slurm/train_mamba_amyloid_lr1e5.sh` | audio+text+AF | 5-fold CV | **84.55%** | 22, 51, 34, 44, 38 | **0.8430** | — | — | — | `slurm/mamba_amyloid_lr1e5_2565957.txt`; per-fold acc: 82.35%, 84.85%, 82.86%, 90.32%, 82.35% |
| 2565958 | 2026-05-10 | Mamba Amyloid lr5e-5 | see config/mamba_amyloid_lr5e5.yaml | `slurm/train_mamba_amyloid_lr5e5.sh` | audio+text+AF | 5-fold CV | **83.37%** | 22, 47, 28, 44, 37 | **0.8311** | — | — | — | `slurm/mamba_amyloid_lr5e5_2565958.txt`; per-fold acc: 82.35%, 81.82%, 80.00%, 90.32%, 82.35% |
| 2565645 | 2026-05-10 | Cross Amyloid | see config/cross_amyloid.yaml | `slurm/train_cross_amyloid.sh` | audio+text+AF | 5-fold CV | **83.90%** | 21, 18, 20, 22, 30 | **0.8354** | — | — | — | `slurm/cross_amyloid_2565645.txt`; per-fold acc: 85.29%, 81.82%, 80.00%, 87.10%, 85.29% |
| 2565646 | 2026-05-10 | BiCross Amyloid | see config/bicross_amyloid.yaml | `slurm/train_bicross_amyloid.sh` | audio+text+AF | 5-fold CV | **85.68%** | 23, 66, 19, 24, 46 | **0.8526** | — | — | — | `slurm/bicross_amyloid_2565646.txt`; per-fold acc: 85.29%, 93.94%, 80.00%, 83.87%, 85.29% |
| 2565647 | 2026-05-10 | Mamba Amyloid | see config/mamba_amyloid.yaml | `slurm/train_mamba_amyloid.sh` | audio+text+AF | 5-fold CV | **83.37%** | 22, 47, 28, 44, 37 | **0.8311** | — | — | — | `slurm/mamba_amyloid_2565647.txt`; per-fold acc: 82.35%, 81.82%, 80.00%, 90.32%, 82.35% |
| 2565645 | 2026-05-10 | Cross Amyloid | see config/cross_amyloid.yaml | `slurm/train_cross_amyloid.sh` | audio+text+AF | 5-fold CV | **83.90%** | 21, 18, 20, 22, 30 | **0.8354** | — | — | — | `slurm/cross_amyloid_2565645.txt`; per-fold acc: 85.29%, 81.82%, 80.00%, 87.10%, 85.29% |
| 2565646 | 2026-05-10 | BiCross Amyloid | see config/bicross_amyloid.yaml | `slurm/train_bicross_amyloid.sh` | audio+text+AF | 5-fold CV | **85.68%** | 23, 66, 19, 24, 46 | **0.8526** | — | — | — | `slurm/bicross_amyloid_2565646.txt`; per-fold acc: 85.29%, 93.94%, 80.00%, 83.87%, 85.29% |
| 2565647 | 2026-05-10 | Mamba Amyloid | see config/mamba_amyloid.yaml | `slurm/train_mamba_amyloid.sh` | audio+text+AF | 5-fold CV | **83.37%** | 22, 47, 28, 44, 37 | **0.8311** | — | — | — | `slurm/mamba_amyloid_2565647.txt`; per-fold acc: 82.35%, 81.82%, 80.00%, 90.32%, 82.35% |
| 2569847 | 2026-05-10 | Mamba Amyloid lr1e-5 | see config/mamba_amyloid_lr1e5.yaml | `slurm/train_mamba_amyloid_lr1e5.sh` | audio+text+AF | 5-fold CV | **80.85%** | 30, 40, 59, 38, 57 | **0.8055** | — | — | — | `slurm/mamba_amyloid_lr1e5_2569847.txt`; per-fold acc: 82.35%, 75.76%, 82.86%, 83.87%, 79.41% |
| 2569848 | 2026-05-10 | Mamba Amyloid lr5e-5 | see config/mamba_amyloid_lr5e5.yaml | `slurm/train_mamba_amyloid_lr5e5.sh` | audio+text+AF | 5-fold CV | **83.41%** | 15, 48, 21, 23, 16 | **0.8299** | — | — | — | `slurm/mamba_amyloid_lr5e5_2569848.txt`; per-fold acc: 82.35%, 87.88%, 80.00%, 90.32%, 76.47% |
---

## Changelog

Style: `example_experiment.md` (repo root). Newest entries: append at **bottom** unless you standardise otherwise.

### Changelog template (copy per run)

```markdown
### Experiment <ID>: <1–7 word title>

**Status:** Complete ✓ | In progress | Failed  
**Results:** max Val Acc (mean) = … \| epoch @ max = … \| max F1 (mean) = … \| Val Acc last = … (optional)

**Config:**
- File / branch / commit:
- Split, seed:

**What changed:**

**Observations:**

**Artifacts:** logs, W&B, checkpoints
```

### Experiment 2538663: CogniAlign amyloid plus AF
25386632538663
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **85.74%** | epoch @ max = per-fold 23, 59, 21, 24, 25 (W&B `best_epoch`) | max F1 (mean) = **0.8537**

**Config:** `modules/configs/amyloid/default.yaml`, acoustic features on, 5-fold CV.

**Artifacts:** `/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm/cogni_amyloid_af_2538663.txt`

### Experiment 2553274: Mamba fusion 5-fold CV
25532742553274
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **82.80%** | epoch @ max = per-fold 31, 44, 15, 49, 22 | max F1 (mean) = **0.8229** | early stopping per fold at 51, 64, 35, 69, 42

**Config:** `modules/configs/amyloid/default_mamba.yaml`; `fusion: mamba`; `mamba_d_state: 16`, `mamba_d_conv: 4`, `mamba_expand: 2`; n_layers=6; hidden_size=768; lr=2e-5; cosine LR with 20-epoch warmup.

**What changed:** Swapped bicross encoder for pure-PyTorch `MambaFusionEncoder` (same as ADReSSo; audio+text concatenated, acoustic token fused via self-attention).

**Observations:**
- Per-fold max Val Acc: 85.29%, 81.82%, 77.14%, 90.32%, 79.41%
- Mean **82.80%** vs bicross **85.74%** — Mamba slightly underperforms bicross on amyloid; the longer fold 1 convergence (ep 44/64) and lower fold 2 (77%) suggest the SSM needs more tuning on this smaller dataset.

**Artifacts:** `logs/slurm/cogni_mamba_amyloid_2553274.txt` · W&B project `CogniAligned-Amyloid`

---

## Provisional baseline (legacy — likely not max-over-epochs)

| Legacy ref | Job | Description | Val Acc (legacy) | F1 (legacy) | Comment |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Exp 1 | — | Baseline binary | *unclear* | — | Setup |
| Exp 3 | **2398093** | + acoustic features | **~78.5%** | **~0.78** | Re-log with max metric when possible |
| **Exp 4** | **2398098** | LR / WD tune | **~83.9%** | **~0.834** | **Strong interim baseline** |

Detail: `modules/amyloid/EXPERIMENTS.md`.

---


### Experiment 2565645: Cross Amyloid
25656452565645
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.90%** | epoch @ max = per-fold 21, 18, 20, 22, 30 | max macro-F1 (mean) = **0.8354** | **Params:** 7.69M | **Inference:** 0.26 ms/splnfig:** `slurm/train_cross_amyloid.sh`; fusion=cross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 85.29%, 81.82%, 80.00%, 87.10%, 85.29%

**Artifacts:** `CogniAligned/logs/slurm/cross_amyloid_2565645.txt`


### Experiment 2565646: BiCross Amyloid
25656462565646
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **85.68%** | epoch @ max = per-fold 23, 66, 19, 24, 46 | max macro-F1 (mean) = **0.8526** | **Params:** 14.78M | **Inference:** 1.36 ms/splnfig:** `slurm/train_bicross_amyloid.sh`; fusion=bicross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 85.29%, 93.94%, 80.00%, 83.87%, 85.29%

**Artifacts:** `CogniAligned/logs/slurm/bicross_amyloid_2565646.txt`


### Experiment 2565647: Mamba Amyloid
25656472565647
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.37%** | epoch @ max = per-fold 22, 47, 28, 44, 37 | max macro-F1 (mean) = **0.8311** | **Params:** 4.37M | **Inference:** 0.34 ms/splnfig:** `slurm/train_mamba_amyloid.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 82.35%, 81.82%, 80.00%, 90.32%, 82.35%

**Artifacts:** `CogniAligned/logs/slurm/mamba_amyloid_2565647.txt`


### Experiment 2565957: Mamba Amyloid lr1e-5
25659572565957
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **84.55%** | epoch @ max = per-fold 22, 51, 34, 44, 38 | max macro-F1 (mean) = **0.8430** | **Params:** 4.37M | **Inference:** 0.37 ms/splnfig:** `slurm/train_mamba_amyloid_lr1e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 82.35%, 84.85%, 82.86%, 90.32%, 82.35%

**Artifacts:** `CogniAligned/logs/slurm/mamba_amyloid_lr1e5_2565957.txt`


### Experiment 2565958: Mamba Amyloid lr5e-5
25659582565958
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.37%** | epoch @ max = per-fold 22, 47, 28, 44, 37 | max macro-F1 (mean) = **0.8311** | **Params:** 4.37M | **Inference:** 0.33 ms/splnfig:** `slurm/train_mamba_amyloid_lr5e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 82.35%, 81.82%, 80.00%, 90.32%, 82.35%

**Artifacts:** `CogniAligned/logs/slurm/mamba_amyloid_lr5e5_2565958.txt`


### Experiment 2565645: Cross Amyloid
25656452565645
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.90%** | epoch @ max = per-fold 21, 18, 20, 22, 30 | max macro-F1 (mean) = **0.8354** | **Params:** 7.69M | **Inference:** 0.26 ms/splnfig:** `slurm/train_cross_amyloid.sh`; fusion=cross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 85.29%, 81.82%, 80.00%, 87.10%, 85.29%

**Artifacts:** `CogniAligned/logs/slurm/cross_amyloid_2565645.txt`


### Experiment 2565646: BiCross Amyloid
25656462565646
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **85.68%** | epoch @ max = per-fold 23, 66, 19, 24, 46 | max macro-F1 (mean) = **0.8526** | **Params:** 14.78M | **Inference:** 1.36 ms/splnfig:** `slurm/train_bicross_amyloid.sh`; fusion=bicross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 85.29%, 93.94%, 80.00%, 83.87%, 85.29%

**Artifacts:** `CogniAligned/logs/slurm/bicross_amyloid_2565646.txt`


### Experiment 2565647: Mamba Amyloid
25656472565647
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.37%** | epoch @ max = per-fold 22, 47, 28, 44, 37 | max macro-F1 (mean) = **0.8311** | **Params:** 4.37M | **Inference:** 0.34 ms/splnfig:** `slurm/train_mamba_amyloid.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 82.35%, 81.82%, 80.00%, 90.32%, 82.35%

**Artifacts:** `CogniAligned/logs/slurm/mamba_amyloid_2565647.txt`


### Experiment 2565645: Cross Amyloid
25656452565645
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.90%** | epoch @ max = per-fold 21, 18, 20, 22, 30 | max macro-F1 (mean) = **0.8354** | **Params:** 7.69M | **Inference:** 0.26 ms/splnfig:** `slurm/train_cross_amyloid.sh`; fusion=cross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 85.29%, 81.82%, 80.00%, 87.10%, 85.29%

**Artifacts:** `CogniAligned/logs/slurm/cross_amyloid_2565645.txt`


### Experiment 2565646: BiCross Amyloid
25656462565646
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **85.68%** | epoch @ max = per-fold 23, 66, 19, 24, 46 | max macro-F1 (mean) = **0.8526** | **Params:** 14.78M | **Inference:** 1.36 ms/splnfig:** `slurm/train_bicross_amyloid.sh`; fusion=bicross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 85.29%, 93.94%, 80.00%, 83.87%, 85.29%

**Artifacts:** `CogniAligned/logs/slurm/bicross_amyloid_2565646.txt`


### Experiment 2565647: Mamba Amyloid
25656472565647
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.37%** | epoch @ max = per-fold 22, 47, 28, 44, 37 | max macro-F1 (mean) = **0.8311** | **Params:** 4.37M | **Inference:** 0.34 ms/splnfig:** `slurm/train_mamba_amyloid.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 82.35%, 81.82%, 80.00%, 90.32%, 82.35%

**Artifacts:** `CogniAligned/logs/slurm/mamba_amyloid_2565647.txt`


### Experiment 2569847: Mamba Amyloid lr1e-5
25698472569847
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **80.85%** | epoch @ max = per-fold 30, 40, 59, 38, 57 | max macro-F1 (mean) = **0.8055** | **Params:** 4.37M | **Inference:** 0.37 ms/splnfig:** `slurm/train_mamba_amyloid_lr1e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 82.35%, 75.76%, 82.86%, 83.87%, 79.41%

**Artifacts:** `CogniAligned/logs/slurm/mamba_amyloid_lr1e5_2569847.txt`


### Experiment 2569848: Mamba Amyloid lr5e-5
25698482569848
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **83.41%** | epoch @ max = per-fold 15, 48, 21, 23, 16 | max macro-F1 (mean) = **0.8299** | **Params:** 4.37M | **Inference:** 0.33 ms/splnfig:** `slurm/train_mamba_amyloid_lr5e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 82.35%, 87.88%, 80.00%, 90.32%, 76.47%

**Artifacts:** `CogniAligned/logs/slurm/mamba_amyloid_lr5e5_2569848.txt`

## Archive — narrative (legacy)

- Exp 3: Acoustic features; consistent moderate performance.
- Exp 4: Reduced underfitting via LR/WD; gains across folds in legacy notes.
