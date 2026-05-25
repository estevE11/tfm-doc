# CogniAligned — ADReSSo experiment log

Structured logging **from now on**. Short legacy rows come from `CogniAligned/EXPERIMENTS.md`; verify before citing.

---

## Instructions for future agents (read first)

1. Read **Metric clarification** and **AGENT** before editing.
2. One **Run summary** row per training run; **`max Val Acc`** = max val accuracy **across epochs**, not last epoch by default.
3. **`Experiment`** (1–7 words) ≠ **`Config / branch`** or model path — see **Table columns**.
4. Use **Changelog** for anything you’d need to reproduce or defend in a meeting; copy the template below.

### How to run (SLURM — CogniAligned ADReSSo + acoustic)

From `CogniAligned/`: `sbatch slurm/train_af.sh`. Log: `logs/slurm/cogni_adresso_af_<jobid>.txt`. Config: `modules/configs/default.yaml` with `use_acoustic_features: True`; ADReSSo train `acoustic_features.csv`. Scripts use `#SBATCH --exclude=veuc05,veuc01` for GPU compatibility.

**Full train + test-dist as validation (bicross):** `sbatch slurm/train_maximize_test_accuracy.sh` (optional: pass config YAML as first arg). Default config `modules/configs/default_bicross_fulltrain_testval.yaml` (`train.validate_on_test_dist: true`, `cross_validation: false`). Same pipeline with `fusion: cross`: `default_cross_fulltrain_testval.yaml`. `fusion: ''` (`ElementWiseFusionEncoder`): `default_elementwise_fulltrain_testval.yaml` — **currently crashes** on first batch (list vs tensor in `TransformerEncoder`; see job **2538766** in Run summary). Trains on all train embeddings; each epoch logs **test/** accuracy in W&B and early-stops on **max test accuracy** (same as `evaluation()` return value). Put `./task1.csv` in repo root for labels. Use responsibly — this optimizes on the test set.

**Test-dist (official) inference:** not run by `main.py`. After checkpoints exist under `logs/adresso_distil_wav2vec2_P_cross_mean/`, run `modules/test.py`’s `test()` with the same config as training. Job **2538671** used **`fusion: cross`** (checkpoint dir name), even if `default.yaml` lists `bicross` — override fusion before `save_config` / `test()`, e.g.:

```bash
cd /home/usuaris/veussd/roger.esteve.sanchez/CogniAligned && source .venv/bin/activate
export PYTHONPATH=modules:$PYTHONPATH WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
python -u -c "import os,sys; os.chdir('/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned'); sys.path.insert(0,'modules'); from utils import get_config, save_config; from test import test; c=get_config('modules/configs/default.yaml'); c.model.fusion='cross'; save_config(c); c.path_name='adresso_'+c.path_name; test(c)"
```

---

## Table columns: `Experiment` vs technical columns

| Column | Purpose |
| :--- | :--- |
| **`Experiment`** | **1–7 word** label for the run (e.g. “Late fusion acoustic fifty-five”). |
| **`Model / fusion`** | Architecture / fusion style (can be slightly longer). |
| **`Config / branch`** | Yaml, git ref, SLURM script, or config overrides. |

---

## Workflow

1. Run training (`modules/main.py` + ADReSSo-oriented config / SLURM).
2. Extract **validation accuracy per epoch** (and macro-F1 per epoch if logged).
3. Append **Run summary** row: **`Experiment`**, **`Model / fusion`**, metrics.
4. Add **Changelog** from template when useful.

---

## Metric clarification (human + agent — read this first)

**Primary score for comparing runs:** **`max Val Acc`** = **maximum validation accuracy across all epochs**.

- **Not** the validation accuracy at the **last** epoch unless you note that they coincide; if you log both, use **`Val Acc last`** for final epoch.
- **K-fold:** For each fold, max val acc over epochs → then **`max Val Acc (mean)`** = mean of fold-wise maxima; document per-fold maxima in `Notes` when fold spread matters.
- **`epoch @ max`:** Epoch index where the monitored validation accuracy reaches its maximum (per fold or single run).

**Secondary:** **`max macro-F1`** with the same best-over-epochs rule on validation.

**Test set (ADReSSo test-dist):** **`Test acc`** / **`Test macro-F1`** = ensemble over folds from `modules/test.py` (`test()`): mean of sigmoid probabilities across `model_fold_*.pth`, then threshold 0.5, evaluated on labeled subjects (`task1.csv` in repo root). Distinct from CV val metrics above.

---

## AGENT: how to auto-fill this log

1. **`max Val Acc`** = `max(val_accuracy[t])` over training steps/epochs `t` from the validation curve.
2. **`epoch @ max`** = `argmax_t val_accuracy[t]` (per fold if applicable).
3. **Do not** substitute the final epoch’s validation accuracy for **`max Val Acc`** without checking the full curve.
4. CV: report **`max Val Acc (mean)`** as mean of per-fold maxima; flag suspicious folds (e.g. 100%) in `Notes`.

---

## What this file is for

- **Project:** CogniAligned.
- **Task:** ADReSSo-style **HC vs AD** (or equivalent in your config).
- **Typical entrypoint:** `modules/main.py` + configs / SLURM scripts in repo.

---

## How to append (field cheat-sheet)

| Field | Record |
| :--- | :--- |
| `ID` | Job id / run name |
| **`Experiment`** | **1–7 words**; Changelog title |
| `Model / fusion` | e.g. CogniAlign, cross-attention |
| `Config / branch` | yaml / git / SLURM script path |
| `Features` | Audio + text + acoustic dim |
| **`max Val Acc (mean)`** | Best-over-epochs, mean over folds if CV |
| **`epoch @ max`** | As defined above |
| **`max macro-F1 (mean)`** | Best-over-epochs if logged |
| **`Val Acc last`** | Optional |
| **`Test acc`** | `modules/test.py` ensemble on **test-dist** (see How to run) |
| **`Test macro-F1`** | Same run; sklearn banner / `classification_report` macro |
| `Notes` | Split sanity, class balance, log path |

---

## Run summary (living table)

| ID | Date | **Experiment** | Model / fusion | Config / branch | Features | Split | **max Val Acc (mean)** | **epoch @ max** | **max macro-F1 (mean)** | Val Acc last (opt.) | **Test acc** | **Test macro-F1** | **Params** | **Infer. (ms/spl)** | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2538671 | 2026-05-05 | Cogni ADReSSo cross plus AF | CrossAttentionTransformerEncoder | `slurm/train_af.sh` → `modules/main.py --config modules/configs/default.yaml` | audio + text + 55-dim AF → 768 proj | 5-fold CV | **86.63%** | per-fold W&B `best_epoch` 30, 21, 17, 21, 25 | **0.8637** | — | **77.46%** | **0.7647** | `logs/slurm/cogni_adresso_af_2538671.txt`; test ensemble 2026-05-05; preds `logs/adresso_distil_wav2vec2_P_cross_mean/test_predictions.csv`; per-fold max val acc: 91.18%, 87.88%, 72.73%, 84.38%, 96.97% | — || — | |
| 2538764 | 2026-05-05 | Full train max test-dist bicross | BidirectionalCrossAttentionTransformerEncoder | `slurm/train_maximize_test_accuracy.sh` → `modules/configs/default_bicross_fulltrain_testval.yaml` | audio + text + 55-dim AF | full train / **val = test-dist** | **83.10%** | **30** (early stop epoch 45) | **0.8301** | — | — | — | W&B `best/test_accuracy` = `best/val_accuracy` (same loader); no `test.py` ensemble. `logs/slurm/cogni_max_testacc_2538764.txt`; run `distil_wav2vec2_P_bicross_job2538764` | — || — | |
| 2538765 | 2026-05-05 | Full train max test-dist cross | CrossAttentionTransformerEncoder | `slurm/train_maximize_test_accuracy.sh` → `modules/configs/default_cross_fulltrain_testval.yaml` | same | full train / **val = test-dist** | **83.10%** | **31** (early stop epoch 46) | **0.8293** | — | — | — | `logs/slurm/cogni_max_testacc_2538765.txt`; run `distil_wav2vec2_P_cross_job2538765` | — || — | |
| 2538766 | 2026-05-05 | Full train max test-dist elementwise | ElementWiseFusionEncoder (`fusion: ''`) | `slurm/train_maximize_test_accuracy.sh` → `modules/configs/default_elementwise_fulltrain_testval.yaml` | same | full train / val = test-dist | — | — | — | — | — | — | **Failed** epoch 1: `AttributeError: 'list' object has no attribute 'dtype'` in `model.py` → `TransformerEncoder` (`src` list vs tensor). `logs/slurm/cogni_max_testacc_2538766.txt`; W&B `distil_wav2vec2_P__job2538766` | — || — | |
| 2553273 | 2026-05-09 | Mamba fusion 5-fold CV | MambaFusionEncoder (pure-PyTorch SSM, seq-cat fusion) | `slurm/train_mamba_adresso.sh` → `modules/configs/default_mamba.yaml`; `fusion: mamba`; d_state=16, d_conv=4, expand=2 | audio + text + 55-dim AF | 5-fold CV | **88.43%** | per-fold best_epoch 43, 26, 36, 23, 28 | **0.8822** | — | — | — | `logs/slurm/cogni_mamba_adresso_2553273.txt`; per-fold acc: 94.12%, 87.88%, 78.79%, 84.38%, 96.97%; 11.1M params (3.77M Mamba + 7.09M AF-attn + 0.20M head) | — || — | |
| 2565642 | 2026-05-10 | Cross ADReSSo | see config/cross_adresso.yaml | `slurm/train_cross_adresso.sh` | audio+text+AF | 5-fold CV | **87.82%** | 45, 21, 17, 21, 25 | **0.8766** | — | — | — | `slurm/cross_adresso_2565642.txt`; per-fold acc: 94.12%, 90.91%, 72.73%, 84.38%, 96.97% | — || — | |
| 2565643 | 2026-05-10 | BiCross ADReSSo | see config/bicross_adresso.yaml | `slurm/train_bicross_adresso.sh` | audio+text+AF | 5-fold CV | **85.41%** | 31, 24, 14, 26, 25 | **0.8517** | — | — | — | `slurm/bicross_adresso_2565643.txt`; per-fold acc: 91.18%, 87.88%, 69.70%, 84.38%, 93.94% | — || — | |
| 2565644 | 2026-05-10 | Mamba ADReSSo | see config/mamba_adresso.yaml | `slurm/train_mamba_adresso.sh` | audio+text+AF | 5-fold CV | **89.07%** | 31, 25, 59, 25, 33 | **0.8876** | — | — | — | `slurm/mamba_adresso_2565644.txt`; per-fold acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97% | — || — | |
| 2565955 | 2026-05-10 | Mamba ADReSSo lr1e-5 | see config/mamba_adresso_lr1e5.yaml | `slurm/train_mamba_adresso_lr1e5.sh` | audio+text+AF | 5-fold CV | **89.07%** | 31, 25, 59, 25, 33 | **0.8876** | — | — | — | `slurm/mamba_adresso_lr1e5_2565955.txt`; per-fold acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97% | — || — | |
| 2565956 | 2026-05-10 | Mamba ADReSSo lr5e-5 | see config/mamba_adresso_lr5e5.yaml | `slurm/train_mamba_adresso_lr5e5.sh` | audio+text+AF | 5-fold CV | **89.07%** | 31, 25, 59, 25, 33 | **0.8876** | — | — | — | `slurm/mamba_adresso_lr5e5_2565956.txt`; per-fold acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97% | — || — | |
| 2565642 | 2026-05-10 | Cross ADReSSo | see config/cross_adresso.yaml | `slurm/train_cross_adresso.sh` | audio+text+AF | 5-fold CV | **87.82%** | 45, 21, 17, 21, 25 | **0.8766** | — | — | — | `slurm/cross_adresso_2565642.txt`; per-fold acc: 94.12%, 90.91%, 72.73%, 84.38%, 96.97% | — || — | |
| 2565643 | 2026-05-10 | BiCross ADReSSo | see config/bicross_adresso.yaml | `slurm/train_bicross_adresso.sh` | audio+text+AF | 5-fold CV | **85.41%** | 31, 24, 14, 26, 25 | **0.8517** | — | — | — | `slurm/bicross_adresso_2565643.txt`; per-fold acc: 91.18%, 87.88%, 69.70%, 84.38%, 93.94% | — || — | |
| 2565644 | 2026-05-10 | Mamba ADReSSo | see config/mamba_adresso.yaml | `slurm/train_mamba_adresso.sh` | audio+text+AF | 5-fold CV | **89.07%** | 31, 25, 59, 25, 33 | **0.8876** | — | — | — | `slurm/mamba_adresso_2565644.txt`; per-fold acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97% | — || — | |
| 2565642 | 2026-05-10 | Cross ADReSSo | see config/cross_adresso.yaml | `slurm/train_cross_adresso.sh` | audio+text+AF | 5-fold CV | **87.82%** | 45, 21, 17, 21, 25 | **0.8766** | — | — | — | `slurm/cross_adresso_2565642.txt`; per-fold acc: 94.12%, 90.91%, 72.73%, 84.38%, 96.97% | — || — | |
| 2565643 | 2026-05-10 | BiCross ADReSSo | see config/bicross_adresso.yaml | `slurm/train_bicross_adresso.sh` | audio+text+AF | 5-fold CV | **85.41%** | 31, 24, 14, 26, 25 | **0.8517** | — | — | — | `slurm/bicross_adresso_2565643.txt`; per-fold acc: 91.18%, 87.88%, 69.70%, 84.38%, 93.94% | — || — | |
| 2565644 | 2026-05-10 | Mamba ADReSSo | see config/mamba_adresso.yaml | `slurm/train_mamba_adresso.sh` | audio+text+AF | 5-fold CV | **89.07%** | 31, 25, 59, 25, 33 | **0.8876** | — | — | — | `slurm/mamba_adresso_2565644.txt`; per-fold acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97% | — || — | |
| 2569845 | 2026-05-10 | Mamba ADReSSo lr1e-5 | see config/mamba_adresso_lr1e5.yaml | `slurm/train_mamba_adresso_lr1e5.sh` | audio+text+AF | 5-fold CV | **79.18%** | 27, 36, 43, 2, 28 | **0.7708** | — | — | — | `slurm/mamba_adresso_lr1e5_2569845.txt`; per-fold acc: 88.23%, 87.88%, 72.73%, 53.12%, 93.94% | — || — | |
| 2569846 | 2026-05-10 | Mamba ADReSSo lr5e-5 | see config/mamba_adresso_lr5e5.yaml | `slurm/train_mamba_adresso_lr5e5.sh` | audio+text+AF | 5-fold CV | **88.44%** | 17, 19, 41, 16, 25 | **0.8813** | — | — | — | `slurm/mamba_adresso_lr5e5_2569846.txt`; per-fold acc: 91.18%, 87.88%, 81.82%, 84.38%, 96.97% | — || — | |
---

## Changelog

Style: `example_experiment.md`. Append new `###` blocks at **bottom** (convention).

### Changelog template (copy per run)

```markdown
### Experiment <ID>: <1–7 word title>

**Status:** Complete ✓ | In progress | Failed  
**Results:** max Val Acc (mean) = … \| epoch @ max = … \| max macro-F1 (mean) = … \| Val Acc last = … (optional)

**Config:**
- Model / fusion, yaml, branch:

**What changed:**

**Observations:** (per-fold max Val Acc if CV)

**Artifacts:**
```

### Experiment 2538671: Cogni ADReSSo cross plus AF
25386712538671
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **86.63%** | epoch @ max = per-fold 30, 21, 17, 21, 25 | max macro-F1 (mean) = **0.8637**  
**Test (ensemble, test-dist, 2026-05-05):** acc = **77.46%** | F1 (banner) = **0.7647** | precision **0.7879** | recall **0.7429** | `classification_report` macro avg F1 **0.77**

**Config:** `modules/configs/default.yaml`, cross-attention encoder, ADReSSo train acoustic CSV; `fusion` overridden to **`cross`** for test to match checkpoint dir `adresso_distil_wav2vec2_P_cross_mean`.

**Artifacts:** `/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/slurm/cogni_adresso_af_2538671.txt` · predictions `/home/usuaris/veussd/roger.esteve.sanchez/CogniAligned/logs/adresso_distil_wav2vec2_P_cross_mean/test_predictions.csv`

### Experiment 2538764: Full train max test bicross
25387642538764
**Status:** Complete ✓  
**Results:** max monitored test-dist acc (W&B `best/val_accuracy`) = **83.10%** (`0.83099`) \| **`epoch @ max` = 30** \| max val F1 = **0.8301** \| early stopping at epoch **45**

**Config:** `default_bicross_fulltrain_testval.yaml`; `train.validate_on_test_dist: true`; checkpoint reload per `train_maximize_test_accuracy.sh`.

**Artifacts:** `logs/slurm/cogni_max_testacc_2538764.txt` · W&B run `distil_wav2vec2_P_bicross_job2538764`

### Experiment 2538765: Full train max test cross
25387652538765
**Status:** Complete ✓  
**Results:** max monitored test-dist acc = **83.10%** (`0.83099`) \| **`epoch @ max` = 31** \| max val F1 = **0.8293** \| early stopping at epoch **46**

**Config:** `default_cross_fulltrain_testval.yaml` (`fusion: cross`).

**Observations:** Same peak **test-dist** accuracy as bicross to five decimals; cross-attention best epoch one later; F1 slightly lower than bicross on this monitor.

**Artifacts:** `logs/slurm/cogni_max_testacc_2538765.txt` · W&B run `distil_wav2vec2_P_cross_job2538765`

### Experiment 2538766: Full train max test elementwise
25387662538766
**Status:** Failed  
**Results:** — (crashed before first epoch completed)

**Config:** `default_elementwise_fulltrain_testval.yaml` (`fusion: ''` → `ElementWiseFusionEncoder`).

**Observations:** First training step raises `AttributeError: 'list' object has no attribute 'dtype'` — `ElementWiseFusionEncoder.forward` passes a **list** of modality tensors into `nn.TransformerEncoder`, which expects a single tensor `src`. Fix would be to stack/concat modalities before the encoder or adapt the fusion path for this dataloader feature format.

**Artifacts:** `logs/slurm/cogni_max_testacc_2538766.txt` · W&B run `distil_wav2vec2_P__job2538766` (partial)

### Experiment 2553273: Mamba fusion 5-fold CV
25532732553273
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **88.43%** | epoch @ max = per-fold 43, 26, 36, 23, 28 | max macro-F1 (mean) = **0.8822** | early stopping per fold at 58, 41, 51, 38, 43

**Config:** `modules/configs/default_mamba.yaml`; `fusion: mamba`; `mamba_d_state: 16`, `mamba_d_conv: 4`, `mamba_expand: 2`; n_layers=6; hidden_size=768; lr=2e-5 (AdamW); cosine LR schedule with 20-epoch warmup.

**What changed:** Replaced bidirectional cross-attention encoder with a pure-PyTorch Mamba SSM (`MambaBlock` + `MambaFusionEncoder` in `modules/model.py`). Audio and text token sequences are concatenated along the sequence dimension and passed through `n_layers` Mamba blocks; acoustic feature vector is fused as a learnable token via a small self-attention layer.

**Observations:**
- Per-fold max Val Acc: 94.12%, 87.88%, 78.79%, 84.38%, 96.97%
- Mean **88.43%** vs bicross **86.63%** (5-fold CV, same split) — Mamba slightly outperforms.
- High fold variance (F2 low at 78.79%, F4 high at 96.97%) consistent with small dataset size.

**Artifacts:** `logs/slurm/cogni_mamba_adresso_2553273.txt` · W&B project `CogniAligned-ADReSSo`

---

## Provisional baseline (legacy — metric definition unclear)

From `CogniAligned/EXPERIMENTS.md`. Legacy “Val Acc” may be last epoch or fold snapshot — **re-derive `max Val Acc` from curves when possible.**

| Job | Model | Inputs | Val Acc (legacy) | macro-F1 (legacy) | max Val Acc (new log) | Test acc | Test macro-F1 | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2397188 | CrossAttention | Audio + Text + Acoustic (47) | Failed | Failed | — | — | — | TypeError |
| **2398091** | CogniAlign (late fusion) | Audio + Text + Acoustic (55) | **87.8%** | **0.876** | — | — | — | High fold spread; fold 4 was **100%** in legacy table — audit split |

Per-fold legacy snapshot (2398091): F0 91.2%, F1 87.9%, F2 75.8%, F3 84.4%, F4 **100%** — **not** using max-over-epochs unless recomputed.

---


### Experiment 2565642: Cross ADReSSo
25656422565642
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **87.82%** | epoch @ max = per-fold 45, 21, 17, 21, 25 | max macro-F1 (mean) = **0.8766** | **Params:** 14.42M | **Inference:** 0.40 ms/splnfig:** `slurm/train_cross_adresso.sh`; fusion=cross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 94.12%, 90.91%, 72.73%, 84.38%, 96.97%

**Artifacts:** `CogniAligned/logs/slurm/cross_adresso_2565642.txt`


### Experiment 2565643: BiCross ADReSSo
25656432565643
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **85.41%** | epoch @ max = per-fold 31, 24, 14, 26, 25 | max macro-F1 (mean) = **0.8517** | **Params:** 21.51M | **Inference:** 0.89 ms/splnfig:** `slurm/train_bicross_adresso.sh`; fusion=bicross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 91.18%, 87.88%, 69.70%, 84.38%, 93.94%

**Artifacts:** `CogniAligned/logs/slurm/bicross_adresso_2565643.txt`


### Experiment 2565644: Mamba ADReSSo
25656442565644
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **89.07%** | epoch @ max = per-fold 31, 25, 59, 25, 33 | max macro-F1 (mean) = **0.8876** | **Params:** 11.10M | **Inference:** 0.65 ms/splnfig:** `slurm/train_mamba_adresso.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97%

**Artifacts:** `CogniAligned/logs/slurm/mamba_adresso_2565644.txt`


### Experiment 2565955: Mamba ADReSSo lr1e-5
25659552565955
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **89.07%** | epoch @ max = per-fold 31, 25, 59, 25, 33 | max macro-F1 (mean) = **0.8876** | **Params:** 11.10M | **Inference:** 1.36 ms/splnfig:** `slurm/train_mamba_adresso_lr1e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97%

**Artifacts:** `CogniAligned/logs/slurm/mamba_adresso_lr1e5_2565955.txt`


### Experiment 2565956: Mamba ADReSSo lr5e-5
25659562565956
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **89.07%** | epoch @ max = per-fold 31, 25, 59, 25, 33 | max macro-F1 (mean) = **0.8876** | **Params:** 11.10M | **Inference:** 0.96 ms/splnfig:** `slurm/train_mamba_adresso_lr5e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97%

**Artifacts:** `CogniAligned/logs/slurm/mamba_adresso_lr5e5_2565956.txt`


### Experiment 2565642: Cross ADReSSo
25656422565642
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **87.82%** | epoch @ max = per-fold 45, 21, 17, 21, 25 | max macro-F1 (mean) = **0.8766** | **Params:** 14.42M | **Inference:** 0.40 ms/splnfig:** `slurm/train_cross_adresso.sh`; fusion=cross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 94.12%, 90.91%, 72.73%, 84.38%, 96.97%

**Artifacts:** `CogniAligned/logs/slurm/cross_adresso_2565642.txt`


### Experiment 2565643: BiCross ADReSSo
25656432565643
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **85.41%** | epoch @ max = per-fold 31, 24, 14, 26, 25 | max macro-F1 (mean) = **0.8517** | **Params:** 21.51M | **Inference:** 0.89 ms/splnfig:** `slurm/train_bicross_adresso.sh`; fusion=bicross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 91.18%, 87.88%, 69.70%, 84.38%, 93.94%

**Artifacts:** `CogniAligned/logs/slurm/bicross_adresso_2565643.txt`


### Experiment 2565644: Mamba ADReSSo
25656442565644
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **89.07%** | epoch @ max = per-fold 31, 25, 59, 25, 33 | max macro-F1 (mean) = **0.8876** | **Params:** 11.10M | **Inference:** 0.65 ms/splnfig:** `slurm/train_mamba_adresso.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97%

**Artifacts:** `CogniAligned/logs/slurm/mamba_adresso_2565644.txt`


### Experiment 2565642: Cross ADReSSo
25656422565642
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **87.82%** | epoch @ max = per-fold 45, 21, 17, 21, 25 | max macro-F1 (mean) = **0.8766** | **Params:** 14.42M | **Inference:** 0.40 ms/splnfig:** `slurm/train_cross_adresso.sh`; fusion=cross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 94.12%, 90.91%, 72.73%, 84.38%, 96.97%

**Artifacts:** `CogniAligned/logs/slurm/cross_adresso_2565642.txt`


### Experiment 2565643: BiCross ADReSSo
25656432565643
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **85.41%** | epoch @ max = per-fold 31, 24, 14, 26, 25 | max macro-F1 (mean) = **0.8517** | **Params:** 21.51M | **Inference:** 0.89 ms/splnfig:** `slurm/train_bicross_adresso.sh`; fusion=bicross; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 91.18%, 87.88%, 69.70%, 84.38%, 93.94%

**Artifacts:** `CogniAligned/logs/slurm/bicross_adresso_2565643.txt`


### Experiment 2565644: Mamba ADReSSo
25656442565644
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **89.07%** | epoch @ max = per-fold 31, 25, 59, 25, 33 | max macro-F1 (mean) = **0.8876** | **Params:** 11.10M | **Inference:** 0.65 ms/splnfig:** `slurm/train_mamba_adresso.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 91.18%, 87.88%, 81.82%, 87.50%, 96.97%

**Artifacts:** `CogniAligned/logs/slurm/mamba_adresso_2565644.txt`


### Experiment 2569845: Mamba ADReSSo lr1e-5
25698452569845
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **79.18%** | epoch @ max = per-fold 27, 36, 43, 2, 28 | max macro-F1 (mean) = **0.7708** | **Params:** 11.10M | **Inference:** 1.36 ms/splnfig:** `slurm/train_mamba_adresso_lr1e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 88.23%, 87.88%, 72.73%, 53.12%, 93.94%

**Artifacts:** `CogniAligned/logs/slurm/mamba_adresso_lr1e5_2569845.txt`


### Experiment 2569846: Mamba ADReSSo lr5e-5
25698462569846
**Status:** Complete ✓  
**Results:** max Val Acc (mean) = **88.44%** | epoch @ max = per-fold 17, 19, 41, 16, 25 | max macro-F1 (mean) = **0.8813** | **Params:** 11.10M | **Inference:** 0.96 ms/splnfig:** `slurm/train_mamba_adresso_lr5e5.sh`; fusion=mamba; standard CogniAligned setup.

**Observations:**
- Per-fold max Val Acc: 91.18%, 87.88%, 81.82%, 84.38%, 96.97%

**Artifacts:** `CogniAligned/logs/slurm/mamba_adresso_lr5e5_2569846.txt`

## Archive

Raw table: `CogniAligned/EXPERIMENTS.md`.
