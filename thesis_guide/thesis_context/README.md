# Thesis context bundle

Portable context pack for a **writing agent** (another platform). Read **`doc_guide.md`** first (master instructions).

**Canonical location:** `/home/usuaris/veussd/roger.esteve.sanchez/thesis_context/`  
**Sync note:** `doc_guide.md` is copied from the repo root; if the root file changes, re-copy:  
`cp ../doc_guide.md ./doc_guide.md`

## Folder map

| Folder | Contents | Use for |
|--------|----------|---------|
| **`doc_guide.md`** | Full writing spec, narrative, section outlines | Primary agent instructions |
| **`00_planning/`** | `thesis_ablation_plan.md`, `experiments_plan.md` | Experiment phases A/B/C, fairness caveats |
| **`01_latex/`** | `main.tex`, `state_of_the_art.tex`, `Odyssey2026_BibEntries.bib` | Thesis skeleton, finished SOTA, citations |
| **`02_results/`** | All ablation CSVs (+ `_es` Spanish decimal) | Tables and numbers — **source of truth** |
| **`03_pdfs/`** | `AD_Disease_Odyssey.pdf`, `Master_Thesis_Lucas_Takanori.pdf` | Mid-project paper; Lucas thesis structure/tone |
| **`04_code/`** | Alignment preprocess + SER embedding scripts | Methodology accuracy (alignment modes, Wav2Vec) |
| **`05_orchestrators/`** | `run_*.py` for each ablation phase | How experiments were defined (not for line-by-line reading) |
| **`06_experiment_notes/`** | `cogni_*` / `ser_*` `.md` | **Stale** (~May 11): templates only — ablation numbers are in `02_results/` |
| **`07_configs/`** | Sample YAML configs | Example training settings |

## File index (`02_results/`)

| File | Phase | Use |
|------|-------|-----|
| `modality_ablation.csv` | A — modality | CogniAligned only |
| `step_b_results.csv` | B — fusion | Both implementations |
| `alignment_tier_a.csv` | C — alignment | CogniAligned; uniform/char vs word→tok |
| `alignment_wv2vec_ablation.csv` | Align × Wav2Vec | Main alignment + weighted grid |
| `wav2vec_layer_ablation.csv` | Wv2Vec final vs weighted | Both implementations |
| `alignment_pilot.csv` | Pilot only | Appendix / footnote |
| `experiment_results.csv` | Legacy sweep | **Do not** use for main tables |

Prefer `*_es.csv` only if the thesis is written in Spanish with comma decimals.

## File index (`04_code/`)

| File | Purpose |
|------|---------|
| `preprocessembeddings.py` | **Alignment algorithms** (`word_token`, `token_token_uniform`, `token_token_char`) |
| `run_preprocessing.py` | ADReSSo preprocess CLI (`--alignment_mode`, `--wav2vec2_layer_mode`) |
| `run_preprocessing_wab.py` | Amyloid/PPA (WAB tree) preprocess |
| `speech_feature_extractor.py` | SER Wav2Vec layer modes |
| `precompute_ser_embeddings.py` | SER embedding cache |

## Not included (remain on cluster / full repo)

- Full `CogniAligned/` and `ad-detection/` trees, trained checkpoints, W&B logs
- `introduction.tex` (not written yet)
- `tools/packages.tex`, `bibliography.bib` referenced by `main.tex` (may be elsewhere)
- SLURM logs under `CogniAligned/logs/`, `ad-detection/wandb/`

For those, see paths in `doc_guide.md` (§N bundle, §O gaps).
