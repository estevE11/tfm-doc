# Experiments Plan

Goal: compare models not only by validation/test accuracy, but also by parameter efficiency and runtime:

- `total_parameters`
- `trainable_parameters`
- `trainable_parameters_pct`
- `inference_ms_per_sample`
- `inference_samples_per_second`
- `best_val_accuracy`
- `best_macro_f1`

All new runs should log these metrics to W&B and the experiment `.md` logs.

---

## Currently Running

Update this section when launching and completing jobs. When a training job finishes, also update the task-specific experiment log.

| Job ID | Started | Family | Task | Experiment | Status | Node | Follow-up logs |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| — | 2026-05-10 20:03 | — | — | Batch **ALL BATCHES** completed. Next batch queued. | DONE | — | — |

---

## Immediate Priorities

1. **Create fair 5-fold baselines for SER SelfAttention.** Current SER baselines are mostly single-split, while Mamba was run as 5-fold CV.
2. **Do not launch large SER Mamba sweeps until the runtime bottleneck is fixed.** The current pure-PyTorch Mamba selective scan is much slower than attention in practice.
3. **Run small architecture sweeps with the same frozen encoder policy.** This isolates the fusion module rather than the base encoders.
4. **Only after the fusion comparison is stable, test unfreezing and larger learning-rate sweeps.** Unfreezing changes both capacity and runtime too much for the first comparison.

---

## Runtime Investigation Notes

The long SER runtimes are expected from the current implementation, but not because Mamba is theoretically slow.

Main causes:

- **CogniAligned uses precomputed embeddings.** Its datasets load `.pt` tensors for text/audio/acoustic features, then train only the fusion/classifier stack.
- **SER computes Wav2Vec2 XLSR-300M and ModernBERT features inside every training batch.** This makes SER inherently heavier than CogniAligned even before comparing fusion modules.
- **The current Mamba implementation is pure PyTorch and uses a Python `for t in range(L)` selective scan.** This creates thousands of tiny GPU operations per forward pass instead of one fused CUDA/Triton scan.
- **SER Mamba operates on long sequences.** A 40-second Wav2Vec2 window produces a long audio-token sequence, then text tokens are concatenated. With 4 Mamba blocks, the Python scan repeats over the full concatenated sequence 4 times per batch.
- **Attention baselines use batched matrix multiplications (`torch.bmm`).** Even though attention is theoretically quadratic, PyTorch runs it as large optimized kernels, so it is much faster here than the unfused Mamba scan.

Observed SER training speeds from current logs:

| Run family | Mean speed |
| :--- | :---: |
| SER ADReSSo SelfAttention | ~40 batches/min |
| SER Amyloid SelfAttention | ~55 batches/min |
| SER PPA SelfAttention | ~42 batches/min |
| SER ADReSSo Mamba | ~1.6 batches/min |
| SER Amyloid Mamba | ~2.5 batches/min |
| SER PPA Mamba | ~2.1 batches/min |

Conclusion: **current SER Mamba is 15–30x slower than SER SelfAttention**. This is implementation overhead, not a proof that Mamba is slower as a model family.

Recommended fixes before more SER Mamba sweeps:

1. **Precompute SER embeddings** from Wav2Vec2 + ModernBERT and train only fusion modules, matching CogniAligned's speed profile. Implemented via `ad-detection/scripts/precompute_ser_embeddings.py` and `--use_precomputed_embeddings`.
2. **Replace pure-PyTorch Mamba with optimized `mamba-ssm`/Triton selective scan** if the environment supports it. Implemented in `MambaSeqToSeq`: it uses `mamba_ssm.Mamba` on CUDA and falls back to the slow PyTorch block only when CUDA or `mamba-ssm` is unavailable.
3. **Shorten the SER sequence length** using first-window-only, pooling/downsampling Wav2Vec2 frames, or projecting audio tokens before fusion.
4. **Benchmark only the fusion layer** on cached embeddings to report fair inference-time comparisons between attention and Mamba.

Implementation status (2026-05-10):

- Installed `mamba-ssm` in `ad-detection/.venv` after loading `cuda/12.8` for `nvcc`.
- Added `CachedEmbeddingDataset` and `pad_collate_embeddings`.
- Added cached-embedding arguments to all SER trainers:
  - `--use_precomputed_embeddings`
  - `--precomputed_embeddings_dir`
- Updated SER Mamba run scripts to use cached embeddings.
- Submitted precompute job `2564785` (`ser_precompute_embeddings`).

---

## SER Experiments

Base setting unless overridden:

- Scripts: `ad-detection/adresso_classify/train_adresso.py`, `ad-detection/amyloid/train_amyloid.py`, `ad-detection/ad_classification/train_ad.py`
- Tasks: ADReSSo, Amyloid, PPA
- Split: 5-fold CV
- Speech encoder: `WAV2VEC2_XLSR_300M`
- Text encoder: `MODERN_BERT_BASE`
- Adapters: `LinearAdapter`, 256 dim
- Pooling: `AttentionPooling`
- Acoustic features: enabled
- Freeze policy: `--freeze_base_models_only`
- Batch size: 4
- Epochs: 25 for baseline sweep, 50 for long Mamba sweep

| Priority | fusion / seq_to_seq_method | trans_heads | lr | dropout | mamba_blocks | mamba_state | epochs | scheduler | freeze policy | Why |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- | :--- |
| P0 | `SelfAttention` | n/a | 5e-5 | 0.3 | n/a | n/a | 25 | plateau | base frozen | Fair 5-fold baseline for all tasks |
| P0 | `MultiHeadAttention` | 2 | 5e-5 | 0.3 | n/a | n/a | 25 | plateau | base frozen | Parameter/runtime comparison vs simple self-attention |
| P0 | `MultiHeadAttention` | 4 | 5e-5 | 0.3 | n/a | n/a | 25 | plateau | base frozen | Check if more heads improve accuracy enough to justify cost |
| P1 | `CrossAttention` | 4 | 5e-5 | 0.3 | n/a | n/a | 25 | plateau | base frozen | Direct multimodal interaction baseline |
| P1 | `CrossAttentionReduced` | 4 | 5e-5 | 0.3 | n/a | n/a | 25 | plateau | base frozen | Faster/lighter cross-attention variant |
| P1 | `MambaSeqToSeq` | n/a | 5e-5 | 0.3 | 4 | 16 | 50 | plateau patience 15 | base frozen | Current Mamba underfit/slow convergence check |
| P1 | `MambaSeqToSeq` | n/a | 1e-4 | 0.3 | 4 | 16 | 50 | plateau patience 15 | base frozen | Test if Mamba needs higher LR |
| P2 | `MambaSeqToSeq` | n/a | 5e-5 | 0.3 | 8 | 16 | 50 | plateau patience 15 | base frozen | More depth, modest state size |
| P2 | `MambaSeqToSeq` | n/a | 5e-5 | 0.3 | 4 | 32 | 50 | plateau patience 15 | base frozen | Larger SSM state, similar depth |
| P2 | `TransformerStacked` | 4 | 5e-5 | 0.3 | n/a | n/a | 25 | plateau | base frozen | Full transformer fusion cost/accuracy reference |
| P3 | `SelfAttention` | n/a | 1e-5 | 0.3 | n/a | n/a | 25 | plateau | no base freeze | Check benefit/cost of end-to-end fine-tuning |
| P3 | `MambaSeqToSeq` | n/a | 1e-5 | 0.3 | 4 | 16 | 50 | plateau patience 15 | no base freeze | Test whether Mamba needs representation adaptation |

Recommended first batch:

| Batch | Experiments | Tasks |
| :--- | :--- | :--- |
| SER-A | `SelfAttention`, `MultiHeadAttention` heads 2/4, `CrossAttentionReduced` | ADReSSo, Amyloid, PPA |
| SER-B | Mamba long-run: 50 epochs, patience 15, lr 5e-5 and 1e-4 | ADReSSo, Amyloid, PPA |
| SER-C | Larger Mamba: blocks 8, state 16; blocks 4, state 32 | Best task from SER-B first, then all tasks if promising |

---

## CogniAligned Experiments

Base setting unless overridden:

- Scripts: `CogniAligned/modules/main.py`, `CogniAligned/modules/amyloid/main.py`, `CogniAligned/modules/ppa/main.py`
- Tasks: ADReSSo, Amyloid, PPA
- Split: 5-fold CV
- Features: text + audio + acoustic features
- Optimizer: AdamW
- Scheduler: cosine with warmup
- Compare by accuracy, macro-F1, total parameters, trainable parameters, and inference time.

| Priority | fusion | trans_heads | lr | dropout | mamba_blocks / n_layers | mamba_state | epochs | pooling | Why |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| P0 | `cross` | 8 | 2e-5 | 0.3 | n/a | n/a | current | mean | Paper-style baseline with new timing/params logging |
| P0 | `bicross` | 8 | 2e-5 | 0.3 | n/a | n/a | current | mean | Strong CogniAligned reference |
| P0 | `mamba` | n/a | 2e-5 | 0.3 | 6 | 16 | current | mean | Current Mamba baseline with new logging |
| P1 | `mamba` | n/a | 1e-5 | 0.3 | 6 | 16 | current | mean | Check if Mamba benefits from lower LR |
| P1 | `mamba` | n/a | 5e-5 | 0.3 | 6 | 16 | current | mean | Check if Mamba benefits from higher LR |
| P1 | `mamba` | n/a | 2e-5 | 0.2 | 6 | 16 | current | mean | Lower regularization |
| P1 | `mamba` | n/a | 2e-5 | 0.4 | 6 | 16 | current | mean | Higher regularization |
| P2 | `mamba` | n/a | 2e-5 | 0.3 | 4 | 16 | current | mean | Smaller/faster Mamba |
| P2 | `mamba` | n/a | 2e-5 | 0.3 | 8 | 16 | current | mean | Deeper Mamba |
| P2 | `mamba` | n/a | 2e-5 | 0.3 | 6 | 32 | current | mean | Larger SSM state |
| P2 | `cross` | 4 | 2e-5 | 0.3 | n/a | n/a | current | mean | Cheaper attention baseline |
| P2 | `cross` | 2 | 2e-5 | 0.3 | n/a | n/a | current | mean | Very cheap attention baseline |
| P3 | `bicross` | 4 | 2e-5 | 0.3 | n/a | n/a | current | mean | Parameter/runtime trade-off vs 8 heads |

Recommended first batch:

| Batch | Experiments | Tasks |
| :--- | :--- | :--- |
| Cogni-A | Re-run `cross`, `bicross`, `mamba` with new parameter/timing logging | ADReSSo, Amyloid, PPA |
| Cogni-B | Mamba LR sweep: 1e-5, 2e-5, 5e-5 | Amyloid first, because Mamba underperformed there |
| Cogni-C | Cheap attention: `cross` heads 2/4/8 | ADReSSo and PPA |
| Cogni-D | Mamba scale: layers 4/6/8 and state 16/32 | Best task after Cogni-B |

---

## Reporting Template

Each final comparison table should include:

| model family | task | fusion | heads | lr | dropout | total params | trainable params | trainable % | inference ms/sample | best val acc | best macro-F1 |
| :--- | :--- | :--- | :---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | ---: |

Notes:

- Keep parameter counts and inference timing from the same run that produced the accuracy result.
- For DDP runs, inference time is aggregated across ranks, so compare only runs with the same GPU count and similar hardware where possible.
- When hardware differs by node, report the node in the experiment log notes.
