# Figures to build by hand

Status of all thesis figures. The four data plots and four schematic diagrams below
are already implemented (TikZ/pgfplots, in the PDF). The remaining **four** need to be
drawn by hand (draw.io / Inkscape), exported to `img/`, and dropped in.

## How to insert a hand-drawn figure
Export as **PDF or SVG** into `img/`, then in the `.tex` replace the placeholder line

```latex
\fbox{\parbox{0.9\linewidth}{\vspace{4cm}\centering\small\textit{[Figure to be added]}\vspace{4cm}}}
```

with

```latex
\includegraphics[width=\linewidth]{img/<name>}
```

Keep the surrounding `\caption{...}` and `\label{...}` unchanged.

## Style, so they match the auto-generated figures
- Palette: **blue!8** = frozen, **orange!15** = trained; **blue!25 / green!30 / orange!35**
  for audio / text / groupings.
- Rounded-corner boxes, thin arrows with Stealth (filled) heads.

---

## Already done (no action needed)

**Results (pgfplots):**
- `fig:res_fusion_bar` — fusion-family accuracy across the three tasks
- `fig:res_compute` — accuracy vs per-sample inference time
- `fig:res_layer_mixer` — learned per-layer sample-mixer weights

**Schematics (TikZ):**
- `fig:system_architecture` — end-to-end two-stream pipeline
- `fig:alignment_schemes` — word→token vs uniform vs proportional
- `fig:fusion_paradigms` — early / intermediate / late
- `fig:cross_modal_alignment` — unsynchronised vs word→token aligned

---

## TO BUILD (4)

### A. `fig:wav2vec_architecture` — Wav2Vec 2.0 encoder (State of the Art)

Left→right block flow:
1. **Raw waveform** (16 kHz) →
2. **CNN feature encoder** (label "7 conv layers") → output **latent frames _z_**,
   drawn as a short row of boxes; annotate "~20 ms / 50 fps" →
3. Two branches from _z_:
   - up to a **Quantizer** (codebook _q_);
   - right into the **Transformer context network** (label "12 layers, 768-d")
     → **context vectors _c_** (row of boxes).
4. Shade a couple of latents as **masked**; draw a **contrastive-loss** double-arrow
   between the masked context output _c_ and the quantized target _q_.

**Must match the text:** 20 ms frames, 12 layers / 768-d (base encoder). Redraw from
Baevski et al. 2020 — do not screenshot.

---

### B. `fig:self_attention_fusion` — self-attention fusion (Methodology)

Show the modalities **concatenated into one sequence**:
- A single row of tokens: a few **audio frames** (one colour) immediately followed by a
  few **text tokens** (another colour) — visually one bag.
- Feed into a **Self-attention** block; draw **all-to-all** arrows (or a small
  attention-matrix grid) so every position attends to every other, regardless of modality.
- Output: contextualised sequence → note "→ pool → head".
- In a caption corner, note the two variants: **multi-head** (several parallel heads) and
  **+ positional encoding** (sinusoid added to the inputs).

**Must match the text:** premise is "no architectural separation between the streams" —
keep audio + text visibly merged into one sequence.

---

### C. `fig:cross_attention_variants` — standard / gated / bidirectional (Methodology)

Three small panels side by side, each with two stacks (Audio, Text):
- **(a) Standard:** Text = **Query**, Audio = **Key/Value** → "text enriched by audio."
  Label Q/K/V explicitly.
- **(b) Gated:** same as (a) plus a **sigmoid gate** node (⊗) that scales the
  cross-attended output before it is added back to the query stream. The gate is the
  whole point — make it prominent.
- **(c) Bidirectional:** two cross-attentions at once (Audio→Text **and** Text→Audio),
  both enriched outputs **merged**. Label it "default in this thesis."

**Must match the text:** a/b/c exactly as described; bidirectional is the grid default.
This is the fiddliest — draw.io will beat TikZ here.

---

### D. `fig:mamba_fusion` — selective state-space fusion (Methodology)

The concatenated audio–text sequence fed left→right through a **recurrent state-space block**:
- Draw an **unrolled** chain: inputs _x₁…x_T_ along the bottom, a **hidden state _h_t_**
  box passing its arrow to the next step (h₁→h₂→…), outputs _y_t_ on top.
- Annotate the **input-dependent selection** gate on each step (what makes it "selective").
- Add a contrast note: "linear-time recurrence" vs attention's all-pairs comparison.

**Must match the text:** recurrent hidden state, input-dependent selection, linear vs
quadratic cost. Do **not** imply Mamba is fastest or best — it is an evaluated comparator,
not the recommended default.
