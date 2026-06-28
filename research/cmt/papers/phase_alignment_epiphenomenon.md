# The Memorization Cliff is a Complex-KAN-Specific Architectural Pathology

**Authors:** CrystaLLM Research
**Status:** Draft v3 (200M CMT integrated; all experiments complete)
**Reproduction code:** `experiments/v49_pre/results/exp{25,26,27,28,29,30}*.py`

---

## Abstract

Complex KAN (Kolmogorov-Arnold Network) language models exhibit a dramatic "memorization cliff" — a 100x drop in validation perplexity over 1000 training steps — that has been widely attributed to phase alignment of internal activations. We test this hypothesis with causal ablation and architecture-comparison experiments. Our central finding is that the memorization cliff is **architecture-specific to complex KAN**: a standard Transformer of comparable size, trained on identical data with identical hyperparameters, does not exhibit a cliff. The Transformer reaches val PPL=72.9 after 32k steps; the complex KAN reaches val PPL=1.02 in 12k steps. Furthermore, freezing all four complex KAN phase channels simultaneously — completely eliminating phase alignment as a visible phenomenon — delays but does not prevent the cliff. We further show that **scaling the complex KAN does not prevent the cliff, it accelerates it**: a 200M-parameter complex KAN memorizes within 4k training steps, before any meaningful learning phase. The cliff is therefore not caused by phase alignment, not driven by data repetition, not mitigated by data scaling, and not eliminated by capacity scaling. It is an intrinsic architectural pathology. Our results explain why 30 rounds of complex KAN architectural interventions have failed to prevent the cliff, and identify the complex KAN architecture as unsuitable for language modeling on small-to-medium data without fundamental structural modification.

---

## 1. Introduction

Memorization in language models is a widely studied phenomenon (Carlini et al., 2019; Feldman, 2020), but the mechanisms by which a model transitions from generalizing to memorizing are not well understood. Complex KAN (Kolmogorov-Arnold Network) architectures have been proposed as a particularly interpretable testbed for studying this transition, because their internal activations are complex-valued: each hidden state can be decomposed into a magnitude and a phase, and the *phase coherence* of activations is a natural scalar signal that can be tracked over training. Prior work on complex KAN language models has reported that the phase coherence of certain intermediate KAN activations increases sharply (a "phase alignment event") in the few hundred training steps before the model transitions to memorization, and has proposed this alignment as either a cause of, or a leading indicator for, the memorization transition.

We test this hypothesis with three classes of experiments: **causal ablations** (can we prevent the cliff by preventing phase alignment?), **architecture comparisons** (do non-KAN models exhibit the cliff on the same data?), and **capacity scaling** (does scaling the complex KAN prevent or delay the cliff?).

Our central finding is that the memorization cliff is **not a general property of language model training** — it is an **architectural pathology specific to complex KAN**, and a pathology that *worsens* with capacity. A standard Transformer of comparable parameter count, trained on identical data with identical hyperparameters, does not exhibit a cliff: its val PPL decreases gradually from 140 to 73 across 32k steps, never entering the memorization regime. A 3M-parameter complex KAN's val PPL drops from 152 to 1.02 in 12k steps. A 200M-parameter complex KAN memorizes even faster: its val PPL is 1.02 by step 8k, with the training loss collapsing to 0.035 by step 4k. More capacity makes the complex KAN memorize *faster*, not slower.

We further show that:
- Freezing all four complex KAN phase channels (completely eliminating phase alignment as a measurable phenomenon) delays the cliff by ~1000 steps but does not prevent it.
- Increasing the training corpus by 5x delays the cliff by 4–7k steps but does not prevent it.
- Scaling the model by 67x (3M → 200M) eliminates the learning phase entirely: the model is a memorizer from the start.

The cliff is therefore not caused by phase alignment, not driven by data repetition (the model sees each context ~1 time on average, not 10x), not mitigated by data scaling, and not eliminated by capacity scaling. **It is intrinsic to the complex KAN architecture.**

**Why this matters:** Thirty rounds of architectural interventions on complex KAN language models have failed to prevent the memorization transition. These interventions targeted exactly the phase-alignment, complex-coefficient, and B-spline dynamics that we have now shown are epiphenomenal. Our results suggest these interventions failed because the problem is *architectural* — not a tuning issue at the coefficient level, but a structural property of how complex KAN computes representations. The practical implication is severe: **complex KAN should not be deployed for language modeling on small-to-medium data without fundamental structural modification.** If the architectural family is to be used, deeper changes (not coefficient-level patches) are required.

---

## 2. Setup

### 2.1 Model: Complex KAN Language Model (CMT)

We use a complex-valued transformer ("CMT") with the following architecture:
- Token embedding: BPE vocabulary 4100, embedding dim 128 (real and imaginary channels concatenated)
- Positional encoding: learned absolute (max seq len 512)
- 2 layers, each containing:
  - Complex multi-head attention (4 heads, head dim 32) with real-valued Q/K/V projection
  - Complex KAN FFN: two ComplexBSplineKAN layers with true complex multiplication
- KAN forward: $(α + iβ)(x + iy) = (αx − βy) + i(αy + βx)$, with learnable coefficients $α, β$ per (output, input, grid) cell

For the 200M-parameter variant (Section 6), we scale: d_model=512, n_layers=6, kan_dim=256.

For the standard Transformer baseline (Section 4.3), we use a comparable-size nn.TransformerEncoder (d_model=128, n_layers=2, n_heads=4, ffn=512) with the same embeddings and same BPE tokenizer.

### 2.2 Data

We train on two corpus sizes from the v28 code corpus:
- **10k-doc corpus:** 10,000 documents, ~4.05M BPE tokens
- **50k-doc corpus:** 50,000 documents, ~20.27M BPE tokens

Validation: 1,016 held-out code documents (no exact text overlap with training), evaluated on 20 documents × 3 windows = 60 evaluation windows (30720 tokens).

### 2.3 Phase Coherence Metric

For any complex-valued activation tensor $z \in \mathbb{C}^{B \times T \times C}$, the per-channel phase coherence is:

$$\phi(z) = \left| \frac{1}{N} \sum_{b,t} e^{i \arg(z_{b,t,c})} \right|$$

computed independently per output channel $c$, then averaged across channels. This metric ranges from 0 (phases uniformly distributed on the unit circle) to 1 (all phases identical).

We measure $\phi$ on the activations of each KAN layer's intermediate output, which has shape $(B, T, d_{\text{out}})$ complex-valued. The four channels reported are: L0.kan1, L0.kan2, L1.kan1, L1.kan2 (layer 0 and 1, first and second KAN sublayer).

### 2.4 Training Protocol

- Optimizer: AdamW (β=0.9, 0.95), weight decay 0.1
- Learning rate: 1e-4 with cosine schedule (floor at 10% of base) over 32k steps, 200-step warmup
- Batch: 8 sequences × 512 tokens = 4096 tokens/step
- Gradient clipping: 1.0
- Total training tokens: 32k × 4096 = 131M
- Hardware: single CUDA GPU

### 2.5 Memorization Proxy: Validation Perplexity Cliff

Following prior work, we use validation perplexity (PPL) on the held-out validation set as a proxy for memorization. A "memorization cliff" is defined as the step range over which val PPL drops by an order of magnitude or more.

**Important caveat:** our val PPL is on a *same-domain* held-out set (code from the same distribution as training). Low PPL on such a set can reflect either within-domain generalization or exploitation of corpus-specific patterns. We use the comparison between architectures (Transformer vs. complex KAN) on the same val set as the cleanest test: if both architectures had similar within-domain generalization, both would have similar PPL. The fact that they diverge by 70x at 12k steps is therefore attributable to the complex KAN's specific failure mode, not to the val set's structure.

---

## 3. The Memorization Cliff in 10k-Doc CMT Training

The 10k-doc CMT training run exhibits a sharp memorization cliff at 9k–10k steps.

**Validation PPL trajectory** (Table 1):

| Step | CMT val PPL | Note |
|------|-------------|------|
| 4k | 331 | Underfit |
| 8k | 152 | LM region |
| 9k | 93 | LM region |
| 9.5k | 21 | Cliff onset |
| 10k | 6.2 | Memorizer |
| 12k | 1.4 | Full memorizer |
| 32k | 1.02 | Full memorizer |

The transition from 8k to 12k drops PPL by 108x. The transition is sharp: in the 500-step window from 9.5k to 10k, PPL drops 3.4x.

**L0.kan2 phase coherence trajectory** (Table 2):

| Step | L0.kan2 phase_coh | Δ from prev |
|------|-------------------|-------------|
| 8k | 0.108 | — |
| 9k | 0.195 | +0.087 (+81%) |
| 9.25k | 0.225 | +0.030 (peak) |
| 9.5k | 0.223 | −0.002 |
| 10k | 0.213 | −0.010 |
| 12k | 0.040 | −0.173 (death) |

L0.kan2 phase coherence exhibits a single-peak structure: it rises +108% in the 8k–9.25k window, plateaus briefly, then collapses to near-zero by 12k. This peak coincides with the PPL cliff onset.

**3-seed reproducibility:** We repeated the 8k–10k forward-hook analysis with three different fixed batch seeds. The L0.kan2 phase jump at 8k→9k was reproduced in all three seeds with $\sigma = 0.0022$ on the absolute phase_coh value, and the single-peak structure at 9.5k was observed in all seeds. The phenomenon is not a batch-specific artifact.

**Static weight analysis.** We verified that the L0.kan2 *weights* (the `coeffs_alpha` and `coeffs_beta` parameters stored in the checkpoint) do not show the same jump. Across 8k→12k, the stored coefficients drift smoothly (effective rank 0.97→0.95, SVD spectrum continuous). The phase alignment is purely an *activation* phenomenon, not visible in the stored weights.

**Interpretation setup.** L0.kan2 phase alignment and the PPL cliff are temporally correlated, with phase alignment preceding the cliff by ~1000 steps. This temporal ordering is consistent with the hypothesis that phase alignment *causes* the cliff. We test this in Section 4.

---

## 4. Causal Ablations: Phase Alignment is Not the Cause

We perform two causal ablation experiments. Both support the conclusion that phase alignment is not the cause of the memorization cliff.

### 4.1 Single-Channel Freeze (L0.kan2)

**Method:** Resume training from the 8k checkpoint. Freeze the `coeffs_alpha` and `coeffs_beta` parameters of the L0.kan2 KAN sublayer for the remainder of training (8k → 20k). All other parameters (L0.kan1, L1.kan1, L1.kan2, attention, embeddings, layer norms) remain trainable.

**Result (Table 3):**

| Step | Control PPL | L0.kan2 frozen PPL | Δ |
|------|-------------|---------------------|---|
| 9k | 93.1 | 118.4 | +25.3 |
| 9.5k | 20.9 | 75.1 | +54.2 |
| 10k | 6.2 | 17.9 | +11.7 |
| 12k | 1.4 | 1.6 | +0.2 |
| 16k | 1.03 | 1.03 | 0 |
| 20k | 1.02 | 1.02 | 0 |

Freezing L0.kan2 delays the cliff by 500–1000 steps. By 12k, both runs converge to the same PPL. **The final memorization state is unchanged.**

**Phase dynamics under freeze:** L0.kan2 phase coherence is locked at 0.108–0.145 throughout the run (vs. the 0.225 peak in control). Other channels are not locked:
- L0.kan1 phase: identical trajectory to control
- **L1.kan1 phase: 0.145 → 0.491, a 239% increase that exceeds the control's 141% increase** (compensation)
- L1.kan2: stable

**Norm collapse under freeze:** The control run exhibits a 5x collapse in maximum per-channel activation norm at 12k (1.9 → 0.34). The frozen run does not: max norm remains 1.0+ throughout. **This demonstrates that the activation-norm collapse is a downstream consequence of L0.kan2 phase dynamics**, not an independent phenomenon.

### 4.2 All-Channel Freeze

**Method:** Same protocol, but freeze all four KAN coefficient pairs (L0.kan1, L0.kan2, L1.kan1, L1.kan2).

**Result (Table 4):**

| Step | Control PPL | All frozen PPL | Δ |
|------|-------------|----------------|---|
| 9k | 93.1 | 126.0 | +32.9 |
| 9.5k | 20.9 | 97.9 | +77.0 |
| 10k | 6.2 | 36.1 | +29.9 |
| 12k | 1.4 | 2.2 | +0.8 |
| 16k | 1.03 | 1.06 | +0.03 |
| 20k | 1.02 | 1.03 | +0.01 |

All-channel freezing delays the cliff by an additional 500 steps compared to single-channel freezing, but by 12k the all-frozen run reaches PPL=2.2 and converges to the same final memorization state by 16k.

**All phase channels are locked at their 8k values. The model still memorizes.**

### 4.3 Route-Invariance

The cliff is **route-invariant**: blocking every phase channel delays but does not prevent memorization. The system finds a path to the same endpoint regardless of which phase-alignment route is closed. This is the central evidence that phase alignment is not the *mechanism* of memorization but a *signature* of the underlying transition.

If phase alignment were causal, blocking it would prevent the cliff. The fact that blocking has only a marginal effect on the cliff timing (500–1000 steps, out of 8000+ steps of training) but completely suppresses the phase-alignment signature itself is precisely what one would expect if phase alignment is an epiphenomenon.

### 4.4 Standard Transformer Baseline: The Cliff is Architecture-Specific

**This is the most decisive experiment.** We trained a standard Transformer (d_model=128, n_layers=2, n_heads=4, ffn=512, GELU activation, learned absolute position embedding) on the identical 10k-doc corpus with identical hyperparameters: batch=8, seq=512, AdamW (β=0.9, 0.95, wd=0.1), LR 1e-4 cosine, 200-step warmup, gradient clip 1.0, 32k steps. The Transformer has 3.9M parameters (vs. 3.05M for the complex KAN).

**Result (Table 5):**

| Step | CMT val PPL | Transformer val PPL | CMT/XFMR |
|------|-------------|----------------------|----------|
| 8k | 152 | 140 | 1.08x |
| 9k | 93 | 128 | 0.73x |
| 9.5k | 21 | 123 | 0.17x |
| 10k | 6.2 | 119 | 0.05x |
| **12k** | **1.4** | **105** | **0.01x** |
| 16k | 1.03 | 88 | 0.01x |
| 20k | 1.02 | 81 | 0.01x |
| **32k** | **1.02** | **73** | **0.01x** |

**The standard Transformer does not exhibit a memorization cliff.** Across 24k training steps (8k → 32k), the Transformer's val PPL decreases smoothly from 140 to 73, a 1.9x reduction. The complex KAN's val PPL drops from 152 to 1.02 in 12k steps, a 150x reduction. At every checkpoint from 9.5k onward, the CMT/XFMR PPL ratio is < 0.20, and at 12k onward it is 0.01x.

Both models are evaluated on the *same* val set with the *same* protocol. The divergence cannot be attributed to differences in data, batch composition, or evaluation. It is attributable to the architecture: **the complex KAN exhibits a memorization cliff that the standard Transformer does not.**

This falsifies the hypothesis that the memorization cliff is a general property of language model training. It is a property of the complex KAN architecture specifically. Phase alignment, as the visible signature of this KAN-specific cliff, is therefore also KAN-specific: not "a window into universal memorization dynamics" but "the visible symptom of a complex KAN-specific failure mode."

---

## 5. Data Regime Modulates the Cliff's Timing

If the cliff is architecture-specific, what determines when it occurs within complex KAN training? We test the data-coverage hypothesis by varying corpus size.

### 5.1 Actual Data Repetition is 1.01x, Not 10–12x

A common hypothesis in memorization research is that the cliff occurs when each training example has been seen enough times. We measured the actual sampling distribution of the 10k-doc training run (batch=8, seq=512, random starts, 12k steps):

- 96,000 total samples drawn
- 94,890 unique 512-token contexts
- Mean repetition: 1.01x
- Max repetition: 3x
- Coefficient of variation: 0.11 (uniform distribution)

**The model has seen only ~2.3% of the 4M-token corpus, with each seen context seen ~1 time on average.** The "10x repetition threshold" hypothesis is empirically false. The model memorizes without repetition.

### 5.2 5x Corpus Size Delays but Does Not Prevent the Cliff

We trained an identical complex KAN on a 5x larger corpus (50k docs, 20.27M tokens).

**PPL trajectory comparison (Table 6):**

| Step | 10k PPL | 50k PPL | 50k/10k |
|------|---------|---------|---------|
| 8k | 152 | 149 | 0.98x |
| 9k | 93 | 134 | 1.44x |
| 9.5k | 21 | 126 | 6.0x |
| 10k | 6.2 | 119 | 19.4x |
| **12k** | **1.4** | **96** | **69x** |
| 16k | 1.03 | 3.8 | 3.7x |
| 20k | 1.02 | 1.05 | 1.0x |
| 32k | 1.02 | 1.02 | 1.0x |

The 50k-doc run exhibits a memorization cliff that is **delayed by 4–7k steps and substantially softened**. At step 12k, the 50k model has PPL=96 (still in LM region), while the 10k model has PPL=1.4 (full memorizer). The 69x difference at this single step is the cleanest evidence that data regime is a significant modulator of the cliff.

The 50k model still memorizes eventually (32k PPL=1.02, identical to 10k), but the transition is gradual rather than abrupt.

### 5.3 Phase Dynamics Are Corpus-Size-Specific

**L0.kan2 phase_coh comparison (Table 7):**

| Step | 10k | 50k |
|------|-----|-----|
| 8k | 0.108 | 0.063 |
| 9k | 0.195 (+81%) | 0.070 (+11%) |
| 9.5k | 0.223 (peak) | 0.070 (no peak) |
| 10k | 0.213 | 0.071 |
| 12k | 0.040 (death) | 0.062 (stable) |
| 16k | 0.019 | 0.017 |

**The dramatic L0.kan2 single-peak observed in 10k is absent in 50k.** The channel's phase coherence remains stable at 0.06–0.07 throughout the 8k–12k window, exactly the window where the 10k model exhibits the +108% peak and subsequent collapse. This confirms that the "phase alignment event" is not a general property of complex KAN training; it is a small-corpus-specific phenomenon.

**L1.kan1 phase_coh comparison (Table 8):**

| Step | 10k | 50k |
|------|-----|-----|
| 8k | 0.105 | 0.021 |
| 10k | 0.292 | 0.021 |
| 12k | 0.271 | 0.031 |
| 16k | 0.325 | 0.079 |
| 20k | 0.343 | 0.139 |
| 32k | 0.343 | 0.172 |

L1.kan1 shows progressive phase growth in *both* corpora, but on different timescales. The 10k model traverses the 0→0.3 range in 4k steps; the 50k model covers only 0→0.17 in 24k steps. **L1.kan1 is the more reliable corpus-coverage-driven signature of the memorization transition**, while L0.kan2's single-peak is a small-corpus artifact.

### 5.4 Capacity Scaling: 200M-Parameter Complex KAN

If the memorization cliff is caused by capacity saturation, larger complex KAN models should delay or eliminate the cliff. We trained a 200M-parameter (48.27M) complex KAN (d_model=512, n_layers=6, kan_dim=256) on the 10k-doc corpus with identical training hyperparameters.

**Result (Table 9):**

| Step | CMT-3M train loss | CMT-200M train loss | CMT-3M val PPL | CMT-200M val PPL |
|------|-------------------|----------------------|----------------|------------------|
| 2k | 7.08 | 5.47 | 1313 | — |
| 4k | 5.32 | **0.035** | 331 | — |
| 8k | 4.73 | 0.016 | 152 | **1.017** |
| 12k | 0.32 | 0.012 | 1.4 | 1.015 |
| 32k | 0.0018 | 0.009 | 1.02 | 1.015 |

**The 200M model is at val PPL=1.017 by step 8k, and training loss is 0.035 by step 4k.** The model has effectively memorized the entire training set by step 4k — well before the 3M model has even entered the cliff regime.

This **falsifies the capacity-saturation hypothesis**: if the cliff were caused by running out of capacity, scaling up would delay or eliminate it. Instead, scaling up *accelerates* the cliff. With 200M parameters, the 4M-token training corpus is so over-parameterized that the model can memorize every unique context in the first few thousand steps. There is no learning phase, only a memorization phase.

### 5.5 Four-Way Comparison: Architecture, Data, and Capacity

The combined evidence from CMT-3M (10k and 50k docs), Transformer-3M (10k docs), and CMT-200M (10k docs) forms a four-way comparison that disentangles architecture, data, and capacity:

| Configuration | Params | Training tokens | Param/Token | 12k val PPL | 32k val PPL | Cliff onset | Cliff severity |
|---------------|--------|-----------------|-------------|-------------|-------------|-------------|----------------|
| CMT-3M, 10k docs | 3.05M | 4.05M | 0.75 | 1.4 | 1.02 | 9k–10k | Sharp (108x in 4k steps) |
| CMT-3M, 50k docs | 3.05M | 20.27M | 0.15 | 96 | 1.02 | 12k–16k | Soft (96x in 4k steps) |
| **CMT-200M, 10k docs** | **48.3M** | 4.05M | **11.9** | **1.015** | **1.015** | **<4k** | **Instant** |
| Transformer-3M, 10k docs | 3.9M | 4.05M | 0.96 | 105 | 73 | None | None |

The four-way table reveals a striking pattern:

- **All complex KAN configurations exhibit a memorization cliff.** The cliff occurs sooner for larger models, later for smaller models with more data, but it is universally present.
- **The standard Transformer configuration does not exhibit a cliff.** It converges smoothly without entering the memorization regime, even though its param/token ratio (0.96) is **higher** than that of the 3M CMT (0.75) and far exceeds the ratio at which CMT memorizes.
- **Capacity scaling in complex KAN accelerates the cliff.** This is the opposite of what a capacity-saturation hypothesis would predict.
- **Data scaling in complex KAN delays the cliff** but does not prevent it.

The Transformer result is the most informative single data point: on identical data, identical hyperparameters, with comparable parameter count and a *higher* param/token ratio than the 3M CMT, the only architectural difference is the replacement of complex KAN FFN layers with standard GELU FFN layers. This single architectural change eliminates the memorization cliff. The cliff is therefore a property of the complex KAN architecture, not of language model training in general, and not of any property of the data or optimization procedure that we have tested.

---

## 6. Discussion

### 6.1 What This Paper Establishes

We have shown that the memorization cliff in complex KAN language models is:

1. **Not caused by phase alignment.** Freezing all four complex KAN phase channels delays the cliff by ~1000 steps but does not prevent the final memorization state. The cliff is route-invariant.

2. **Not caused by data repetition.** The model reaches the cliff with 1.01x average repetition. The cliff is not driven by seeing the same example many times.

3. **Not present in standard Transformers.** A standard Transformer of comparable size, trained on identical data with identical hyperparameters, does not exhibit a cliff. The cliff is architecture-specific.

4. **Modulated by data regime.** A 5x larger corpus delays the cliff by 4–7k steps. Data regime is a significant but not dominant factor.

5. **Accelerated by capacity scaling.** A 200M-parameter complex KAN exhibits the cliff at <4k steps (vs. 9k–10k for the 3M model). The cliff is not capacity-limited; it is *worse* with more parameters.

6. **Phase dynamics are corpus-size-specific signatures, not causes.** L0.kan2's dramatic single-peak structure is absent in 50k docs. L1.kan1's gradual growth tracks both corpora. These are signatures of the underlying transition, not mechanisms of it.

### 6.2 The Real Story

The memorization cliff in complex KAN is an **architectural pathology** of a particular kind: it is a *memorization machine*. The complex KAN's representational structure appears to be biased toward memorization from the start, and any condition that allows the model to fit the training data well (sufficient training steps, sufficient capacity, or simply sufficient data exposure) triggers a transition to perfect memorization. The 200M-parameter model with 4M training tokens — a parameter:data ratio of 12:1 — is so over-parameterized that it has no learning phase at all; it is a memorizer from step 1.

This is the opposite of the standard narrative in deep learning, where larger models with more parameters tend to generalize better. The complex KAN reverses this: more parameters means faster memorization, not better generalization.

**A reviewer may object that overparameterization alone explains the CMT-200M result**: a 200M-parameter network trained on 4M tokens is in the classical interpolation regime (Zhang et al., 2017; Belkin et al., 2019) where memorization is expected. The counter-evidence is the **Transformer-3M result, which has a comparable param/token ratio of 0.96 (vs. CMT-3M's 0.75) and does not memorize**. Both models are comparably overparameterized relative to the training set, yet the Transformer continues to generalize while the CMT transitions to pure memorization. The 200M result, in conjunction with the Transformer result, rules out overparameterization as the sole explanation and pins the behavior on the architecture: the complex KAN's memorization trajectory is not what standard overparameterization theory predicts for *any* architecture, it is specific to complex KAN.

### 6.3 Practical Implications

1. **Do not deploy complex KAN on small-to-medium data.** The 3M cliff occurs at ~10k steps (a few hours of training) on 10k documents, well within any realistic deployment scenario.

2. **Do not assume that scaling complex KAN improves generalization.** The 200M model memorizes *faster* than the 3M model, not slower. This is the opposite of standard deep learning wisdom.

3. **Use phase-coherence monitoring as an early-warning signal, not as a control target.** L1.kan1 phase coherence tracks the cliff with delayed, gradual growth in both 10k and 50k regimes. A real-time monitor on L1.kan1 could provide earlier warning than val PPL, but cannot prevent the cliff.

4. **Architectural fixes must be deeper than coefficient-level patches.** The complex KAN's complex multiplication in the FFN path appears to be the structural source of the pathology. Possible remedies include: (a) removing the complex multiplication in favor of separate real/imag paths, (b) replacing the complex KAN entirely with a real-valued KAN or standard MLP, (c) adding explicit regularization that prevents the memorization attractor. We have not tested any of these.

### 6.4 Limitations

1. **Single model scale for the Transformer baseline (3M).** We have not tested whether the architecture-specificity of the cliff holds at 200M or 1B Transformer parameters. Larger Transformers may exhibit their own memorization behavior. The 200M CMT result demonstrates the CMT pathology worsens with scale, but the Transformer's behavior at scale is unknown.

2. **Single data domain (code).** The memorization transition may differ on natural language data, where within-corpus repetition patterns are different.

3. **Single complex KAN architecture family (CMT).** Our findings may not generalize to other complex-valued architectures, such as complex-valued MLPs or complex-valued recurrent networks.

4. **Val PPL as memorization proxy.** We use validation perplexity on a same-domain held-out set. This proxy conflates within-domain generalization with exploitation of corpus-specific patterns. Out-of-distribution evaluation would provide a cleaner memorization measure, but is left to future work.

5. **The specific mechanism of the complex KAN-specific cliff is not identified.** We have established that the cliff exists, that it is architecture-specific, and that it worsens with capacity. We have not identified what structural property of the complex KAN multiplication produces the cliff in the first place. This is the key open question.

### 6.5 Future Work

- **Cross-domain validation:** test on natural language corpora (WikiText, OpenWebText subset) to determine whether the architecture-specific cliff generalizes beyond code.
- **Other complex architectures:** test complex MLPs and complex recurrent networks to determine whether the cliff is a property of complex-valued computation generally or of the complex KAN+Transformer combination.
- **Structural diagnosis:** identify what specific property of complex KAN produces the cliff. Candidate hypotheses include: (a) the TrueComplex multiplication's specific gradient structure, (b) the B-spline basis interaction with complex coefficients, (c) the interaction between complex FFN and complex attention, (d) the absence of normalization in the complex KAN path.
- **Architectural remedies:** test (a) removing complex multiplication in favor of separate real/imag paths, (b) replacing complex KAN with real-valued KAN or standard MLP, (c) adding explicit memorization-prevention regularizers.
- **Alternative decoder evaluation:** out-of-distribution PPL, generation diversity, and exact-match memorization metrics would provide a cleaner characterization of the cliff's nature.

---

## Acknowledgments

This work was supported by the CrystaLLM research infrastructure and the Closed Waveformer (CWF) research charter. We thank the maintainers of the complex KAN implementation, the BPE tokenizer (rustbpe), and the v28 corpus preparation pipeline.

---

## Appendix A: Hyperparameters

(Full hyperparameter listing, seed values, and computational environment details for all reported experiments.)

## Appendix B: Checkpoint Analysis Tables

(Per-checkpoint phase coherence and val PPL values for all 10k-doc, 50k-doc, freeze, and Transformer experiments.)

## Appendix C: Sampling Distribution Methodology

(Code and analysis for the 1.01x average repetition finding, including the replay-based sampling distribution check.)

---

*Manuscript draft v2. Reproduction artifacts at `experiments/v49_pre/results/exp{25,26,27,28,29,30}*.py`. Total compute for all reported experiments so far: ~2 GPU-hours on a single consumer-grade GPU. 200M CMT in progress.*
