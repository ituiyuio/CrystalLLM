# Closed Waveformer (CWF): A Research Manifesto

**Version**: 0.1 (2026-06-23)
**Status**: Charter (no engineering commitment)
**Branch**: `cwf-manifesto`
**Inherits from**: 30 rounds of CMT experiments + Soft-Exp knife-4 + exp30 failure analysis

---

## 1. Motivation: Why AR + Discrete Tokens is Fundamentally Limited

The current language modeling paradigm has a structural defect: **discrete token input/output but continuous internal representations**. This defect manifests in three empirical pathologies:

### 1.1 Exposure Bias (exp29 evidence)
- V49 1.2B baseline: Teacher Forcing PPL = 3.26
- Autoregressive (argmax) PPL = **64.74** (19.88x exposure bias)
- Soft-Exp feedback PPL = 33.27 (10.22x exposure bias, +48.6% over argmax)
- **Implication**: Discrete argmax at inference creates a distributional mismatch that no amount of training data eliminates.

### 1.2 Semantic Discontinuity (CMT exp 25 evidence)
- Tokens like "跑" and "慢跑" live in discrete bins separated by arbitrary boundaries
- A continuous semantic interpolation ("running slowly") has no natural representation
- The model's "knowledge" is quantized at token boundaries, then forced back into continuous embeddings via arbitrary embedding matrices

### 1.3 Closure Failure (exp30 evidence)
- Soft-Exp at inference only: works (+48.6%)
- Soft-Exp at training too: catastrophic failure (argmax PPL 8554 vs 64.74 baseline, 132x worse)
- **Implication**: Training-time distribution shift breaks the model's adaptation to discrete-token input distribution

These three pathologies share a common root cause: **the discrete token interface leaks through every continuous-internal attempt**.

---

## 2. Closure Theorem (Conjecture Form)

### 2.1 Statement

> **Closure Conjecture (C0)**: Let $\psi \in \mathbb{D}^d \subset \mathbb{C}^d$ be a wave function state in the open unit disk. Let $\mathcal{F}_l: \mathbb{D}^d \to \mathbb{D}^d$ be any of the 6 component operators (defined in §3). Then:
> 1. $\mathcal{F}_l$ is well-defined (maps into the open disk)
> 2. $\mathcal{F}_l$ is a smooth map (infinitely differentiable)
> 3. The composition $\mathcal{F}_N \circ \cdots \circ \mathcal{F}_1$ preserves $\|\psi\| < 1$
> 4. The gradient $\nabla_\theta \mathcal{L}$ exists and is bounded for any reasonable loss $\mathcal{L}$

If this conjecture holds, we have a well-defined continuous-time language model that:
- Has no discrete token interface internally
- Can be trained via standard gradient methods
- Has well-defined information flow (no bottleneck)

### 2.2 Why This is Non-Trivial

Standard Transformer components (LayerNorm, Softmax, ReLU, etc.) all map $\mathbb{R}^d$ to $\mathbb{R}^d$ but lack a closure structure. CWF requires **every** component to be either:
- An isometry of $\mathbb{D}^d$ (preserves $\|\psi\|$)
- A contraction (decreases $\|\psi\|$)

These are **strict** constraints. Most existing deep learning operators violate them (ReLU has discontinuous gradient at 0; LayerNorm can have norm explosion; Softmax has finite-domain issues in complex domain).

---

## 3. The 6 Closed Components

Each component must satisfy the closure property of §2.1.

### 3.1 Component 1: Input Encoding $\mathcal{E}: \mathcal{X} \to \mathbb{D}^d$

**Goal**: Map raw continuous signals (bytes, spectrograms, patches) into the open unit disk in $\mathbb{C}^d$.

**Candidate designs**:
- **(A) Frequency-domain embedding**: $\mathcal{E}(x) = \text{FFT}(x) / \|x\|_2$ (unit norm after normalization)
  - Pros: Natural for periodic signals; clean closure if $\|x\|_2 > 0$
  - Cons: Loses local information
- **(B) Sparse dictionary encoding**: $\mathcal{E}(x) = D \alpha$ where $D \in \mathbb{C}^{d \times k}$ is a dictionary, $\alpha$ are sparse codes
  - Pros: Composable; interpretable
  - Cons: $\ell_2$ normalization needed to maintain closure
- **(C) Neural encoder with closure loss**: Train any encoder with regularization $\|\mathcal{E}(x)\|^2 < 1 - \epsilon$
  - Pros: Flexible
  - Cons: Closure is a soft constraint, not structural

**First prototype**: Use **(A) FFT** for synthetic periodic data. Defer (B) and (C) to later phases.

### 3.2 Component 2: Lie Group Position Rotation $\mathcal{P}_\theta: \mathbb{D}^d \to \mathbb{D}^d$

**Goal**: Encode position via Lie group element, preserving closure.

**Design**: 
$$\mathcal{P}_\theta(\psi) = \exp(A(\psi)) \cdot \psi$$
where $A(\psi) \in \mathfrak{so}(d)$ is a skew-symmetric matrix (context-dependent) and $\exp$ is the matrix exponential mapping to $SO(d)$.

**Closure property**: 
- $\exp(A) \in SO(d)$ is an isometry of $\mathbb{R}^d$ (and $\mathbb{C}^d$ as real 2d vector space)
- Therefore $\|\mathcal{P}_\theta(\psi)\| = \|\psi\|$
- Strictly preserves the norm → trivially closed

**Engineering note**: Matrix exponential is $O(d^3)$. Use Cayley transform for efficiency:
$$R = (I + A/2)(I - A/2)^{-1}$$
This is also in $SO(d)$ and requires only one matrix inversion.

### 3.3 Component 3: Complex-Valued Attention $\mathcal{A}_\theta: \mathbb{D}^d \to \mathbb{D}^d$

**Goal**: Wave interference in Hilbert space, replacing softmax with unitary-weighted sum.

**Design**:
$$\mathcal{A}_\theta(\psi)_i = \sum_j \frac{\langle \phi_i, \psi_j \rangle}{|\langle \phi_i, \psi_j \rangle| + \epsilon} \cdot V_j$$
where $\phi_i, V_j \in \mathbb{C}^d$ are complex-valued projections and $\langle \cdot, \cdot \rangle$ is the Hermitian inner product.

**Closure property**:
- Attention weights $\frac{\langle \phi_i, \psi_j \rangle}{|\langle \phi_i, \psi_j \rangle| + \epsilon}$ are complex unit-magnitude (after normalization)
- Output is a weighted sum: $\|\sum_j w_j V_j\| \leq \sum_j \|w_j\| \|V_j\| = \sum_j \|V_j\|$
- Closure requires **explicit renormalization** post-attention

**Critical difference from Transformer**: No softmax. The "normalization" is unit-magnitude phase extraction, not exponential. This preserves phase information (which softmax destroys via $\exp$ dominance).

### 3.4 Component 4: Complex KAN-FFN $\mathcal{K}_\theta: \mathbb{D}^d \to \mathbb{D}^d$

**Goal**: Continuous function on complex manifold, not discrete ReLU partition.

**Design**:
$$\mathcal{K}_\theta(\psi) = \sum_{k=1}^{K} \alpha_k \cdot B_k(\psi)$$
where $B_k$ are B-spline basis functions on $\mathbb{C}^d$ and $\alpha_k \in \mathbb{C}^d$ are learnable coefficients.

**Closure property**:
- B-spline basis is bounded: $|B_k(\psi)| \leq M$ for some $M$ depending on control points
- $\mathcal{K}_\theta(\psi)$ is bounded if $\sum_k |\alpha_k| \cdot M$ is bounded
- Need explicit spectral norm constraint on $\alpha_k$ during training

**Alternative simpler design**: Use Siren activation (sinusoidal) instead of B-spline for first prototype:
$$\mathcal{K}_\theta(\psi) = W_2 \cdot \sin(W_1 \psi + b_1) + b_2$$
with explicit norm constraint.

### 3.5 Component 5: Born-Stable Normalization $\mathcal{N}: \mathbb{C}^d \to \mathbb{D}^d$

**Goal**: Project any state back to the open unit disk after any operation that might exceed $\|\psi\| = 1$.

**Design**:
$$\mathcal{N}(\psi) = \frac{\psi}{\max(\|\psi\|, 1 - \epsilon)}$$

This is a hard projection, not a soft rescaling. It guarantees $\|\mathcal{N}(\psi)\| \leq 1 - \epsilon < 1$.

**Why not LayerNorm?** Standard LayerNorm is affine, not projective. It can scale arbitrarily, violating closure. The Born rule is fundamentally projective.

### 3.6 Component 6: Output via Born Rule (ONLY at final layer)

**Goal**: Convert internal wave state to discrete output **only at the final measurement**, never feedback into the loop.

**Design**:
$$P(v) = \frac{|\langle \Phi_v, \psi_{\text{final}} \rangle|^2}{\sum_v |\langle \Phi_v, \psi_{\text{final}} \rangle|^2}$$

where $\Phi_v \in \mathbb{C}^d$ are vocabulary basis vectors (or codebook entries).

**Critical property**: The output $P(v)$ is a probability distribution, used only for **loss computation and human display**. The next training step uses $\psi_{\text{final}}$ directly, **not** $\arg\max P(v)$ or any discrete sample.

---

## 4. Three Critical Conjectures

Before any large-scale engineering, these three must be addressed (proof or counter-example):

### C1: Convergence
**Statement**: Gradient descent on CWF with any reasonable loss converges to a local optimum.
**Risk**: Complex Wirtinger derivatives have non-trivial interaction with non-convex loss landscapes.
**Test**: Run CWF on synthetic task; monitor loss landscape.

### C2: Isometric Embedding
**Statement**: There exists a low-distortion encoder $\mathcal{E}: \text{text} \to \mathbb{D}^d$ such that semantic similarity in text space is preserved by inner product in $\mathbb{D}^d$.
**Risk**: Text is fundamentally discrete. The encoder may destroy information at the discrete-continuous interface.
**Test**: Compare $\langle \mathcal{E}(s_1), \mathcal{E}(s_2) \rangle$ to human-rated similarity for sentence pairs.

### C3: Gradient Stability
**Statement**: Wirtinger gradient of CWF loss is numerically stable in FP32 (no explosion, no vanishing) for sequences up to length 1024.
**Risk**: Complex gradients can have phase cancellations that lead to silent underflow.
**Test**: Train CWF on long sequences; monitor gradient norms and individual parameter trajectories.

**None of these has been verified**. The first prototype must address all three simultaneously.

---

## 5. CMT 25-Round Autopsy: Lessons From Failures

A complete accounting of what went wrong in 30 rounds of CMT (Continuous Manifold Transformer) experiments.

### 5.1 What was tried

| Knives | Components | Rounds | Outcome |
|---|---|---|---|
| Knife 1 | KAN + 复数 FFN | Exp 6-15 | FAIL (memorizer @ 12k) |
| Knife 2 | 复数 Attention | Exp 6-15 | FAIL (combined with 1) |
| Knife 3 | LieRE 流形 PE | Exp 6-15 | FAIL (combined with 1) |
| Knife 4 | Soft-Exp 解码 | Exp 26-29 | **PASS** (+48.6% inference) |
| Knife 5a | 渐进式 (训练 Soft-Exp) | Exp 30 | **FAIL** (132x worse) |

### 5.2 Why Knife 4 succeeded but the rest failed

Knife 4 (Soft-Exp at inference only) succeeded because:
- Training distribution is **unchanged** (still standard TF)
- Inference distribution is shifted to a "safer" zone (continuous expected embedding)
- The shift exploits existing model robustness, not creating new requirements

Knives 1-3 failed because:
- They modified the **training distribution** to include continuous-flavored inputs
- The model overfits to GT inputs (its anchor) and becomes fragile on self-generated inputs
- This is exactly the exp30 pathology scaled up

### 5.3 The fundamental lesson

**Closure is all-or-nothing**. Half-closed architectures are **strictly worse** than fully closed (or fully open) ones because they introduce additional distribution drift that the optimizer cannot resolve.

### 5.4 Forbidden Patterns (for future CWF work)

Based on 30 rounds of failures, these patterns are explicitly forbidden:

1. ❌ **Mixing real and complex in same model**: causes dimension collapse and phase ambiguity
2. ❌ **Using Teacher Forcing on wave models**: breaks the ODE interpretation, introduces exposure bias
3. ❌ **Discrete argmax as next input**: reintroduces the discrete-continuous bottleneck
4. ❌ **Without explicit closure verification (norm monitoring)**: silent drift can corrupt training
5. ❌ **KAN B-spline without spectral bound**: spline coefficients can blow up
6. ❌ **Cayley/exp on large matrices without numerical safeguards**: matrix inversion is fragile
7. ❌ **Mixing multiple wave components (KAN + 复数 + Lie)**: combinatorially more failure modes

---

## 6. Minimal Viable Prototype

### 6.1 Scope

A single-block CWF model on synthetic harmonic sequence task.

**Why synthetic first**: Natural language has too many confounding factors (tokenization, vocabulary coverage, data noise). A synthetic task isolates the architecture's closure properties from linguistic concerns.

### 6.2 Architecture (single block)

```
Input (continuous, ℝ^S): y_1, y_2, ..., y_S  (sin(ωt + φ))
       ↓ FFT + norm
ψ ∈ 𝔻^d  (d=64, single frequency)
       ↓ Lie rotation
ψ ← exp(A(ψ)) · ψ  (preserves ‖ψ‖)
       ↓ Complex attention
ψ ← unit-phase-weighted sum  (preserves ‖ψ‖ < 1 by design)
       ↓ Complex KAN-FFN (Siren)
ψ ← W_2 · sin(W_1 ψ + b)  (with norm clipping)
       ↓ Born-stable norm
ψ ← ψ / max(‖ψ‖, 1-ε)
       ↓ Born rule (only at end)
P(ŷ_{S+1}) = |⟨Φ, ψ⟩|² / Σ |⟨Φ, ψ⟩|²
```

### 6.3 Training

- **Loss**: MSE on continuous output (not cross-entropy)
- **Optimizer**: AdamW with separate learning rate for real/imaginary parts
- **Batch size**: 32
- **Sequence length**: 64-256
- **Epochs**: 100 (synthetic, can afford long training)
- **Hardware**: Single GPU (5080/5090 with 24GB+)

### 6.4 Validation Metrics

1. **Closure invariant**: Monitor $\|\psi\|$ at each layer; verify always < 1
2. **Gradient norm**: Monitor $\|\nabla_\theta \mathcal{L}\|$; verify no explosion
3. **Task loss**: MSE on held-out synthetic data
4. **Comparison**: Same-size AR Transformer baseline (real-valued)

### 6.5 Success Criteria

- **PASS**: CWF MSE ≤ 0.5x AR Transformer MSE on synthetic harmonic task
- **PARTIAL**: CWF MSE within 1.0-2.0x of AR
- **FAIL**: CWF MSE > 2x AR, or closure invariant violated, or gradient explosion

---

## 7. Synthetic Task Roadmap

### 7.1 Phase 0: Single Harmonic (1 week)
- Task: predict $y_{t+1} = \sin(\omega t + \phi)$ given $y_1, ..., y_t$
- Single frequency, single phase
- Validates: encoding (FFT works?), single-block closure, basic dynamics

### 7.2 Phase 1: Multi-Harmonic Superposition (2 weeks)
- Task: predict $y_{t+1} = \sum_{k=1}^{K} a_k \sin(\omega_k t + \phi_k)$
- Multiple frequencies, random amplitudes
- Validates: superposition principle, complex attention's interference model

### 7.3 Phase 2: Chaotic / Lorenz System (3 weeks)
- Task: predict next state of Lorenz attractor
- Continuous dynamics, deterministic but chaotic
- Validates: long-term stability, ODE interpretation

### 7.4 Phase 3: Speech Spectrogram (4 weeks, IF Phases 0-2 pass)
- Task: predict next spectrogram frame from audio
- Real data, but still continuous
- Validates: encoder design on real-world continuous signals

### 7.5 Phase 4: Natural Language (months, IF Phase 3 passes)
- Use byte-level encoding (no tokenizer)
- Train on text directly
- **Hard gate**: Phase 3 must demonstrate CWF beats AR by 2x on perplexity-equiv metric

---

## 8. Why This Might Fail (Pre-Mortem)

### 8.1 Diffusion LM Precedent
- 5 years of diffusion language model research (Diffusion-LM, SEDD, MDLM, Plaid)
- **All** underperform GPT-style AR by 5-30% on perplexity at 1B scale
- This suggests **language may be fundamentally discrete** in ways continuous models cannot capture

### 8.2 Engineering Cost
- Custom CUDA kernels for complex GEMM, complex FFT, Lie group operations
- Estimated 1-2 person-years to match Transformer efficiency
- No guarantee the investment pays off

### 8.3 Information-Theoretic Concerns
- Continuous representation may not preserve enough information about discrete text
- The encoder $\mathcal{E}$ may need to be lossless, contradicting the closure constraint
- Possible theorem: no low-distortion encoding exists from text to $\mathbb{D}^d$ with $\|\psi\| < 1$

### 8.4 Verification Asymmetry
- AR Transformer: works immediately on any text task
- CWF: must be proven on synthetic tasks first, then scaled
- The verification burden is **strictly higher** for CWF

---

## 9. Timeline (Realistic)

| Phase | Duration | Personnel | Outcome |
|---|---|---|---|
| Manifesto (this doc) | 1 week | 1 person | Done (v0.1) |
| Phase 0 prototype | 1 week | 1 person | Single-block CWF on single harmonic |
| Phase 1 multi-harmonic | 2 weeks | 1 person | Validate interference model |
| Phase 2 chaotic | 3 weeks | 1 person | Validate ODE dynamics |
| **Total (Phase 0-2)** | **6-8 weeks** | **1 person** | **GO/NO-GO decision** |
| Phase 3 speech | 4 weeks | 1-2 people | Real continuous data |
| Phase 4 language | 6+ months | 2-4 people | Publication-quality result |

**GO/NO-GO gate at end of Phase 2**: If CWF does not beat AR Transformer by 2x on synthetic chaos, halt the project. The mathematics suggests this is unlikely; the empirical evidence will confirm.

---

## 10. Open Questions

1. Can B-splines on $\mathbb{C}^d$ be made numerically stable? Or must we use simpler activations (sin)?
2. Does the closure property hold under minibatch stochastic gradient noise? Or do we need batch norm?
3. Is there a way to do supervised training without ever leaving the continuous representation?
4. What is the natural "decoder" if we want to output continuous signals (e.g., audio) directly?
5. Can attention be replaced by a simpler interference operator (e.g., holography)?

---

## 11. Status

This manifesto is **version 0.1**, sufficient to begin prototype development. Updates will be tracked via:
- Code: `research/cwf/prototype/`
- Tasks: `research/cwf/tasks/`
- Experiments: `research/cwf/experiments/`
- Docs: `research/cwf/docs/`

**No engineering commitment is made here**. If Phase 0-2 do not show CWF promise, this charter is archived.

---

## Appendix A: Notation

- $\mathbb{C}^d$: $d$-dimensional complex vector space
- $\mathbb{D}^d = \{z \in \mathbb{C}^d : \|z\| < 1\}$: open unit disk (hyperbolic space)
- $\mathfrak{so}(d)$: Lie algebra of skew-symmetric $d \times d$ matrices
- $SO(d)$: special orthogonal group (rotations)
- $\langle \cdot, \cdot \rangle$: Hermitian inner product
- $\|\cdot\|$: $\ell_2$ norm

## Appendix B: References (To Be Filled)

- Bengio et al. 2015 (Scheduled Sampling)
- Hinton et al. 2015 (Knowledge Distillation)
- Lie group foundations (Hall 2015)
- Diffusion LM literature (Austin 2024, Lou 2024)
- CMT experimental autopsy (memory/2026-06-23-v50-final-decision.md)
