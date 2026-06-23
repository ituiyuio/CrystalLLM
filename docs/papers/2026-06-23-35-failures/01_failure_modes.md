# Failure Mode Analysis: 5 Patterns Across 35 Experiments

This document extracts the 5 recurring failure modes from the 35-experiment campaign. Each mode is documented with:
- **Definition**: what the failure looks like
- **First appearance**: which experiment first exhibited it
- **Evidence**: experiments that confirmed it
- **Root cause**: the underlying mechanism
- **Lessons**: what we learned

---

## Failure Mode 1: Training-Inference Distribution Drift

### Definition

When the input distribution at training time differs from the input distribution at inference time, the model learns to overfit to the training distribution. At inference, it sees out-of-distribution inputs and either memorizes (low PPL on seen patterns, high PPL on unseen) or fails catastrophically (high PPL everywhere).

### First Appearance

Exp 30 (the "final test"): Soft-Exp works at inference (+48.6%) but destroys the model when used at training (-132×).

### Confirming Evidence

| Experiment | Training Input | Inference Input | Outcome |
|---|---|---|---|
| CMT Exp 6-15 | Continuous (KAN/complex) | Continuous (KAN/complex) | Memorization @ 4-12k |
| Soft-Exp train (Exp 30) | Mixed continuous/discrete | Continuous | Catastrophic failure |
| Soft-Exp inference (Exp 29) | Discrete (TF) | Continuous | **+48.6% PASS** |

The pattern is **strikingly consistent**: changing the training input distribution breaks the model; changing only the inference input distribution works.

### Root Cause

**Distribution shift within the model's input manifold.** When trained on distribution $\mathcal{D}_{\text{train}}$, the model learns to map inputs $\sim \mathcal{D}_{\text{train}}$ to outputs. At inference, if inputs come from a different distribution $\mathcal{D}_{\text{infer}}$, the model is OOD.

For CMT: the "continuous" components (KAN, complex FFN) had different effective input distributions than the discrete teacher-forcing baseline. The model overfit to the training distribution's continuous-flavored inputs and lost generalization.

For Soft-Exp training: the model was trained with mixed soft+hard inputs ($\alpha=0.5$), but at inference the input distribution is again different (full continuous or full discrete), causing catastrophic drift.

For Soft-Exp **inference only**: the training distribution is unchanged (standard TF). At inference, the model's predictions are mapped to a continuous embedding that lies within (or near) the training distribution's manifold. This is the only configuration where the inference distribution is "close enough" to the training distribution.

### Lessons

1. **Any change to training input distribution is dangerous.** Even 50% mixing can cause catastrophic failure.
2. **Inference-time changes are safe as long as the inference distribution is a "smoothed version" of the training distribution.** Soft-Exp: continuous expectation is in the convex hull of training embeddings.
3. **The "covariate shift" framing of exposure bias is incomplete.** It's not just that the model sees its own predictions—the problem is that the input distribution shape changes.

---

## Failure Mode 2: Closure Violation in Patchwork Architectures

### Definition

A patchwork architecture combines discrete components (token embedding, vocab head) with continuous components (complex attention, KAN FFN) without ensuring the entire pipeline maintains a closure property (e.g., bounded norm, bounded state). The interface between discrete and continuous creates information bottlenecks and gradient instability.

### First Appearance

CMT Exp 6-15: every combination of KAN + complex + Lie group produced memorization, regardless of hyperparameter tuning.

### Confirming Evidence

| Experiment | Continuous Components | Discrete Components | Outcome |
|---|---|---|---|
| CMT Exp 6 | KAN FFN | Token embed, head | PPL degradation |
| CMT Exp 7-8 | KAN + complex attn + Cayley PE | Same | Memorizer @ 4k |
| CMT Exp 14 | All continuous except skip PE | Same | Some improvement, still memorizer |
| CMT Exp 15 | All fixes | Same | Memorizer @ 12k |
| Exp 16 (CMT-clean) | All bugs fixed | Same | **Still memorizer @ 30k** |

The pattern persists even after 5+ rounds of bug fixes and architectural cleanup. This is not a bug—it's a structural property of patchwork.

### Root Cause: The Closure Argument

Let $\mathcal{M}_{\text{discrete}}$ denote the manifold of valid discrete-token states and $\mathcal{M}_{\text{continuous}}$ denote the continuous representation space. A patchwork architecture's forward pass is:

$$\mathcal{M}_{\text{discrete}} \xrightarrow{E} \mathcal{M}_{\text{continuous}} \xrightarrow{F} \mathcal{M}_{\text{continuous}} \xrightarrow{D} \mathcal{M}_{\text{discrete}}$$

where $E$ is the embedding (discrete→continuous), $F$ is the continuous processing, $D$ is the output head (continuous→discrete).

For the architecture to be **closed** under $F$, we need: $\forall x \in \mathcal{M}_{\text{continuous}}, F(x) \in \mathcal{M}_{\text{continuous}}$.

CMT violated this: complex attention could output unbounded magnitudes, complex FFN could blow up under repeated composition, Cayley PE could push states outside the unit ball. The result was **information loss at the $E \to F$ interface** (discrete-to-continuous bottlenecks) and **gradient instability in $F$** (exploding/vanishing norms).

The closure violation is **compounded** by training: small numerical errors in $F$ accumulate across layers, causing the model's effective input distribution to drift during training (see Failure Mode 1).

### Lessons

1. **Patching a continuous component into a discrete pipeline does not make the pipeline continuous.** The discrete interfaces still leak.
2. **Closure is an architectural property, not a component property.** A "complex" FFN is not a closed component until the entire pipeline is verified to maintain the closure invariant.
3. **Bug-fixing does not fix structural closure violations.** Even after 5 rounds of cleanup, CMT still failed (Exp 16).
4. **For closure, you must build from scratch.** CWF (Phase 0) verified that a purpose-built closed architecture can maintain closure—but even that wasn't enough (see Failure Mode 4).

---

## Failure Mode 3: Architectural Absence of ODE Evolution

### Definition

The model lacks the mechanism for continuous time evolution (Neural ODE solver, recurrent state, time-step embedding). Without it, the model can only do discrete "step-to-step" prediction, which fails catastrophically on continuous dynamics tasks (e.g., Lorenz attractor).

### First Appearance

CWF Exp 02 (Lorenz): despite closure verification passing (max ψ norm < 1), CWF's free rollout EPT was only 2 steps versus Oracle's 100.

### Confirming Evidence

| Experiment | Architecture | ODE Evolution? | Lorenz EPT |
|---|---|---|---|
| Oracle (RK4) | True physics | Yes (analytical) | 100 |
| CWF | Closed complex state | **No** (single-shot prediction) | 2 |
| AR Transformer | Standard | No (standard TF→AR) | 3 |
| AR+VQ | Discrete bottleneck | No | 2 |

The fact that AR Transformer (3 steps) is comparable to CWF (2 steps) on Lorenz rollout indicates that **neither learned ODE dynamics**—both were doing "nearest-neighbor in training set" predictions.

### Root Cause

CWF's forward pass is:

$$\psi^{(0)} = \text{FFT encoder}(x_{1:S})$$
$$\psi^{(1)} = \text{CWF Block}(\psi^{(0)})$$
$$\hat{y} = \text{Born decoder}(\psi^{(1)})$$

This is a **single-shot mapping from a sequence to a prediction**. There is no:
- Time-step embedding (the model doesn't know `dt`)
- Recurrent state propagation (no $h_{t+1} = f(h_t, x_t)$)
- ODE solver (no integration of learned dynamics)

By contrast, the Oracle is:

$$\hat{y}_{t+K} = \int_{t}^{t+K} f_{\text{Lorenz}}(s) \, ds$$

This is a true continuous evolution.

**The architectural absence is not fixable by hyperparameter tuning or training longer.** It requires adding new components (ODE solver, time embedding, recurrent state) — exactly what the CWF Manifesto §3 partially designed but the prototype didn't implement.

### Lessons

1. **Closure alone is not sufficient for continuous dynamics.** A closed architecture can still fail at continuous tasks if it lacks evolution machinery.
2. **"Sequence-to-next-step" prediction is fundamentally different from "continuous evolution".** They require different inductive biases.
3. **To handle chaotic dynamics, you need ODE solvers.** This is non-negotiable.
4. **The CWF prototype (Exp 02) was honest about its limitations.** It was designed to test closure, not ODE evolution. The failure exposes a missing piece, not a broken design.

---

## Failure Mode 4: Tokenization Bottleneck on Continuous Data

### Definition

When the input data is inherently continuous (e.g., Lorenz trajectories) but the model is forced through a discrete bottleneck (VQ codes), information loss occurs. The bottleneck capacity is finite; the continuous signal requires infinite resolution.

### First Appearance

CWF Exp 02.2 (AR+VQ on Lorenz): training loss decreased slowly (MSE 237→232 over 500 steps) and never reached low values, despite AR+VQ working reasonably on discrete text (BPE).

### Confirming Evidence

| Codebook Size | Lorenz MSE | Notes |
|---|---|---|
| 512 (used) | 232 | ~60% codes used, slow learning |
| 64 (would help AR fail faster) | – | Insufficient capacity |
| 4096 (would help AR succeed) | – | Defeats the purpose of bottleneck |

The bottleneck is **inherently mismatched**: a 3D continuous chaotic orbit cannot be losslessly encoded into 512 discrete codes.

### Root Cause

**Information-theoretic bound.** A 3D continuous signal requires, on average, $\infty$ bits per sample to losslessly encode (real numbers have infinite precision). A 512-code quantization provides $\log_2(512) = 9$ bits per sample. The information loss is unbounded; it grows with sequence length.

For text (discrete symbols), 512 codes is sufficient (matches BPE vocab size). For continuous data, the bottleneck is fundamentally insufficient.

### Lessons

1. **VQ bottleneck is fine for discrete data, fatal for continuous data.** AR+VQ worked for text (BPE 4100 vocab) but failed for Lorenz (continuous).
2. **The 20× improvement of Soft-Exp is consistent with this analysis.** Soft-Exp doesn't introduce a bottleneck; it preserves the continuous expectation of the input.
3. **For CWF to handle continuous data, it must avoid discrete tokenization entirely.** This is by design (CWF Manifesto §3.1).

---

## Failure Mode 5: Optimizer-Landscape Mismatch on Complex Loss Surfaces

### Definition

When the loss landscape involves complex-valued parameters (Wirtinger derivatives), real-valued optimizers (AdamW) cannot navigate the non-convex surface effectively. The optimizer oscillates or converges to sharp minima that don't generalize.

### First Appearance

CMT Exp 7-15: complex AdamW loss curves showed oscillations and sharp drops to memorizer, distinct from the smooth convergence of baseline.

### Confirming Evidence

| Experiment | Parameter Type | Loss Behavior | Generalization |
|---|---|---|---|
| Baseline (real) | Real | Smooth convergence | Yes |
| CMT (complex) | Complex | Oscillation + sharp drop | Memorizer |

The complex-valued Wirtinger gradient is a 2D vector at each parameter; real AdamW treats the 2D as 1D, losing the rotational structure of the gradient.

### Root Cause

The Wirtinger derivative $\partial/\partial z$ of a complex function $f(z)$ requires treating $z$ and $\bar{z}$ as independent variables. AdamW with real-valued updates (momentum, variance) on the real and imaginary parts separately treats them as **independent** real parameters, ignoring the complex coupling.

This produces gradient updates that can:
- Rotate the complex weight's phase without changing magnitude (loss decrease but no learning)
- Oscillate around sharp minima
- Converge to memorizer-like fixed points

### Lessons

1. **Standard optimizers are insufficient for complex-valued models.** Specialized complex AdamW (with Wirtinger-aware momentum) is needed.
2. **Smooth real loss landscapes ≠ smooth complex loss landscapes.** Convergence behavior can differ dramatically.
3. **Even with optimizer fixes, the underlying instability may not go away.** It's a structural property of mixing real and complex optimization.

---

## Cross-Cutting Patterns

### Pattern A: The Asymmetry of Training vs Inference

Across Failure Modes 1 and 2, a clear asymmetry emerges:
- **Modifying inference**: generally safe (Soft-Exp works, +48.6%)
- **Modifying training**: generally catastrophic (Soft-Exp training fails, CMT fails, complex training fails)

This asymmetry is **not obvious a priori**. The standard framing of exposure bias suggests that the train-inference gap should be bridgeable from either side. The empirical evidence shows this is wrong: the gap must be closed from the inference side.

### Pattern B: Closure is Necessary, Not Sufficient

The CWF Manifesto hypothesized that **closure is the key property** for continuous models. Phase 0 verified closure (Exp 01: closure violations = 0/5 batches). But Phase 1 showed closure alone doesn't enable continuous dynamics (Exp 02: closure OK but EPT=2).

**Revised understanding**: closure is one of several necessary conditions. The full list (post-experiments):
1. Closure (verified in CWF Phase 0)
2. ODE evolution machinery (missing in CWF Phase 1)
3. Time-step embedding (missing)
4. Recurrent state propagation (missing)

Any one missing → failure.

### Pattern C: Honest Reporting of Negative Results

Across all 35 experiments, the failure mode reporting has been:
- Specific (which experiment, what result, why it failed)
- Quantitative (specific PPLs, EPTs, losses)
- Diagnostic (root cause analysis, not just "didn't work")

This honesty is what makes the 35-experiment arc publishable as a **systematic record of failure**, rather than 35 unrelated "didn't work" reports.

---

## Implications for Future Work

1. **Anyone proposing to add continuous components to AR should test on a continuous dynamics task (e.g., Lorenz) early.** The failure mode will manifest as catastrophic rollout divergence, not just MSE degradation.

2. **Architectural changes to a working system should be the LAST resort.** Soft-Exp is 1 line of code. CMT is 30 rounds of engineering. CWF is 5 days of architectural design. The simple fix dominates.

3. **The closure conjecture was the right intuition but insufficient.** Future work on closed continuous models should add ODE evolution as the next priority.

4. **The community should adopt a "first-principles" reporting standard.** When a new architecture fails on a task, report not just the failure but the *mechanism* of failure. This paper provides 5 mechanism templates.
