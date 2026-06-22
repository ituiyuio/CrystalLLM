# LieFormer: Lie Group Residual Transformer with Symplectic Integrator

**Status:** Draft v1.0
**Date:** 2026-06-23
**Objective:** Replace Euclidean residual updates with SO(d) Cayley actions and forced symplectic integration, maintaining training stability and enabling geodesic attention. Designed for SkyPile 5B token corpus, BPE 16k, 3-stage scaling (50M → 200M → 1.2B).
**Baseline:** V49 Transformer (val_ppl 2.36 on BPE 4k), soft-exp inference head (刀4).

---

## §1 Architecture Overview

*LieFormer* lifts the residual stream from ℝᵈ to an SO(d)-equivariant representation using:

- **Cayley transform** for parameterizing local SO(d) actions (no matrix exponential).
- **Geodesic attention** with L2 distance under head-wise Cayley rotations.
- **Forced symplectic integrator** (leapfrog) on the residual branch (p,q split), preserving symplectic 2-form while keeping FFN capacity intact.
- **Soft-Exp (Expected Embedding) decoding** at inference (optional, inherited from v50).

The hidden state **x ∈ ℝᵈ remains a vector**, not a full group element. Group actions are applied as operators R·x. The three soft invariants (per-forward, logged but not asserted):

1. `R_h^T R_h ≈ I` (orthogonality, error < 1e-3)
2. `det(R_h) ≈ +1` (orientation)
3. `‖R_h‖_F ≈ √d_h` (Frobenius norm conservation)

---

## §2 Geodesic Attention (Head-wise Cayley + L2 Distance)

**Input:** x ∈ ℝ^{B×L×d}
**Output:** x_attn ∈ ℝ^{B×L×d}

### 2.1 Head-wise Cayley Rotation

For each of H heads (head_dim d_h = d/H):

- Compute A_h from x via a linear projection `Linear_d→d²` → reshape to (d_h, d_h) per head → make skew-symmetric: A_h = (A_h - A_h^T)/2.
- Cayley map: R_h = (I - A_h)^{-1}(I + A_h). Shape (B,L,d_h,d_h) per head.
  *Numerical:* Direct matrix inverse (O(d_h³)) used; for d_h ≤ 64, cost negligible. At 1.2B stage, if d_h > 64, switch to Newton-Schulz iterative inversion (5–10 iterations).

### 2.2 Query/Key Rotation & Score

- **Initial implementation**: Q and K **share the same R_h per head** (single Cayley rotation per head, generated from one A_h). Independent Q/K rotation can be tested as ablation in later stages.
- Q = R_h · (W_Q x), K = R_h · (W_K x).
- Score: S = -‖Q - K‖² / √d_h.
  This is **SO(d)-equivariant** (‖Rq - Rk‖ = ‖q - k‖) and avoids arccos gradient vanishing.

### 2.3 Attention and Output

- Standard causal softmax + V projection.
- Output x_attn ∈ ℝ^{B×L×d} (concatenated heads).

*Complexity:* Each head's Cayley inversion is O(d_h³); total 8×64³ ≈ 2M FLOPs per token, negligible vs. attention matmul.

---

## §2.6 Handoff: Residual Connection to Symplectic Block

We adopt **H-B (Cross-Force Injection)**:

- `x' = x + Dropout(Attn(x))`   # first residual (training anchor)
- Split x' into p, q of equal dimension d/2 along d-axis.
- Split attn output x_a into attn_p, attn_q (chunk along d, zero extra params).
- FFN input = x' (standard Transformer placement, aligned with V49); FFN is **SwiGLU with hidden dim 4×d** (e.g. 50M d=512 → FFN hidden 2048). FFN output split into ffn_p, ffn_q.
- Forces: F_p = attn_p + ffn_p, F_q = attn_q + ffn_q.  *Additive, zero extra params, semantically aligned with vector force composition.*

---

## §3 Symplectic Block (Forced Leapfrog)

**Input:** p,q ∈ ℝ^{B,L,d/2}, forces F_p, F_q from §2.6
**Output:** p_new, q_new ∈ ℝ^{B,L,d/2} → concatenated to x_out

### 3.1 Leapfrog Integrator (3 substeps)

```
p_half = p + 0.5 * Δt * F_q
q_new  = q + Δt * F_p
p_new  = p_half + 0.5 * Δt * F_q
```

Δt is a block-level learnable parameter (init 1.0, clamped [0.01, 5.0]).
Forces are *state-independent* within the step (forced integrator), so Jacobian is identity → trivial symplectic, but leapfrog structure preserves reversibility and 2nd-order accuracy.

### 3.2 Δt Schedule

- 0–500 steps: freeze Δt = 0.1 (warmup, no gradient).
- 500+ steps: unblock gradient, learn with **independent Adam (lr=1e-3, no weight decay, no warmup)**. Δt changes much slower than model params, so the higher lr is acceptable.
- Global clamp [0.01, 5.0] applied after each step.

### 3.3 Monitoring (Soft thresholds, no hard assertions)

| Metric | Computation | Healthy range (5k+ steps) |
|--------|-------------|---------------------------|
| ‖F_p‖_F, ‖F_q‖_F | Frobenius norm | 0.5 – 100 |
| Step sizes ‖Δp‖, ‖Δq‖ | Normalized per element | < 2.0 (later < 0.5) |
| ω drift | relative change in pᵀq | < 0.3 |
| Pseudo-energy drift | **relative** \|E_new - E_old\| / E_old, E = (‖p‖² + ‖q‖²)/2 | < 0.5 |
| Δt value | scalar | 0.1 – 2.0, warning > 5 |

*Thresholds phased:*
- 0–1k steps: record only, no warnings.
- 1k–5k steps: wide thresholds (F norm 1–200, step <5, drift <2, Δt 0.01–5).
- 5k+ steps: strict thresholds as above.

*Soft invariants logged (not asserted):* R_h orthogonality error, det(R_h) drift, ‖R_h‖_F drift.

### 3.4 Honest declaration (for paper / future reviewers)

> This architecture applies a **forced symplectic integrator** (leapfrog) on the residual branch with learnable step size Δt. F_p, F_q are derived from attention and FFN outputs split along d, and **do not form gradients of a single scalar function**. Therefore this is **not** a Hamiltonian Neural Network (Greydanus et al., 2019). We use the name "Symplectic Block" to avoid confusion. When F is state-independent (as in this design), the leapfrog Jacobian is the identity matrix — symplectic structure is preserved in the *update form* (reversibility, 2nd-order accuracy) rather than the update *Jacobian*. This "forced force field + explicit symplectic scheme" combination is studied in Neural ODE literature (e.g. Chen et al. 2018) as a first-order splitting scheme extension; it provides reversibility but not energy conservation.

---

## §4 Training & Evaluation Pipeline

### 4.1 Loss

Standard next-token cross-entropy with label smoothing 0.1. No symplectic regularization (would be meaningless for forced integrator without a true H).

### 4.2 Optimizer

AdamW, β=(0.9,0.95), weight decay 0.1. Peak learning rates: 50M: 3e-4, 200M: 2e-4, 1.2B: 1.5e-4. Cosine decay to min lr (10× lower). Warmup 2000 steps (linear). Gradient clipping 1.0 (L2). Δt optimizer separate (Adam, lr=1e-3, no weight decay, no warmup — Δt has its own warmup via §3.2).

| Param | 50M POC | 200M | 1.2B |
|---|---|---|---|
| Peak LR | 3e-4 | 2e-4 | 1.5e-4 |
| Min LR | 3e-5 | 2e-5 | 1.5e-5 |
| Batch (seqs × tokens) | 64 × 2048 | 64 × 2048 | 32 × 2048 × 4 accum (eff. 128) |

### 4.3 Soft-Exp Inference (刀4, optional)

At each decoding step:

```python
p = logits.softmax(-1)                  # (B, V)
expected_emb = p @ token_embedding.weight   # (B, d)
refined = F.linear(expected_emb, lm_head.weight)  # (B, V) — IMPORTANT: ensure lm_head.weight ≠ token_embedding.weight (not tied); if tied, detach or copy before this op
next_token = refined.argmax(-1)
```

- Training: **not** used.
- **50M POC Round 1: disabled.** First round measures baseline PPL and 8-dim without Soft-Exp. Round 2 enables Soft-Exp and re-evaluates.
- Soft-Exp is orthogonal to LieFormer internals (operates only on lm_head / token_embedding weights, both shapes identical to V49).

### 4.4 8-Dimensional Evaluation (SkyPile held-out)

**V1.0 (5 dims):** PPL, Diversity (4-gram entropy), Coherence (6-prompt human/LM-judge), OOD (Pile/CodeParrot PPL), BPC. All compared against V49 baseline thresholds.
**V1.1 (3 dims):** n-gram entropy (Shannon on 4-grams of generated text, threshold > 4.0), top-1 confidence (max softmax prob average, healthy 0.3–0.7), val-train PPL gap (relative, threshold < 0.15).

All compared against V49 baseline (PPL 2.36, BPC baseline, etc.).

### 4.5 Three-Stage Scaling & Gates

| Stage | Params | Steps | Data | Gate Criteria |
|-------|--------|-------|------|---------------|
| 50M POC | ~50M | 30k | SkyPile 5B subset (100M tokens) | 8-dim ≥ 5 pass; PPL < 30; pseudo-energy stable; **Round 1 without Soft-Exp** |
| 200M | ~200M | 30k | SkyPile 5B subset (500M tokens) | ≥ 6 pass; PPL < 8; diversity > 3.5; Soft-Exp Round 2 evaluated |
| 1.2B | ~1.2B | 50k | SkyPile 5B full (5B tokens) | All 8 pass; PPL < 4; coherence ≥ 5/6; Soft-Exp integrated |

Checkpoints at 8k, 16k, 30k (50M/200M); 10k, 25k, 50k (1.2B).

**Gate decision tree:**
- PPL fail + diversity collapse → check Δt clamping, re-run with LR halved.
- PPL good + diversity bad → "memorizer illusion" (per Exp 22), skip to next stage with more data.
- NaN → force Δt=0.05, extend warmup to 5k steps, restart from last checkpoint.

### 4.6 Data & Tokenizer

- **Corpus:** SkyPile-150B (HuggingFace), random 5B token subset (70% Chinese, 20% code, 10% math formulas).
- **Tokenizer:** BPE 16k (vocab 16384), trained with rustbpe, different from V49's 4k (5B corpus needs larger vocab).
- **Sequence length:** 2048, packing enabled with cross-document attention mask.

```yaml
data:
  source: SkyPile-150B
  subset_tokens: 5_000_000_000
  mix: {zh: 0.7, code: 0.2, math: 0.1}
  tokenizer: {algo: BPE, vocab_size: 16384, min_freq: 50, special: ["<pad>","<bos>","<eos>","<unk>"]}
  seq_length: 2048
  packing: true
```

### 4.7 Checkpointing

Each checkpoint stores:

```python
{
    "model_state_dict": ...,        # includes Δt
    "optimizer_state_dict": ...,    # main AdamW
    "dt_optimizer_state_dict": ..., # Δt's separate Adam
    "lr_scheduler_state_dict": ...,
    "step": step,
    "metrics_8d": {all 8 dimensions},
    "Δt": current_value,
    "monitor_history": {F norms, ω drift, pseudo-energy drift},
    "rng_state": ...,
}
```

**Resumption resets monitoring thresholds** (0–1k: record only; 1k–5k: wide; 5k+: strict), not inheriting the previous run's "converged" assumption.

### 4.8 Risk & Fallback

LieFormer is a "superset attempt" over v50. If it fails at any stage, fall back to:

- **50M fail** → baseline + Cayley PE only (proven in Exp 24, PPL 1.0435 < RoPE 1.05).
- **200M fail** → baseline + Cayley PE + BPE (V50 original main line).
- **1.2B fail** → V49 1.2B (val_ppl 2.36, PPL dim known to pass).

Soft-Exp is independent and can be applied to any baseline fallback.

**Key risk points:**
1. Head-wise Cayley inversion (8 × 64³) at training start: OOM risk if memory tight → 50M POC must validate this first.
2. Δt learnable may push outside [0.01, 5.0] clamp — clamp is the last line of defense.
3. Soft-Exp in BPE 16k has never been tested → 50M POC Round 1 must run without it.

---

## Appendix A: Full Block Shape Chain (50M POC, d=512, H=8, d_h=64)

```
输入: x ∈ R^{B,L,d=512}        # post-embedding, post-R_0 Cayley
       ↓
[Linear d→d², reshape (B,L,8,64,64), skew-symmetric, H×Cayley]   # 8 × Cayley(64³)
       ↓
[Q, K 共享 R_h, V=W_V x, Geodesic score = -‖Q-K‖²/√d_h, softmax, V aggregate]
       ↓
x_a ∈ R^{B,L,d=512}              # Attention output
       ↓
x' = x + Dropout(x_a)             # 第一残差, (B, L, 512)
       ↓
p, q = x'.chunk(2, dim=-1)        # 各 (B, L, 256)
attn_p, attn_q = x_a.chunk(2, dim=-1)  # 各 (B, L, 256)
ffn_out = SwiGLU(W_1·x', W_2·x', W_3·x') with W_i ∈ R^{d×4d}    # (B, L, 512)
ffn_p, ffn_q = ffn_out.chunk(2, dim=-1)  # 各 (B, L, 256)
       ↓
F_p = attn_p + ffn_p              # (B, L, 256)
F_q = attn_q + ffn_q              # (B, L, 256)
       ↓
p_half = p + 0.5·Δt·F_q
q_new  = q + Δt·F_p
p_new  = p_half + 0.5·Δt·F_q
       ↓
x_out = cat([p_new, q_new], dim=-1)  # (B, L, 512)
       ↓
(RMSNorm → next block, ×L)
```

## Appendix B: Numerical Cost Analysis (50M POC, d=512, d_h=64, H=8, L=2048)

| Op | Complexity | FLOPs (B,L=64,2048) |
|---|---|---|
| Linear→A (×8 head) | 8 × L × d × d_h² | 0.5G |
| Cayley inversion (×8) | 8 × L × d_h³ | 0.005G |
| Q, K rotation | L × H × d_h² | 0.04G |
| Attention score | L² × H × d_h | 1.0G |
| Softmax + V | L² × H × d_h | 1.0G |
| FFN (SwiGLU 4×d) | 3 × L × d × 4d | 6.0G |
| **Block total** | | **~8.5G** |

vs V49 baseline: ~6G (no Cayley, no chunk). **~40% slower per step**, but 30k steps × 3h vs 3.1h × 1.2B → still tractable.

## Appendix C: Decision Log (Why this design, not alternatives)

| Choice | Rejected alternative | Reason |
|---|---|---|
| SO(N) Cayley | SE(3) exp map | Cayley O(d³) one-shot, no quaternion overhead; Cayley PE proven in Exp 24 |
| Forced symplectic | True HNN | F_p, F_q are not gradients of scalar H; forced integrator simpler, trainable |
| 3-stage 50M→200M→1.2B | Direct 1.2B | CMT 25 轮全失败因无 gate; need cheap POC to validate |
| SkyPile 5B | V49 corpus (12GB) | 25 轮都在 v23 数据, 难区分骨架 vs 数据; 换数据换骨架同时变 |
| BPE 16k | V49 BPE 4k | 5B token 数据需更大词表, 16k 是 BPE-中文甜点 |
| Soft-Exp in Round 2 | Round 1 | 50M POC 先验证骨架, 再测推理时增强 |
| Q/K share R_h | Independent R_q, R_k | Initial stability; ablation in later stages |
| Additive F_p, F_q | Element-wise product (gating) | Additive aligned with vector force composition; gating adds 2nd-order interactions that destabilize small models |

---

*End of spec v1.0. Next: user review → writing-plans skill.*
