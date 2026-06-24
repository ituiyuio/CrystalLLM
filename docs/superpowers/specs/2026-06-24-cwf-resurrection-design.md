# CWF Resurrection: RK4 Integration over Closed Wave Block

**Spec**: `docs/superpowers/specs/2026-06-24-cwf-resurrection-design.md`
**Branch**: `cwf-manifesto` (existing; new experiment lives under `research/cwf/experiments/exp03_rk4_lorenz/`)
**Date**: 2026-06-24
**Status**: Draft (pending user review)

---

## 1. Background

The CWF (Closed Waveformer) Manifesto claims that a continuously-evolving, closure-preserving architecture can model any continuous information source — including language, physics, and multimodal signals. Two empirical results frame this spec:

**CWF Phase 1 (Lorenz, FAIL)**: `docs/papers/2026-06-23-35-failures/04_cwf_lorenz.md`
- Closure property verified across 10 random batches at initialization.
- 1-step teacher-forced MSE = 9.5 vs AR+VQ baseline = 232.6 (24× advantage).
- **100-step free rollout EPT@0.9 = 2 steps** vs Oracle = 100 steps (factor 25 short of GENUINE_PASS).
- Diagnosis (§6.5): single-shot forward pass lacks ODE evolution machinery — no time-step embedding, no recurrent state, no ODE solver.

**CMT text experiments (35 rounds, FAIL)**: char-level + discrete tokenization fundamentally misfit a continuous wave architecture. v50 was locked to baseline + inference Soft-Exp.

This spec targets the *first* failure (Phase 1) directly. We attempt to fulfill the three components enumerated in §6.7 of the 35-failures paper — **Neural ODE solver**, **time-step embedding**, **recurrent state propagation** — with the minimal patch that preserves the closure property CWF was designed around.

---

## 2. Goal

**Primary**: Bring CWF on the Lorenz system from EPT@0.9 = 2 steps to EPT@0.9 ≥ 30 steps (10× improvement). This is the GO threshold for Phase 2 of the CWF charter.

**Non-goals**:
- Recovering the language-modeling track (v50 owns this; CMT line is closed).
- Reaching Oracle-level EPT (100 steps). 30 steps is sufficient to demonstrate CWF can *learn dynamics*, not merely approximate locally.
- Replacing the Phase 1 baseline. It is preserved as the comparison reference.

**Why 30 steps, not 50 or 100**: 30 is the smallest integer that demonstrates the architecture actually learned *continuous evolution* rather than memorized a fixed-trajectory mapping. Anything below 10 would still be consistent with "single-shot next-state mapping" failure mode.

---

## 3. Architecture

### 3.1 High-level flow

```
Input:   x = (B, T, 3)        continuous Lorenz trajectory
         Δt = scalar          integration step (e.g. 0.01)

For t in 0 .. T-1:
    ψ_t       = FFT encoder(x[:, t, :])    # (B, 3d, 2), in 𝔻^3d
    k1        = CWF_block(ψ_t,                       Δt)
    k2        = CWF_block(ψ_t + 0.5·Δt·k1,           Δt)
    k3        = CWF_block(ψ_t + 0.5·Δt·k2,           Δt)
    k4        = CWF_block(ψ_t +     Δt·k3,           Δt)
    ψ_{t+1}   = ψ_t + (Δt/6)·(k1 + 2·k2 + 2·k3 + k4)

Output:  ŷ = Born decoder(ψ_{T})                 # (B, 3) predicted next state
```

### 3.2 Closure preservation theorem (informal)

> If `CWF_block(ψ, Δt)` is closure-preserving (‖out‖ < 1 whenever ‖ψ‖ < 1, and Δt is bounded), then for Δt ≤ Δt_max the RK4-integrated trajectory satisfies ‖ψ_t‖ < 1 for all t.

This is a standard argument: RK4 is a symplectic integrator on continuous flows; when applied to a Lipschitz-bounded operator with norm contractivity, the discrete orbit stays in the invariant set. We verify this empirically (per-step norm logging) rather than proving it formally. If closure ever violates, we raise `RuntimeError`.

### 3.3 Why not Neural ODE + adjoint?

- The adjoint method (Chen et al. 2018) is memory-efficient but unstable under complex-valued parameters and norm constraints.
- RK4 + autograd is simpler, deterministic, and the closure property is easier to verify.
- The cost of storing intermediate RK4 stages (4× activation memory) is negligible at Lorenz-scale (0.5–2M params).

---

## 4. Components

| Component | Type | Source | Notes |
|---|---|---|---|
| `CWFRK4Cell` | NEW | new file | Single RK4 step = 4× CWF block calls + weighted sum |
| `TimeStepEmbedding` | NEW | new file | Scalar Δt → (B, 3d, 2) complex modulation, broadcast-added to ψ at each CWF call |
| `CWFRollout` | NEW | new file | Wraps `CWFRK4Cell` for T-step unrolled rollout with closure monitoring |
| `CWFSingleBlock` | REUSE | `cwf_minimal.py` | Unchanged; signature extended to accept `(ψ, Δt_embed)` |
| `_FFTChannelEncoder` | REUSE | `cwf_lorenz.py` | Unchanged |
| `_BornChannelDecoder` | REUSE | `cwf_lorenz.py` | Unchanged |
| `lorenz_data.py` | REUSE | `exp02_lorenz/` | Same generator (200 trajectories, Δt=0.01, seq_len=1024) |

**New code estimate**: ~150 lines (architecture) + ~80 lines (training loop) + ~60 lines (eval).

---

## 5. Training Protocol

### 5.1 Curriculum

Three-stage curriculum to avoid the single-step → multi-step distribution shift (cf. `exp30` failure):

| Stage | Steps | Horizon | Purpose |
|---|---|---|---|
| Stage A: 1-step | 0–500 | K=1 | Anchor basic next-state mapping |
| Stage B: 4-step | 500–1500 | K=4 | Force multi-step consistency |
| Stage C: 16-step | 1500–3500 | K=16 | Force rollout-level learning |

Each stage re-initializes the optimizer state and uses the previous stage's checkpoint as warm start (only the model parameters; optimizer momentum reset to avoid stale LR-schedule coupling).

### 5.2 Hyperparameters

- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Schedule: cosine over 3500 steps
- Batch size: 32 (down from 8 in exp02 to improve gradient signal at K=16)
- Gradient clip: ‖g‖_2 ≤ 1.0
- Δt embedding dim: 32 (matches single-channel d)
- Δt_max for closure test: 0.05 (5× training Δt)

### 5.3 Loss

- Primary: MSE(ŷ, y_target) on the final state.
- Closure auxiliary: penalty = 1e-3 · max(0, ‖ψ‖ − 0.95)² summed over rollout steps. This is a *soft* warning — only activates when ‖ψ‖ drifts into the boundary region. Hard closure is enforced by assertion.

---

## 6. Evaluation

Same protocol as Phase 1 (§6.4 of the 35-failures paper) so the result is directly comparable:

| Metric | Definition | Compared against |
|---|---|---|
| 1-step val MSE | teacher-forced next-state error | Phase 1 CWF (9.5), AR+VQ (232.6), Oracle (0.43) |
| K-step val MSE | free rollout MSE at K ∈ {1, 10, 25, 50, 100} | same baselines |
| EPT@0.9 | first step at which per-dimension Pearson r < 0.9 | Phase 1 (2), AR+VQ (3), Oracle (100) |
| Closure rate | fraction of rollout steps with ‖ψ‖ < 1 | must be 1.0 |

### 6.1 GO/NO-GO gate

| Verdict | Condition | Action |
|---|---|---|
| **GO** | EPT@0.9 ≥ 30 AND closure rate = 1.0 | Proceed to Phase 2: other ODE systems (e.g. Duffing, Van der Pol) |
| **PARTIAL** | EPT@0.9 ∈ [10, 30) | Diagnose: is failure at CWF block (capacity) or RK4 (instability)? |
| **FAIL** | EPT@0.9 < 10 | CWF Manifesto hypothesis falsified for ODE-class tasks |

---

## 7. Risks & Honest Caveats

| Risk | Probability | Mitigation |
|---|---|---|
| RK4 + CWF closure violation at multi-step rollout | Medium | Per-step norm assertion; if violated, abort and diagnose which stage caused drift |
| CWF block lacks capacity to represent d/dt | Medium | Will manifest as PARTIAL verdict; failure is informative (architecture limit, not ODE framework) |
| Δt embedding is learned but ignored by the dynamics | Low | Inspect gradient flow into TimeStepEmbedding |
| Numerical instability in RK4 stages (large k1..k4) | Low | Gradient clip + norm assertion per stage |
| This experiment consumes attention budget that v50 needs | Low | Lives entirely under `research/cwf/`, does not touch `experiments/v49_pre/` or `crystalllm/` |

**Critical honesty clause**: If EPT@0.9 = 30 still fails, the *next* paper conclusion is "CWF can approximate locally but cannot model continuous evolution under its current closure constraint" — not "CWF is saved by changing the training set". The user's intuition about "special dataset design" was redirected to "continuous-domain task" by clarifying discussion; we now treat this experiment as the load-bearing test of whether the CWF Manifesto hypothesis survives at all.

---

## 8. File Layout (post-implementation)

```
research/cwf/experiments/
├── exp02_lorenz/                    # UNCHANGED (baseline for comparison)
│   ├── cwf_lorenz.py
│   └── results/cwf_lorenz_500.pt
└── exp03_rk4_lorenz/                # NEW
    ├── cwf_rk4.py                   # CWFRK4Cell + CWFRollout + TimeStepEmbedding
    ├── train.py                     # 3-stage curriculum + closure monitoring
    ├── eval.py                      # 1-step + K-step rollout + EPT@0.9
    ├── lorenz_data.py               # re-export from exp02 (or symlink)
    └── results/
        ├── ckpt_stage_a.pt
        ├── ckpt_stage_b.pt
        ├── ckpt_stage_c.pt
        └── exp03_rk4_lorenz.md      # final report (mirrors Phase 1 format)
```

No new files outside `research/cwf/`. No edits to existing files except the optional signature extension of `CWFSingleBlock` (adds a second arg `Δt_embed`, default `None` for backward compatibility).

---

## 9. Implementation Order (for writing-plans)

1. **TimeStepEmbedding** (~20 lines) — standalone, easily tested.
2. **CWFRK4Cell** (~40 lines) — wraps CWFSingleBlock + TimeStepEmbedding.
3. **CWFRollout** (~50 lines) — runs T-step rollout, returns closure history.
4. **Closure self-test** (~30 lines) — assert ‖ψ‖ < 1 across 100 random batches.
5. **Training loop** (~80 lines) — 3-stage curriculum with checkpoint per stage.
6. **Evaluation script** (~60 lines) — 1-step + K-step + EPT@0.9.
7. **Final report** (markdown, mirrors Phase 1 §6.4 tables).

---

## 10. Open Questions

None at design level. All architectural choices were resolved through the clarifying questions. If implementation surfaces an unforeseen constraint (e.g. closure violation at large Δt), we will surface it before committing the implementation plan.