# Exp03: CWF + RK4 on Lorenz — Partial Report

**Date:** 2026-06-24
**Branch:** cwf-manifesto
**Status:** PARTIAL — Stage A completed, Stage B and C interrupted (training process died silently after Stage B step 100/400).

---

## Goal

Verify that CWF + RK4 integration can learn continuous dynamics on the Lorenz system. GO criterion: EPT@0.9 ≥ 30 (vs Phase 1 baseline = 2, 10× improvement).

---

## Architecture (delivered)

RK4 symplectic integration over a closed wave block:
- `CWFSingleBlock` extended to accept optional `dt_embed` (Task 1, commits `43ac77e`, `124d93a`)
- `TimeStepEmbedding` maps scalar Δt → (d, 2) complex modulation (Task 2, `e7778097`)
- `CWFRK4Cell` = 4 CWF block calls + RK4 sum with `_project_to_disk` safety (Task 3, `1705b67d`)
- `MultiChannelCWFRK4Lorenz` = 3-channel encoder + RK4 rollout + 3 Born decoders, with post-encoder re-projection (Task 4, `834b965`)
- 100-step free rollout closure test: **9/9 unit tests pass**, including 10 batches × 100 RK4 steps × 4 stages with no closure violation

**Two spec gaps surfaced and patched** (documented in memory):
1. Spec §3.2 "closure auto-preserved" is mathematically loose — RK4 affine combinations violate the open disk via triangle inequality. Implemented `_project_to_disk` in CWFRK4Cell.
2. Three-channel encoder concatenation yields ‖ψ‖ = √3 > 1. Implemented post-encoder re-projection.

These patches are engineering-correct (closure preserved) but deviate from the spec's informal claim.

---

## Training (partial)

| Stage | Steps | Rollout | Status | Final train loss | Closure |
|---|---|---|---|---|---|
| **A** | 200 / 200 | k=1 | ✅ Complete | 61.7 (↓ 85% from 422.5) | 0.999 ✓ |
| B | 100 / 400 | k=4 | ⚠️ Interrupted at step 100 | 46.5 (oscillating, expected) | 0.999 ✓ |
| C | 0 / 800 | k=8 | ❌ Not started | — | — |

Training process died silently after Stage B step 100. Output log stopped updating, no Python exception captured. Suspected cause: GPU OOM or background process termination — not investigated.

**Training deviation from plan** (committed in `e447e5a`):
- Plan §5.1 verbatim: batch=32, A:500/B:1000/C:2000 steps → ~60h infeasible on RTX 5090 (Stage C alone = 53h)
- Adjusted: batch=8, A:200/B:400/C:800 → ~3h estimated
- Further reduced in practice by the silent termination

---

## Evaluation (Stage A only)

5 validation trajectories, K_max=100 rollout, same protocol as Phase 1 §6.4.

| Metric | Phase 1 baseline | **Stage A** | Δ |
|---|---|---|---|
| 1-step val MSE | 9.5 | **33.92** | 3.6× worse |
| MSE@100 (rollout) | 46.2 | **134.49** | 2.9× worse |
| **EPT@0.9 (mean)** | **2** | **4.6** | **2.3× better** |
| EPT@0.9 (max) | — | 15 | — |
| Closure rate | 100% | 100% | same |

### Verdict

**PARTIAL** (Stage A only — Stage B/C data unavailable):
- EPT@0.9 (mean) = 4.6 ∈ [2, 30): **architectural improvement over Phase 1** (2.3×) but well below the GO threshold of 30.
- 1-step MSE degradation: Stage A was trained on k=1 rollout only, so direct next-state prediction is weaker than the multi-step-aware Phase 1 baseline.
- Rollout MSE degradation: consistent with k=1 training — model hasn't learned error correction across long horizons.
- Closure property **preserved** throughout rollout (100% closure rate, max ‖ψ‖ = 0.999).

The architectural hypothesis (closure + RK4 helps) has a weak positive signal: EPT@0.9 doubled. The training hypothesis (3-stage curriculum → GO threshold) is **untested** because Stage B/C were interrupted.

---

## Spec gaps (engineering changes vs spec §3.2)

Both gaps were patched in Tasks 3 and 4. They reflect spec informal claims that don't hold under RK4 arithmetic:

1. **RK4 intermediate projection** (`CWFRK4Cell._project_to_disk`): triangle inequality allows `‖ψ + 0.5·Δt·k_i‖ > 1` when ‖ψ‖ is near boundary. Projection to ‖ψ‖ ≤ 0.999 preserves closure at the cost of textbook-RK4 fidelity (O(Δt²·ε) error).

2. **Encoder concatenation projection** (`MultiChannelCWFRK4Lorenz.forward`): per-channel ‖ψ_ch‖ ≤ 1 concatenated gives ‖ψ‖ = √3 > 1. Same `_project_to_disk`-style projection.

Both fixes are engineering-correct and preserve the empirical closure invariant. A future v51+ would need to prove ‖ψ‖ ≤ 1 - Δt·max(‖k_i‖) always holds (strong constraint, likely unsatisfiable) to remove them.

---

## Reproducibility

```bash
# From D:/CrystaLLM
.venv/Scripts/python.exe -m pytest research/cwf/experiments/exp03_rk4_lorenz/tests/ -v
# 9/9 tests pass (~5 min)

.venv/Scripts/python.exe -m research.cwf.experiments.exp03_rk4_lorenz.train
# ~3h for the adjusted Stage A/B/C curriculum

.venv/Scripts/python.exe -m research.cwf.experiments.exp03_rk4_lorenz.eval
# ~7.7 min for Stage A eval with 5 val trajectories
```

Hardware: NVIDIA RTX 5090 (32 GB VRAM), CUDA via PyTorch 2.9.1.

---

## Next steps

To complete the experiment, restart training from scratch (Stage A ckpt is preserved) and investigate why Stage B's process died silently. Likely root cause: GPU OOM at k=4 batch=8, or stdout/tee buffering issue masking an exception.

If Stage C reaches EPT@0.9 ≥ 30, CWF Manifesto hypothesis is empirically supported for ODE-class tasks.
If Stage C plateau < 30 (e.g. ~5-10), the architectural improvement is real but insufficient — consider:
- Larger model capacity (more channels, more layers)
- Multi-resolution training (vary Δt during curriculum)
- Neural CDE framework (Task 2 in the original brainstorm, skipped in favor of RK4)

If Stage C fails (loss explodes or closure violations), the closure-RK4 composition is fundamentally limited and the CWF charter should pivot to non-RK4 continuous-evolution methods.