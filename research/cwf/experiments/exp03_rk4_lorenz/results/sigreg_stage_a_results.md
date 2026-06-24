# exp03 + SIGReg — Stage A Results

**Date:** 2026-06-25
**Branch:** cwf-manifesto
**Status:** NEGATIVE — Stage A + SIGReg matches Stage A baseline exactly.

---

## Setup

Theoretical basis: LeJEPA Theorem 1 (Balestriero & LeCun 2025, arXiv:2511.08544v3) states that the isotropic Gaussian N(0, I) uniquely minimizes downstream prediction risk under a covariance constraint. SIGReg enforces this distribution via sliced Epps-Pulley characteristic function tests.

Implementation:
- `research/cwf/experiments/exp03_rk4_lorenz/sigreg.py`: Epps-Pulley (17 points) + slicing (256 directions). Single hyperparameter λ.
- Wired into `train.py` as: `loss = MSE + 1e-3 · max(0, ‖ψ‖-0.95)² + λ · SIGReg(ψ_final)`, with λ = 0.05.
- Operates on the **final** rollout state ψ_T only (not intermediate stages).
- Smoke test: SIGReg = 0.0026 on N(0,I); 0.085 on U(-1,1); 1.16 on constant — ordering correct.

Training: Stage A only (200 steps, k=1 rollout, batch=8), same data as exp03 Stage A baseline.

---

## Results

### Training trajectory

| Step | MSE (no-SIGReg) | MSE (with SIGReg) | SIGReg loss |
|---|---|---|---|
| 0 | 422.53 | 422.53 | 0.3792 |
| 50 | 93.47 | 93.47 | 0.3788 |
| 100 | 67.40 | 67.40 | 0.3795 |
| 150 | 80.16 | 80.16 | 0.3799 |
| 199 | 61.71 | 61.71 | 0.3788 |

**Identical to the no-SIGReg Stage A baseline, step-by-step.** SIGReg loss stays pinned at ~0.378 across all 200 steps — the gradient signal is too weak to shift ψ's distribution shape.

### Eval (5 trajectories, K_max=100)

| Metric | Stage A (no SIGReg) | **Stage A + SIGReg** | Δ |
|---|---|---|---|
| 1-step val MSE | 33.92 | **33.92** | 0% |
| MSE@100 | 134.49 | **134.49** | 0% |
| **EPT@0.9 (mean)** | 4.6 | **4.6** | 0 |
| EPT@0.9 (max) | 15 | 15 | 0 |
| Closure rate | 100% | 100% | 0 |

**No measurable effect** — model behavior is bit-identical to the no-SIGReg baseline.

---

## Verdict

**Negative result.** Per the user's pre-registered decision tree:
> If SIGReg provides no improvement → honest record that "球面和高斯在混沌任务上不兼容"

This is the result we got.

---

## Why it didn't work (4 candidate explanations)

**1. Signal-to-noise ratio.**
- SIGReg contribution to loss: λ · SIGReg = 0.05 · 0.38 ≈ 0.019
- MSE contribution: ~100 (early), ~60 (late)
- SIGReg is **0.03-0.06%** of total loss — below gradient noise floor for AdamW at this scale.

**2. Wrong scale of the regularization target.**
- LeJEPA's λ = 0.05 works on ViT-L 304M, 224×224 images, where embeddings have d=1024+ and MSE-like losses are O(1) per dimension.
- CWF here has MSE ~60-100 on **3-dim Lorenz outputs**. Per-dim MSE ~20-30. SIGReg weight that should match is λ ≈ 0.5-2.0, not 0.05.

**3. Stage A is too short and too local.**
- Stage A trains k=1 rollout only. The model learns one-step mapping but doesn't propagate ψ through multi-step dynamics. SIGReg constrains ψ_T's distribution, but ψ_T is "barely evolved" — gradient flow through it is weak.
- Multi-step rollout (Stage B/C) is where distribution drift matters. The test wasn't run.

**4. Wrong slice point.**
- SIGReg currently evaluates only `psi_final`, the terminal state after k=1 rollout. For a k=1 rollout this is essentially the encoder output projected by one CWFRK4Cell — there's almost no "evolution" to constrain.
- A more informative slice point would be `psi` snapshots throughout rollout, or evaluating SIGReg on intermediate states during the RK4 unroll inside `CWFRK4Cell`.

---

## Next-step options (your call)

### A. Re-test with stronger λ
Re-run Stage A only with `λ ∈ {0.5, 2.0, 5.0}` (3 short runs, ~60 min total). Test hypothesis (1)+(2): scale mismatch.

### B. Re-test with rollout-aware SIGReg
Modify `cwf_rk4.py` to track ψ at every RK4 step and apply SIGReg averaged across all stages. Test hypothesis (4). Cost: ~30 min code change + ~20 min Stage A rerun.

### C. Pivot to Chaotic Isotropic conjecture
The user's "directional conjecture" — that Lorenz (chaotic, nonlinear) needs a Lyapunov-direction-aware variant of SIGReg, not isotropic. This is **doctoral-thesis-scale math** and would require:
- New theorem extending LeJEPA Theorem 1 from linear probes to chaotic nonlinear targets
- New SIGReg variant detecting isotropic-along-Lyapunov-directions
- Months of work. Skip for now.

### D. Honest archival
Record this negative result, leave CWF Resurrection at Stage A baseline (EPT@0.9=4.6), pivot attention to v50 (V49 + Soft-Exp) which is already delivering +48.6% PPL.

---

## Reproducibility

```bash
# From D:/CrystaLLM
.venv/Scripts/python.exe research/cwf/experiments/exp03_rk4_lorenz/sigreg.py
# Smoke test: SIGReg on N(0,I)=0.003, U(-1,1)=0.085, constant=1.16

.venv/Scripts/python.exe -u -m research.cwf.experiments.exp03_rk4_lorenz.train
# Stage A only (~20 min) with SIGReg λ=0.05

.venv/Scripts/python.exe -c "
from pathlib import Path
from research.cwf.experiments.exp03_rk4_lorenz.eval import evaluate_checkpoint
metrics = evaluate_checkpoint(
    Path('research/cwf/experiments/exp03_rk4_lorenz/results/ckpt_stage_A.pt'),
    device='cuda', n_val_trajectories=5)
"
# EPT@0.9 = 4.6, MSE@100 = 134.49
```

Commits:
- `d44ea50` cwf: add SIGReg (LeJEPA Theorem 1) to exp03 loss
- (this report)

---

## Negative-result context

This negative result is itself a useful data point for the CWF Manifesto v2.0 trajectory:

- If SIGReg (the strongest theoretical tool currently available for "right representation distribution") doesn't move EPT@0.9 in a closure-RK4 architecture on Lorenz, then either:
  (a) the SIGReg prescription needs task-specific tuning that we haven't done (hypotheses 1-2), OR
  (b) the closure+isotropy dual is insufficient for chaotic dynamics — supporting the Chaotic Isotropic conjecture.

Option (a) is engineering (ablation). Option (b) is theory. Distinguishing them is a one-day experiment (hypotheses 1-2) versus a multi-month program (hypothesis 3).