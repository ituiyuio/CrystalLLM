# exp14 — Analytic Gabor Codec: ANALYTIC_INFERIOR (complex advantage vanishes)

**Date:** 2026-07-06
**Branch:** cwf-manifesto
**Status:** **ANALYTIC_INFERIOR** — the analytic Gabor encoder (1.41) is dramatically worse than the CNN encoder (0.50), by 0.91 nat. Most revealingly, the analytic complex encoder (1.41) lands at **almost exactly the real CNN baseline (1.43)** — the 2.8× complex advantage that the CNN encoder shows **completely disappears** with the analytic Gabor frame. The complex advantage in Stage A comes from the CNN's *learnable complex filters*, not from the complex representation per se.

---

## Question

Can an analytic Gabor-frame encoder (wave packet = Gaussian × carrier wave, with byte-dependent frequency) match or beat the black-box CNN encoder's 0.54 reconstruction? This tests whether the analytic structure is a viable inductive bias for the codec.

## Design

4 conditions × 3 seeds × 3000 steps (Stage A reconstruction, same config as exp05/exp12):
1. **CNN** — exp05 WaveTokenizerComplex (baseline), reproduce ~0.54.
2. **CNN+gauge** — exp12 gauge-fixed CNN.
3. **Analytic** — AnalyticWavePacketEncoder (Gabor atoms) + same CNN decoder.
4. **Analytic+gauge** — Gabor + gauge-fix + same decoder.

The analytic encoder maps each byte to a Gabor atom: `ψ[m,c] = Σ_i A[b_i,c] · G[i,m] · exp(i·k[b_i]·2π·m/M)`, where A is learned amplitude (semantic strength), k is learned frequency (token identity), G is Gaussian kernel (localization). 8,449 encoder params vs CNN's ~28,800.

## Results

| condition | s42 | s123 | s2024 | mean |
|---|---|---|---|---|
| CNN | 0.496 | 0.495 | 0.506 | **0.499** |
| CNN+gauge | 0.519 | 0.548 | 0.590 | 0.552 |
| Analytic | 1.450 | 1.415 | 1.372 | **1.413** |
| Analytic+gauge | 1.451 | 1.465 | 1.387 | 1.434 |
| real (exp12 ref) | — | — | — | 1.426 |

**Analytic (1.413) vs CNN (0.499): Δ = +0.914 nat (analytic far worse).** Past the 0.20 "too rigid" threshold by 4.5×.

### The key finding: complex advantage vanishes

| encoder | complex | real | complex advantage |
|---|---|---|---|
| CNN (learnable filters) | 0.499 | 1.426 | **2.86×** |
| Analytic Gabor | 1.413 | 1.426 | **1.01×** (none) |

The analytic complex encoder performs **identically to the real CNN**. The 2.8× complex advantage that the CNN encoder shows — the entire basis for "Stage A is the durable CWF contribution" — **comes from the CNN's learnable complex convolution filters, not from the complex representation itself**. When you replace the learnable filters with an analytic Gabor frame, the complex advantage disappears entirely.

### Trajectory (seed 42)

| step | CNN val | Analytic val |
|---|---|---|
| 200 | 3.335 | 3.582 |
| 500 | 2.321 | 2.891 |
| 1000 | 1.304 | 2.243 |
| 2000 | 0.690 | 1.602 |
| 3000 | 0.496 | 1.450 |

The analytic encoder isn't broken (loss drops 5.5→1.45), but it's ~3× slower and plateaus much higher. It's still descending at 3000 — more steps might help, but not enough to close a 0.91-nat gap.

## Diagnosis

1. **The Gabor frame is too rigid.** A single σ + single frequency-per-byte cannot capture the multi-scale, non-stationary structure that byte text has. The CNN's 5-kernel strided convolutions learn position-dependent, content-dependent filters that adapt; the Gabor atom is a fixed functional form.

2. **The complex advantage comes from learnable filters, not representation.** This is the most important finding. exp12 confirmed the 2.8× advantage survives gauge-fixing, so it's not a gauge artifact. But exp14 shows it **doesn't survive replacing learnable convolutions with an analytic frame**. The advantage lives in the *interaction* between complex arithmetic and learnable filters — the complex Conv1d can learn filters that exploit phase interference in ways a fixed Gabor frame cannot.

3. **This refines the postmortem's Stage A claim.** The postmortem said "Stage A's Born-rule loss is gauge-invariant → complex orthogonality is pure benefit." exp14 shows the benefit is **not** from "complex orthogonality" in the abstract — it's from **learnable complex filters**. Real CNN + ReLU can also learn orthogonality (via sign patterns), but complex Conv1d does it more parameter-efficiently. The advantage is an optimization/capacity efficiency, not a representation-theoretic one.

## What this means for codec design

- **The analytic Gabor frame is not the right direction.** It's too rigid and loses the complex advantage entirely.
- **The CNN encoder's learnable filters are doing the real work.** Any codec design must preserve learnable filters — the complex advantage lives there.
- **Hybrid approach (Gabor → 1 conv refinement)** is the natural next test, but the 0.91-nat gap is so large that one refinement layer is unlikely to close it. The CNN's two strided conv layers build a multi-scale representation that the Gabor frame's single-scale structure cannot match.
- **Multi-scale Gabor** (multiple σ values) might help, but would need many scales to approximate what the CNN learns, defeating the "analytic simplicity" motivation.

## Verdict

**ANALYTIC_INFERIOR.** The analytic Gabor encoder is dramatically worse than the CNN (1.41 vs 0.50) and, most revealingly, performs identically to the real CNN — the complex advantage vanishes. The Stage A 2.8× advantage comes from learnable complex filters, not from the complex representation or the Gabor frame's structure.

**This is a negative result for the analytic codec direction, but a positive result for understanding.** It tells us where the CWF advantage actually lives: in the learnable filters, not in the representation or the physics. Any future codec design should build on learnable complex convolutions (which work), not on analytic frames (which don't).

## Files

- `research/cwf/experiments/exp14_analytic_codec/exp14_analytic_codec.py`
- `research/cwf/experiments/exp14_analytic_codec/results/{cnn,cnn_gauge,analytic,analytic_gauge}_s{42,123,2024}.json`
- `research/cwf/experiments/exp14_analytic_codec/results/exp14_verdict.md`

## Reproducibility

```bash
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp14_analytic_codec.exp14_analytic_codec --condition all --seeds 42 123 2024 --steps 3000
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp14_analytic_codec.exp14_analytic_codec --verdict_only
```
