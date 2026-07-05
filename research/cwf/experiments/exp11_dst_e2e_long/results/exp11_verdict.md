# exp11 — DST + End-to-End + 10M bytes + 12000 steps: Convergence Verdict

**Date:** 2026-07-06
**Branch:** cwf-manifesto
**Status:** **FAIL** (per new locked standard). exp10's 3/3-seed best-val advantage and no-rebound property did **not survive** extended training — at 12000 steps, complex wins only 1/3 seeds, and 2/3 complex runs show rebound (final > best). The convergence dimension the user added to the standard caught what the exp10 6000-step window could not see: the complex advantage was real but transient, concentrated in the 5000-6000 window, and does not hold at convergence. Per the user's hard-stop: **Stage B closes. Carry Stage A + DST + "no-rebound stability advantage (transient)" back to v50. No exp12.**

---

## The question this probe was designed to answer

From `exp11_dst_e2e_long.py` docstring — exp10 was reclassified to HOLD because the locked standard's three branches didn't cleanly cover "3/3 seeds + 3/6 windows + no-rebound + still-descending." The user diagnosed the gap: "纯定量指标漏了收敛性维度" (pure-quantitative metrics missed the convergence dimension). The fix was to extend training and add convergence to the standard:

> 配置: 与 exp10 完全相同, 仅延长训练至 12000 步. 3 seed. 不引入任何新变量.
>
> 新锁死标准 (含收敛性维度):
> - PASS: ≥4/6 跨 seed 窗口 且 3/3 seed best-val complex < real 且 complex 线最终 2000 步无反弹
> - FAIL: ≤2/6 跨 seed 窗口 或 ≤1/3 seed best-val 赢 或 complex 线出现反弹且 final 高于 best-val
> - HOLD: 3/6 窗口 但 3/3 seed best-val 赢 且无反弹 → "已收敛但优势不够宽", 归档不追加 exp12
>
> 硬停止: exp11 PASS → 升级可规模化架构. exp11 FAIL 或 HOLD → 带 Stage A + DST + 无反弹稳定性优势 三个贡献归档回 v50, 不追加 exp12.

Window count rescaled to 12 (12000 steps / 1000-step windows), with density-equivalent bars: PASS ≥8/12, FAIL ≤4/12 or seed/rebound conditions, HOLD 6/12.

---

## Setup

- 6 runs: `{complex, real} × {42, 123, 2024} × 12000 steps`. Everything else identical to exp10: DST on both lines, end-to-end (no frozen encoder), 10M bytes train, 100K val, d=32, modes=16, 2 FNO layers, M=64, AdamW lr=3e-4 WD=0.01, batch=32.
- Eval @ [500, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000] (13 points).
- 12 windows of 1000 steps each.
- Convergence check: complex val@12000 ≤ val@10000 (no rebound in last 2000 steps).

**Reproducibility confirmed**: complex s42 and s123 traces at steps 500-6000 are bit-identical to exp10 (s42 val@6000=2.2713 in both; s123 val@6000=2.4328 in both). The 6000→12000 extension is clean — no RNG drift, no code change beyond step count.

---

## Results

### Best-val per seed (complex vs real)

| seed | complex best | @step | real best | @step | Δ (c−r) | complex wins? |
|---|---|---|---|---|---|---|
| 42  | 2.211 | 11000 | **2.208** | 9000 | +0.003 | **✗** (real wins by 0.003) |
| 123 | 1.886 | 11000 | **1.781** | 9000 | +0.106 | **✗** (real wins by 0.106) |
| 2024 | **1.914** | 12000 | 2.156 | 11000 | −0.242 | ✓ |

**1/3 seeds: complex best-val < real best-val.** This is a complete reversal from exp10's 3/3 (mean −0.330). At 12000 steps:
- s42: real caught up — complex 2.211 vs real 2.208, essentially tied (real wins by 0.003).
- s123: real overtook — real hit 1.781 @ 9000, beating complex's 1.886 @ 11000 by 0.106.
- s2024: complex still wins (−0.242), and is the only seed where complex best is at 12000 (still descending).

### 1000-step cross-seed window wins (12 windows)

| window | s42 | s123 | s2024 | all-seeds? |
|---|---|---|---|---|
| [0,1000] | L | **W** | **W** | no |
| [1000,2000] | **W** | **W** | L | no |
| [2000,3000] | **W** | **W** | **W** | **YES** |
| [3000,4000] | **W** | **W** | **W** | **YES** |
| [4000,5000] | **W** | L | **W** | no |
| [5000,6000] | **W** | **W** | **W** | **YES** |
| [6000,7000] | **W** | **W** | **W** | **YES** |
| [7000,8000] | **W** | **W** | L | no |
| [8000,9000] | L | L | **W** | no |
| [9000,10000] | L | L | L | no |
| [10000,11000] | **W** | **W** | L | no |
| [11000,12000] | **W** | **W** | **W** | **YES** |

**Cross-seed winning windows: 5/12.** Per-seed: 9/12, 9/12, 8/12.

The 5/12 cross-seed wins are concentrated in two clusters: [2000-4000] (early-mid training) and [5000-7000]+[11000-12000]. The **[8000-10000] region is a dead zone** — complex loses in 2/3 seeds in [8000,9000] and all 3 in [9000,10000]. This is where real line's best-points live (real s42@9000, real s123@9000, real s2024@11000).

### Convergence dimension (the new criterion)

| seed | val@10000 | val@12000 | Δ (last 2000) | no-rebound? | best | final | final>best? |
|---|---|---|---|---|---|---|---|
| complex s42  | 2.911 | 2.346 | −0.565 | ✓ | 2.211@11000 | 2.346@12000 | **✗ (rebound +0.13)** |
| complex s123 | 2.385 | 2.400 | +0.016 | **✗ (rebound)** | 1.886@11000 | 2.400@12000 | **✗ (rebound +0.51)** |
| complex s2024 | 2.287 | 1.914 | −0.373 | ✓ | 1.914@12000 | 1.914@12000 | ✓ (still descending) |

**Complex no-rebound (last 2000 steps): 2/3.** Complex final > best (rebound): 2/3.

The convergence criterion — the dimension the user added to the standard specifically to catch this — caught the failure. exp10's "3/3 no-rebound" at 6000 steps did not survive to 12000: s42 and s123 both rebound after their best-points (s42 best@11000 then +0.13 to 12000; s123 best@11000 then +0.51 to 12000). Only s2024 is still descending at 12000.

### vs exp10 (6000 steps)

| Metric | exp10 (6k) | exp11 (12k) | Change |
|---|---|---|---|
| seeds complex best < real | 3/3 (mean −0.330) | 1/3 (mean +0.115) | **reversal** |
| complex best-val mean | 2.148 | 2.004 | improved (−0.14) but real improved more |
| real best-val mean | 2.478 | 2.048 | **real improved −0.43 (more than complex)** |
| cross-seed windows | 3/6 | 5/12 | density 50% → 42% |
| complex no-rebound | 3/3 | 2/3 | s123 developed rebound |
| complex final>best | 0/3 | 2/3 | **rebound appeared** |

The critical finding: **real line improved more with extended training than complex did.** Real best-val mean dropped from 2.478 to 2.048 (−0.43 nat); complex from 2.148 to 2.004 (−0.14 nat). The complex advantage at 6000 steps was largely because real hadn't converged yet — real was still in its rebound phase at 6000 (exp10 real s123 had val@6000=3.543, catastrophic). By 9000-11000, real recovered and surpassed complex in 2/3 seeds.

### Rebound analysis (the key diagnostic)

| line | seed | best | @step | final@12000 | rebound? |
|---|---|---|---|---|---|
| complex | 42  | 2.211 | 11000 | 2.346 | YES (+0.13) |
| complex | 123 | 1.886 | 11000 | 2.400 | YES (+0.51) |
| complex | 2024 | 1.914 | 12000 | 1.914 | NO (still descending) |
| real | 42  | 2.208 | 9000 | 2.704 | YES (+0.50) |
| real | 123 | 1.781 | 9000 | 2.480 | YES (+0.70) |
| real | 2024 | 2.156 | 11000 | 2.238 | YES (+0.08) |

**Both lines rebound at 12000 steps.** exp10's "complex 3/3 no-rebound" was a 6000-step artifact — the complex line's rebound was simply delayed, not eliminated. At 12000, complex rebounds in 2/3 seeds (s42 +0.13, s123 +0.51), real rebounds in 3/3. The complex line's rebound is *milder* (smaller Δ) and *later* (best-points at 11000-12000 vs real's 9000-11000), but it is not eliminated.

The exp10 conclusion "end-to-end cures the rebound for the complex line" is **refined**: end-to-end *delays* the complex rebound from ~5000 (frozen-encoder, exp05-09) to ~11000 (end-to-end, exp11), but does not *cure* it. The underlying memorization/distribution-shift pathology (exp23 diagnosis) still applies; it just takes longer to manifest under joint optimization.

---

## Verdict per the new locked standard

| Criterion | Required | Observed | Met? |
|---|---|---|---|
| Seeds complex best < real best | 3/3 | 1/3 | ✗ |
| Cross-seed windows | ≥8/12 | 5/12 | ✗ |
| Complex no-rebound (last 2000) | 3/3 | 2/3 | ✗ |
| Complex final > best (FAIL trigger) | — | 2/3 | **FAIL triggered** |

**Per the new locked standard: FAIL.** Two independent FAIL conditions triggered:
1. `seeds_won ≤ 1` (1/3 ≤ 1).
2. `complex rebound and final > best` (2/3 seeds).

Per the user's hard-stop: **Stage B closes. Carry Stage A + DST + "no-rebound stability advantage (transient)" back to v50. No exp12.**

---

## Key findings

### 1. exp10's complex advantage was real but transient — concentrated in the 5000-6000 window

At 6000 steps (exp10), complex won 3/3 seeds because real was still in its early-rebound phase (real s123 val@6000=3.543, catastrophic). By 9000-11000 (exp11), real recovered and overtook complex in 2/3 seeds. The complex advantage was not a structural superiority in representation quality — it was a *faster early convergence* that real eventually matched. This is the opposite of what exp10's data suggested at 6000 steps, and it is why the convergence dimension the user added was essential.

### 2. The "no-rebound" property was a delay, not a cure

exp10's headline finding ("end-to-end cures the rebound for complex") does not survive to 12000 steps. Complex rebounds in 2/3 seeds (s42 best@11000→+0.13, s123 best@11000→+0.51). The rebound is *delayed* from ~5000 (frozen-encoder) to ~11000 (end-to-end), and *milder* (complex Δ +0.13/+0.51 vs real +0.50/+0.70/+0.08), but it is not eliminated. The exp23 memorization/distribution-shift pathology still applies under joint optimization; it just takes longer (more data passes) to manifest. This refines — but does not refute — exp10's finding: end-to-end + DST makes the complex line *more stable* than frozen-encoder, but not stable enough to claim a cure.

### 3. Real line converges to a lower best-val than complex in 2/3 seeds

Real best-val mean (2.048) is now *below* complex best-val mean (2.004)... wait, complex mean is still lower (2.004 vs 2.048). But per-seed: s42 real wins (2.208 vs 2.211), s123 real wins (1.781 vs 1.886), s2024 complex wins (1.914 vs 2.156). The mean is misleading because complex s2024's −0.242 advantage outweighs the two small real wins (+0.003, +0.106). Per-seed (1/3) is the honest measure, and it favors real. The real line, given enough training, matches or exceeds the complex representation's next-byte prediction quality on this data/config.

### 4. The window dead-zone [8000-10000] reveals where real overtakes

Complex loses in all 3 seeds in [9000,10000] — this is exactly where real's best-points are (real s42@9000, real s123@9000). The complex line has a temporary worse-region in [8000-10000] before recovering at [11000-12000] (where complex wins 2/3 and ties on the 3rd). This non-monotone behavior — complex good early, real better mid-late, complex recovers late — is a more complex optimization story than "complex is better."

### 5. DST remains a valid structural contribution regardless of Stage B's fate

DST was validated in exp09 (6/6 windows vs exp06 real, frozen-encoder) and exp10 (consistent per-seed 5/6 at 6000 steps). exp11 does not re-test DST vs FFT — it tests DST+complex vs DST+real at convergence. The DST contribution (fixing wrap-around) is orthogonal to the complex-vs-real question and stands. The failure here is "complex does not beat real at convergence," not "DST doesn't work."

---

## What this closes / does not close

**Closes**:
- "End-to-end + DST cures the Stage-B rebound for the complex line." — **Falsified at 12000 steps.** The rebound is delayed (5000→11000) and milder, but not eliminated. exp10's 6000-step "no-rebound" was a transient.
- "Complex representation provides a scalable next-byte prediction advantage over real." — **Falsified at convergence.** 1/3 seeds at 12000 steps; real matches or exceeds in 2/3. The complex advantage at 6000 was faster-early-convergence, not structural superiority.
- "Stage B can be scaled to a deployable architecture." — **No.** Even with DST + end-to-end + 10M bytes + 12000 steps, complex does not stably beat real. The CWF Stage-B prediction line is closed.

**Does not close**:
- **Stage A (Wave Tokenizer reconstruction).** Untouched by exp10/exp11. Still the strongest CWF result (complex 0.542 vs real-d64 0.893, 2.8× advantage, survived 4-challenge audit). Remains the durable CWF contribution.
- **DST as a structural fix.** Validated in exp09 (frozen-encoder, 6/6 windows). Not re-tested in exp11 (both lines use DST). The wrap-around pathology DST fixes is real; DST's contribution is orthogonal to the complex-vs-real question that exp11 settled.
- **Whether a different architecture (not frozen-encoder FNO, not end-to-end FNO) could make complex stably win.** Out of scope. 30+ experiments across 4 architectures (exp04 hand-coded Gaussian, exp05-09 frozen-encoder CNN, exp09 DST variants, exp10-11 end-to-end DST) have now been tested. The complex advantage is consistently transient or marginal. A fundamentally different evolution mechanism (not FNO) is untested but is a new research direction, not a continuation of this line.

---

## Implications for the CWF line

**Stage B is closed.** The user's hard-stop is invoked: FAIL, no exp12. This is the right call — the convergence dimension caught a failure that the exp10 6000-step window could not see, and 4 successive experiments (exp07 DEAD, exp08 double-FALSIFIED, exp10 reclassified-HOLD, exp11 FAIL) have now confirmed that the complex wave field's next-byte prediction advantage does not scale to convergence on byte text.

**What is carried back to v50:**
1. **Stage A (Wave Tokenizer reconstruction)** — the cleanest CWF signal (complex 2.8× better than matched-capacity real, survived 4-challenge audit). A legitimate research finding on complex wave fields as text representations.
2. **DST (absorbing boundary)** — validated structural fix for FFT wrap-around (exp09, 6/6 windows). Orthogonal to Stage B's fate; useful for any future FNO-based work.
3. **"No-rebound stability advantage (transient)"** — complex under end-to-end+DST delays and milder-izes the rebound (11000 vs 5000, +0.13/+0.51 vs real's +0.50/+0.70/+0.08). Not a cure, but a real stability property worth recording. It may matter for architectures where the rebound is the binding constraint.

**v50 mainline unchanged** (V49 baseline + Soft-Exp inference, exp29 +48.6% PPL). CWF returns to research charter status, with Stage A + DST as documented contributions and Stage B closed after 11 experiments.

---

## Files

- `research/cwf/experiments/exp11_dst_e2e_long/exp11_dst_e2e_long.py` — 12000-step extension + new locked-standard verdict (convergence dimension).
- `research/cwf/experiments/exp11_dst_e2e_long/results/{complex,real}_s{42,123,2024}.json` — 6 run traces.
- `research/cwf/experiments/exp11_dst_e2e_long/results/exp11_verdict.md` — this report.

## Reproducibility

```bash
# From D:/CrystaLLM (CUDA). 6 runs, ~18 min total (10M-byte load + 6× ~270s).
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp11_dst_e2e_long.exp11_dst_e2e_long --mode complex --seeds 42 123 2024 --steps 12000
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp11_dst_e2e_long.exp11_dst_e2e_long --mode real    --seeds 42 123 2024 --steps 12000
.venv/Scripts/python.exe -u -m research.cwf.experiments.exp11_dst_e2e_long.exp11_dst_e2e_long --verdict_only
```

## Recommended decision

**Record exp11 as FAIL. Stage B closes.** The convergence dimension the user added to the standard was the right call — it caught a failure that exp10's 6000-step window could not see. The complex line's next-byte prediction advantage does not survive to convergence: 1/3 seeds at 12000 steps, 2/3 complex runs rebound, real matches or exceeds in 2/3 seeds. The "no-rebound" property exp10 celebrated was a delay (5000→11000), not a cure.

Per the user's hard-stop: carry Stage A + DST + "no-rebound stability advantage (transient)" back to v50. No exp12. CWF returns to research charter status. The 11-experiment Stage-B arc (exp04→exp11) is documented; the durable contributions are Stage A (representation) and DST (structural), not Stage B (prediction).

**This is a clean, honest close.** The user's standard design — especially the convergence dimension added for exp11 — prevented a false PASS based on a transient 6000-step advantage. The data spoke, and the standard listened.
