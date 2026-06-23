# Paper 2: The Failed Attempt to Make Transformers Continuous
## Complete Document Structure

**Title**: *The Failed Attempt to Make Transformers Continuous: A Systematic Record of 35 Experiments*

**Status**: ✅ Draft v1.0 (2026-06-23) — Complete, ready for LaTeX conversion

**Target Venue**: NeurIPS Workshop on "What Never Worked and Why" / ICLR Workshop on Meta-Research / arXiv

---

## Document Map (All Sections Complete)

| File | Section | Content | Status |
|---|---|---|---|
| `05_introduction.md` | §1 | Motivation + 3 evidence streams + contributions | ✅ |
| `02_cmt_timeline.md` | §3 | CMT Campaign: 30 experiments in 5 sub-phases | ✅ |
| `06_pivot_decision.md` | §4 | The CWF Pivot: Closure Conjecture + 6-component architecture | ✅ |
| `03_closure_argument.md` | §5 | Mathematical argument: closure is necessary but not sufficient | ✅ |
| `04_cwf_lorenz.md` | §6 | CWF Phase 1: Lorenz decisive experiment | ✅ |
| `07_soft_exp_survivor.md` | §7 | The Soft-Exp Survivor: inverse of the closure argument | ✅ |
| `08_conclusion.md` | §8 | Conclusion + CWF Manifesto as research challenge | ✅ |
| `00_timeline_table.md` | (Reference + Appendix C) | Complete table of all 35 experiments | ✅ |
| `01_failure_modes.md` | (Appendix A) | 5 recurring failure modes | ✅ |
| `09_reproducibility.md` | Appendix B | Reproducibility, hardware, hyperparameters | ✅ |
| `README.md` | (Navigation) | This file | ✅ |

---

## Final Paper Structure

```
Title: The Failed Attempt to Make Transformers Continuous
       A Systematic Record of 35 Experiments

§1. Introduction                                              ✅ 12 KB
    - Motivation: discrete Transformer limitations
    - Our method: systematic experimental campaign  
    - Summary: 30+ rounds of failure, 1 survivor
    - Three independent evidence streams
    - The Closure Argument (preview)
    - Contributions
    - Outline + note on tone

§2. The Continuous Wave Function Vision                         (existing CWF Manifesto material, to be integrated)
    - Why continuous representations are attractive
    - Theoretical foundations: complex, KAN, Lie group, Born
    - The intuition that motivated 35 experiments

§3. The CMT Campaign (30 Experiments)                          ✅ 12 KB
    - 3.1 Pre-CMT Explorations (Exp 1-5)
    - 3.2 CMT Main: 10 Knife Combinations (Exp 6-15)
    - 3.3 Engineering Audit: 15 Bug-Fix Rounds (Exp 16-25)
    - 3.4 Soft-Exp Survivor: 4 Scale Validations (Exp 26-29)
    - 3.5 The Final Test: Training Soft-Exp (Exp 30)
    - 3.6 Lessons from Phase 3

§4. The CWF Pivot                                             ✅ 11 KB
    - 4.1 Why Patchwork Was a Dead End
    - 4.2 The Pivot Decision
    - 4.3 The Closure Conjecture
    - 4.4 The Six-Component Closed Architecture
    - 4.5 The Prototype (Phase 0)
    - 4.6 The Three Critical Conjectures
    - 4.7 Why Phase 1 Was the Right Next Step

§5. Why "Patching" Fails: A Mathematical Argument               ✅ 12 KB
    - 5.1 The Closure Property (definition + lemma + theorem)
    - 5.2 Why Soft-Exp Avoids Drift
    - 5.3 The Closure Verdict
    - 5.4 What a Truly Closed Architecture Requires
    - 5.5 Implications for the Field

§6. CWF Phase 1: The Decisive Experiment                       ✅ 12 KB
    - 6.1 Why Lorenz
    - 6.2 The Three Models
    - 6.3 Training Protocol
    - 6.4 Results (1-step + 100-step rollout + EPT)
    - 6.5 Diagnostic: Why CWF Failed on Rollout
    - 6.6 The Verdict
    - 6.7 What Would Save CWF
    - 6.8 What This Proves and Does Not

§7. Where Continuity Survived: The Soft-Exp Survivor           ✅ 11 KB
    - 7.1 What Soft-Exp Is
    - 7.2 The Empirical Result
    - 7.3 Why Soft-Exp Avoids the Failure Mode
    - 7.4 Why Soft-Exp Training Failed
    - 7.5 The Unique Position of Soft-Exp
    - 7.6 The Soft-Exp Limitation
    - 7.7 Connection to Knowledge Distillation
    - 7.8 What This Means for Future Work

§8. Conclusion and Research Challenge                          ✅ 9 KB
    - 8.1 Summary of 35 Experiments
    - 8.2 The Three Lessons
    - 8.3 The CWF Manifesto as Research Challenge
    - 8.4 What We Recommend
    - 8.5 Limitations of This Paper
    - 8.6 A Note on the Companion Paper
    - 8.7 Final Acknowledgments

§Appendix A. Cross-Cutting Failure Modes                       ✅ 15 KB
    - FM1: Training-Inference Distribution Drift
    - FM2: Closure Violation in Patchwork Architectures
    - FM3: Architectural Absence of ODE Evolution
    - FM4: Tokenization Bottleneck on Continuous Data
    - FM5: Optimizer-Landscape Mismatch on Complex Losses

§Appendix B. Reproducibility                                  ✅ 8 KB
    - Code Repositories
    - Data
    - Hardware
    - Hyperparameters (per experiment group)
    - Per-Experiment Details
    - Reproducibility Checklist
    - Random Seeds
    - Variance and Statistical Significance
    - Compute Cost
    - Software Versions

§Appendix C. Timeline Table                                    ✅ 16 KB
    - 35 experiments with hypothesis/setup/result/verdict
    - Causal diagram
    - Three independent evidence streams
    - What "Soft-Exp Survived" tells us
    - Failure mode counts
```

**Total content**: ~120 KB markdown, equivalent to approximately 25-30 pages of formatted text.

---

## Reading Order

For reviewers (in priority order):

1. Read §1 (Introduction) for the high-level message and contributions
2. Skim §3 (CMT timeline) for empirical evidence
3. Read §5 (Closure argument) for the theoretical contribution
4. Read §6 (CWF Lorenz) for the decisive experiment
5. Read §7 (Soft-Exp) for the single positive result
6. Read §8 (Conclusion) for recommendations and the CWF research charter

For future researchers considering this path:

1. Read §6 first (CWF failed despite closure---avoid this trap)
2. Read §5 for why closure is necessary but not sufficient
3. Read §6.7 for what would need to be built (the CWF-with-ODE roadmap)
4. Decide whether to commit to the 1--2 person-year investment

For practitioners interested in the Soft-Exp method:

1. Read Paper 1 (Soft-Exp paper) for the practical method
2. Read §7 of this paper for the theoretical motivation
3. Read Appendix B for reproducibility details

---

## Key Statistics

- **Total experiments**: 35 (30 CMT/Soft-Exp + 5 CWF)
- **Verdicts**: 1 STRONG PASS, 4 PASS, 1 PARTIAL, 4 INCONCLUSIVE, 25 FAIL
- **Total wall-clock time**: ~3 months (CMT) + 2 weeks (CWF)
- **Single survivor**: Soft-Exp inference (1 line of code, no retraining, +48.6% PPL)
- **Three independent evidence streams** converge on the same conclusion

---

## Companion Paper

This paper is the long-form record. A companion paper documents the single positive result:

> **Paper 1**: "Exposure Bias is Bigger Than You Think: Continuous Expected Embeddings as a Drop-In Fix for Autoregressive Decoding"
> Location: `docs/papers/2026-06-23-soft-exp/main.tex` (24 KB, complete NeurIPS-style LaTeX)

Together, the two papers tell a complete story:
- **Paper 1**: the surviving fix (1 line of code, +48.6% PPL)
- **Paper 2**: the systematic context that produced the fix (35 experiments, closure argument, CWF research charter)

---

## Next Steps

1. ✅ Convert Paper 1 markdown to LaTeX (DONE in `main.tex`)
2. ⏳ Convert Paper 2 markdown to LaTeX (1 day, mechanical)
3. ⏳ Polish both papers (1 week)
4. ⏳ Submit Paper 1 to ICLR workshop / arXiv (1 week)
5. ⏳ Submit Paper 2 to NeurIPS workshop on What Never Worked / arXiv (1 month)

The two papers are designed to be complementary and can be submitted independently or together.
