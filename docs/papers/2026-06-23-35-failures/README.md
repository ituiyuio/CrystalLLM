# Paper 2: The Failed Attempt to Make Transformers Continuous
## Complete Document Structure

**Title**: *The Failed Attempt to Make Transformers Continuous: A Systematic Record of 35 Experiments*

**Status**: Draft v0.1 (2026-06-23)

**Target Venue**: NeurIPS Workshop on "What Never Worked and Why" / ICLR Workshop on Meta-Research

---

## Document Map

| File | Section | Content |
|---|---|---|
| `00_timeline_table.md` | (Reference) | Complete table of all 35 experiments |
| `01_failure_modes.md` | (Cross-cutting) | 5 recurring failure modes with evidence |
| `02_cmt_timeline.md` | §3 | CMT Campaign: 30 experiments in 5 sub-phases |
| `03_closure_argument.md` | §5 | Mathematical argument: closure is necessary but not sufficient |
| `04_cwf_lorenz.md` | §6 | CWF Phase 1: Lorenz decisive experiment |

## Suggested Paper Structure

```
Title: The Failed Attempt to Make Transformers Continuous
       A Systematic Record of 35 Experiments

§1. Introduction (2 pages)
    - Motivation: discrete Transformer limitations
    - Our method: systematic experimental campaign
    - Summary of results: 30+ rounds of failure, 1 survivor

§2. The Continuous Wave Function Vision (3 pages)
    - Theory: complex representations, closure, KAN/Lie/Born
    - Why it was reasonable to try
    - The intuition that motivated 35 experiments

§3. The CMT Campaign (8 pages)
    [from 02_cmt_timeline.md]
    - 3.1 Pre-CMT explorations
    - 3.2 CMT Main: 10 knife combinations
    - 3.3 Engineering Audit: 15 bug-fix rounds
    - 3.4 Soft-Exp Survivor: the 4 scale validations
    - 3.5 The Final Test: why training-side continuous fails
    - 3.6 Lessons

§4. CWF Manifesto and Pivot (2 pages)
    - Why we pivoted from patching to redesign
    - The Closure Conjecture
    - 6-component closed architecture

§5. Why "Patching" Fails: A Mathematical Argument (3 pages)
    [from 03_closure_argument.md]
    - 5.1 The Closure Property (definition, lemma, theorem)
    - 5.2 Why Soft-Exp Avoids Drift
    - 5.3 The Closure Verdict
    - 5.4 What a Truly Closed Architecture Requires
    - 5.5 Implications for the Field

§6. CWF Phase 1: The Decisive Experiment (4 pages)
    [from 04_cwf_lorenz.md]
    - 6.1 Why Lorenz
    - 6.2 The Three Models
    - 6.3 Training Protocol
    - 6.4 Results (1-step + rollout + EPT)
    - 6.5 Diagnostic: Why CWF Failed on Rollout
    - 6.6 The Verdict
    - 6.7 What Would Save CWF
    - 6.8 What This Proves and Does Not

§7. Where Continuity Survived: Soft-Exp (2 pages)
    - The single positive result
    - Why inference-only works (converse of Lemma 5.2)
    - Connection to the broader closure argument

§8. Conclusion and Research Challenge (2 pages)
    - Summary of 35 experiments
    - The CWF Manifesto as research challenge
    - Acknowledgments of negative results

§Appendix A. Failure Modes Cross-Cutting Analysis
    [from 01_failure_modes.md]

§Appendix B. Experiment Logs and Reproducibility
    - Links to all 35 experiment JSON files
    - Hardware/software specifications
    - Hyperparameter tables

§Appendix C. Timeline Table
    [from 00_timeline_table.md]
```

## Reading Order

For reviewers:

1. Read §1 (Introduction) for the high-level message
2. Skim §3 (CMT timeline) for the empirical evidence
3. Read §5 (Closure argument) for the theoretical contribution
4. Read §6 (CWF Lorenz) for the decisive experiment
5. Read §7 (Soft-Exp) for the single positive result
6. Appendix B for reproducibility details

For future researchers considering this path:

1. Read §6 first (CWF failed despite closure---avoid this trap)
2. Read §5 for why closure is necessary but not sufficient
3. Read §6.7 for what would need to be built
4. Decide whether to commit to the 1--2 person-year investment

## Key Statistics

- **Total experiments**: 35 (30 CMT/Soft-Exp + 5 CWF)
- **Verdicts**: 1 STRONG PASS, 4 PASS, 1 PARTIAL, 4 INCONCLUSIVE, 25 FAIL
- **Total wall-clock time**: ~3 months (CMT) + 2 weeks (CWF)
- **Single survivor**: Soft-Exp inference (1 line of code, no retraining)
- **Three independent evidence streams** converge on the same conclusion

## Companion Paper

This paper is the long-form record. A companion paper documents the single positive result:

> Paper 1: "Exposure Bias is Bigger Than You Think: Continuous Expected Embeddings as a Drop-In Fix for Autoregressive Decoding"

Together, the two papers tell a complete story:
- Paper 1: the surviving fix (1 line of code)
- Paper 2: the systematic context that produced the fix (35 experiments)
