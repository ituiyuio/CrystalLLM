# Workshop Submission Deadlines (2026-2027)

This document tracks submission deadlines for relevant venues for Paper 1 and Paper 2.

## Paper 1 (Soft-Exp): "Exposure Bias is Bigger Than You Think"

Recommended venues:
- **ICLR 2027 Workshop on Practical Deep Learning** (typically January deadline)
- **ACL 2027 Findings** (typically May deadline)
- **arXiv** (immediate, no deadline)
- **NAACL 2027 SRW (Student Research Workshop)** (typically October deadline)
- **ICML 2027 Workshop on Knowledge Discovery** (typically May deadline)

### arXiv (IMMEDIATE)

- **Submission**: Anytime
- **Format**: tar.gz with .tex source
- **Endorsement**: Required for first-time cs.CL submitters
- **Approval time**: 1-2 business days
- **Recommended path**: Prepare source bundle, submit immediately, get priority date

## Paper 2 (Failure History): "The Failed Attempt to Make Transformers Continuous"

Recommended venues:
- **NeurIPS 2026 Workshop on "What Never Worked and Why"** (if it exists - common in NeurIPS)
- **ICML 2026 Workshop on "Negative Results"** (some years have this)
- **ICLR 2027 Workshop on Meta-Research** (typically January deadline)
- **NeurIPS 2026 Workshop on "Research Engineering"** (if it exists)
- **arXiv** (immediate)

### Specific 2026 Deadlines (To Be Verified)

Most 2026 deadlines have already passed as of June 2026. The next round of deadlines will be:

| Venue | Typical Deadline | 2027 Conference Date |
|---|---|---|
| ICLR Workshops | January 2027 | April/May 2027 |
| ICML Workshops | May 2027 | July 2027 |
| NeurIPS Workshops | September 2027 | December 2027 |
| ACL/EMNLP | May 2027 | August 2027 |

## Action Items

### Immediate (this week)

- [x] Prepare arXiv submission package (Paper 1)
- [x] Write Paper 2 main.tex (LaTeX)
- [ ] Compile both LaTeX files locally (verify no errors)
- [ ] Submit Paper 1 to arXiv (requires user account)

### Short-term (this month)

- [ ] Polish Paper 1 based on internal review
- [ ] Polish Paper 2 based on internal review
- [ ] Submit Paper 2 to arXiv (companion paper)

### Medium-term (next 6 months)

- [ ] Watch for ICLR 2027 Workshop CFPs (typically October 2026)
- [ ] Submit Paper 1 to ICLR Workshop on Practical DL (January 2027 deadline)
- [ ] Submit Paper 2 to ICLR Meta-Research Workshop (January 2027 deadline)
- [ ] Submit Paper 2 to NeurIPS 2027 Workshop on What Never Worked (September 2027 deadline)

### Long-term (next 12 months)

- [ ] Consider NeurIPS 2027 main track for Paper 1 (May 2027 deadline)
  - Risk: Soft-Exp is engineering, not novelty
  - Mitigation: Frame as "empirical study of exposure bias" + "drop-in fix"
- [ ] Consider JMLR for Paper 2 (continuous submission)
  - Risk: Negative results may be seen as low-impact
  - Mitigation: Frame as "systematic record of closure failures"

## Notes on Venue Selection

**Why workshop over main track for Paper 2?**

The failure-history paper is a negative result, which typically scores poorly in main-track review. Workshops on "What Never Worked" or "Meta-Research" specifically welcome such contributions. The audience is also more receptive.

**Why main track could work for Paper 1?**

Soft-Exp has a $+48.6\%$ improvement at 1.2B scale with no retraining. This is a clear, measurable contribution. The main-track risk is that reviewers may see it as incremental (one-line inference change). The mitigation is to frame the contribution as:
1. The empirical quantification of exposure bias at scale (20$\times$, not 1.1$\times$)
2. The scale-invariance result (rare in LM research)
3. The connection to information-theoretic principles (soft targets)

## Status as of 2026-06-23

- ✅ Paper 1: Markdown draft (14 KB) + LaTeX main.tex (24 KB) + 3 figures
- ✅ Paper 2: Markdown drafts (136 KB) + LaTeX main.tex (50 KB)
- ✅ arXiv submission package (Paper 1) ready
- ⏳ LaTeX compilation (requires pdflatex on user's machine)
- ⏳ arXiv submission (requires user action)
- ⏳ Workshop submission (waiting for CFPs)

## Next Concrete Actions

1. User compiles both main.tex locally with pdflatex (verify no errors)
2. User submits Paper 1 to arXiv via https://arxiv.org/submit
3. Watch for ICLR 2027 Workshop CFP (October 2026)
4. Prepare presentation slides (15 min talk) for workshop use
