# Paper Polish Notes (Known Issues & Improvements)

This document tracks issues identified during the writing process and items to address in the next polish cycle.

## Paper 1 (Soft-Exp): `docs/papers/2026-06-23-soft-exp/main.tex`

### Strengths
- Clear narrative: 20× exposure bias → +48.6% fix
- Concrete numbers throughout
- Strong theoretical connection (information theory, knowledge distillation)
- Three high-quality figures
- Comprehensive appendix

### Known Issues / TODO

1. **References**: Currently uses inline `\bibitem{}`. Should convert to external `references.bib` for better metadata handling. See `ARXIV_SUBMISSION.md` for the BibTeX entries.

2. **Algorithm pseudocode**: The current Method section uses Python listings, but a formal algorithm block would be cleaner. Add in next revision:
```latex
\begin{algorithm}
\caption{Soft-Exp Inference}
\begin{algorithmic}
\For{each decoding step $t$}
    \State $\mathbf{z}_t = \text{head}(f_\theta(\mathbf{h}_{t-1}, \mathbf{e}_{t-1}))$
    \State $\mathbf{p}_t = \text{softmax}(\mathbf{z}_t)$
    \State $\mathbf{e}_t = \mathbf{p}_t^\top \mathbf{E}$ \Comment{Soft-Exp feedback}
    \State $\hat{x}_t = \arg\max \mathbf{p}_t$ \Comment{Output token}
    \State $\mathbf{h}_t = f_\theta(\mathbf{h}_{t-1}, \mathbf{e}_t)$
\EndFor
\end{algorithmic}
\end{algorithm}
```

3. **Statistical significance**: Add confidence intervals or significance test for the +48.6% improvement. We have variance estimates from smaller-scale runs ($\pm 2\%$), so the 1.2B result is $\sim 24\sigma$ significant.

4. **Additional experiments**: A 5-beam baseline comparison would strengthen the paper. Currently we mention it as future work; should include even quick comparison.

5. **Limitation section**: Should explicitly acknowledge that the residual 10× bias is not eliminated and cannot be by inference-only methods.

6. **Discussion of failure modes**: The companion paper covers 5 failure modes; Paper 1 should briefly mention these to motivate "why no training-side intervention".

## Paper 2 (Failure History): `docs/papers/2026-06-23-35-failures/main.tex`

### Strengths
- Complete coverage of 35 experiments
- Strong theoretical contribution (closure theorem)
- Clear narrative arc (5 sub-phases of CMT, pivot to CWF, decisive experiment)
- Honest reporting of negative results
- Three independent evidence streams

### Known Issues / TODO

1. **Length**: Currently $\sim$30 pages. May need to trim to 20 pages for workshop submission. Cuttable sections:
   - Section 2 (Wave Vision) can be shortened by 50%
   - Appendix C (Timeline) can be compressed to 1 page

2. **§2 Wave Vision Theory**: Currently brief. Should add:
   - Brief discussion of complex-valued neural networks (cite Nitta, Trabelsi)
   - Brief discussion of Neural ODEs (cite Chen 2018)
   - Brief discussion of diffusion language models (cite Austin, Lou)

3. **Failure modes appendix**: Currently detailed but could be more compact. Use bullet points for each failure mode.

4. **§5 Closure Argument**: The proof is informal (sketch level). A more rigorous proof would strengthen the paper. However, this is acceptable for a workshop paper.

5. **Statistical rigor**: Some experiments lack multiple seeds. Acknowledged in Appendix B but could be discussed more prominently in main text.

6. **CWF Manifesto integration**: Section 4 references the CWF Manifesto but does not reproduce it. This is intentional (separate document), but should clarify the relationship.

## Cross-Paper Polish

1. **Citation consistency**: Both papers cite each other as `\citep{wang2026softexp}` and `\citep{wang2026failed}`. Make sure these are consistent and well-defined in both bibliographies.

2. **Terminology alignment**:
   - "Patchwork" vs "patching" vs "continuous-component patching" - use consistently
   - "Closure" vs "closed architecture" vs "closure property" - define once, use consistently
   - "Continuous feedback" vs "soft feedback" vs "continuous expected embedding" - prefer "continuous expected embedding" for precision

3. **Numerical consistency**:
   - Both papers should report identical Soft-Exp numbers (PPL 64.74 vs 33.27, +48.6%)
   - Both papers should report identical CWF Phase 1 numbers (CWF EPT=2, Oracle=100)

4. **References**: 
   - Both papers should cite Bengio 2015 (Scheduled Sampling)
   - Both papers should cite Hinton 2015 (Knowledge Distillation)
   - Both papers should cite Chen 2018 (Neural ODEs)
   - Paper 2 should additionally cite the 35 experiment logs

## Next Steps After User Receives This Document

1. **Compile both .tex files locally**:
   ```bash
   cd docs/papers/2026-06-23-soft-exp
   pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
   cd ../2026-06-23-35-failures
   pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
   ```

2. **Fix any compile errors**: Common issues:
   - Missing `\usepackage{}` for a custom command
   - Bibliography style mismatch (we use `\bibliographystyle{plain}` or similar)
   - Figure path issues (we use `figures/` subdirectory which should work)

3. **Run spell check and grammar review** on both .tex files

4. **Submit Paper 1 to arXiv** (user action required)

5. **Watch for ICLR 2027 Workshop CFPs** (October 2026)

## Time Estimates

- LaTeX compilation verification: 30 min
- Bibliography cleanup: 30 min
- Spell/grammar check: 1 hour
- Algorithm pseudocode addition (Paper 1): 30 min
- Trim Paper 2 to 20 pages: 2 hours
- Final review: 1 hour

**Total polish time**: ~5 hours

This can be done over 1-2 days before submission deadlines.
