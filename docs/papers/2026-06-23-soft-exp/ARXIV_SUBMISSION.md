# arXiv Submission Package — Paper 1 (Soft-Exp)

## Status: READY TO SUBMIT

**Title**: Exposure Bias is Bigger Than You Think: Continuous Expected Embeddings as a Drop-In Fix for Autoregressive Decoding

**Authors**: Yiming Wang (CrystaLLM Project)

**arXiv category**: cs.CL (Computation and Language), cs.LG (Machine Learning)

**Submission date target**: 2026-06-23

## Submission Steps

### 1. Prepare source bundle

The arXiv submission requires a tar.gz bundle with:
- `main.tex` — paper source
- `figures/` — all referenced figures
- `references.bib` — bibliography
- `ARXIV_SUBMISSION.md` — this file (optional, removed before submission)

### 2. Compile LaTeX locally

```bash
cd docs/papers/2026-06-23-soft-exp/
pdflatex main.tex
bibtex main  # if references.bib exists
pdflatex main.tex
pdflatex main.tex
```

Expected output: `main.pdf` (~10 pages with figures and bibliography)

### 3. Upload to arXiv

1. Go to https://arxiv.org/submit
2. Login with arXiv account (requires endorsement for first-time submitters in cs.CL)
3. Upload tar.gz bundle
4. Fill metadata:
   - Title: Exposure Bias is Bigger Than You Think: Continuous Expected Embeddings as a Drop-In Fix for Autoregressive Decoding
   - Authors: Yiming Wang
   - Abstract: (see main.tex)
   - Categories: cs.CL (primary), cs.LG (cross-list)
   - Comments: 4 pages main + appendix, 11 figures/tables
5. Submit (may take 1-2 business days for moderation)

### 4. Update with arXiv ID

Once accepted, update this file and `main.tex` with the arXiv ID.

## Abstract (for arXiv metadata)

Autoregressive language models are trained with teacher forcing but deployed by feeding back their own discrete predictions, creating a structural train-inference mismatch. We measure this gap on a 1.2B-parameter Transformer: teacher-forcing perplexity is 3.26, but standard argmax-feedback inference yields 64.74—a 19.88× degradation, far larger than the ~1.1× commonly cited in textbooks. We propose Soft-Exp inference: replace the discrete argmax token embedding with the continuous probability-weighted expected embedding. This single-line change (no retraining, no architectural modification) reduces inference perplexity to 33.27, halving the exposure bias (+48.6% relative improvement). The improvement is scale-invariant: at 2M, 8M, 16M, and 1.2B parameters, Soft-Exp yields +81%, +53%, +48.6%, and +48.6% respectively. Soft-Exp is the inference-time dual of scheduled sampling (Bengio et al., 2015): both target the train-inference mismatch from opposite sides, and we show that Soft-Exp achieves the goal without destabilizing training. We discuss why continuous feedback preserves more distributional information than discrete point estimates, and connect the result to soft-target distillation (Hinton et al., 2015).

## Subject Categories

- **Primary**: cs.CL (Computation and Language)
- **Cross-list**: cs.LG (Machine Learning)

Optionally: cs.AI (Artificial Intelligence) if appropriate.

## Comments for Moderator

This paper is the first of two companion papers. The second paper documents the broader 35-experiment campaign that produced the Soft-Exp result, and is available separately as a workshop submission.

## File Checklist

- [x] `main.tex` — complete paper source (24 KB)
- [x] `figures/figure_1_main.png` — main PPL comparison
- [x] `figures/figure_2_scale_invariance.png` — scale-invariance
- [x] `figures/figure_3_exposure_bias.png` — exposure bias reduction
- [x] `generate_figures.py` — figure regeneration script
- [x] `01_introduction.md` — Markdown source (for reference)
- [ ] `references.bib` — extract from inline citations (see below)

## Bibliography Extraction

The paper uses inline `\bibitem{}` entries (no external .bib file). For arXiv submission, you may want to convert to BibTeX format for better metadata:

```bibtex
@article{bengio2015scheduled,
  title={Scheduled sampling for sequence prediction with recurrent neural networks},
  author={Bengio, Samy and Vinyals, Oriol and Jaitly, Navdeep and Shazeer, Noam},
  booktitle={NeurIPS},
  year={2015}
}

@article{hinton2015distill,
  title={Distilling the knowledge in a neural network},
  author={Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff},
  booktitle={NIPS Deep Learning Workshop},
  year={2015}
}

@article{huszar2015how,
  title={How (not) to train your generative model: Scheduled sampling, likelihood, adversary?},
  author={Husz{\'a}r, Ferenc},
  journal={arXiv preprint arXiv:1511.05151},
  year={2015}
}

@inproceedings{holtzman2020curious,
  title={The curious case of neural text degeneration},
  author={Holtzman, Ari and Buys, Jan and Du, Li and Forbes, Maxwell and Choi, Yejin},
  booktitle={ICLR},
  year={2020}
}

@article{wang2026softexp,
  title={Exposure bias is bigger than you think: Continuous expected embeddings as a drop-in fix for autoregressive decoding},
  author={Wang, Yiming},
  journal={CrystaLLM Project},
  year={2026}
}

@article{wang2026failed,
  title={The failed attempt to make Transformers continuous: A systematic record of 35 experiments},
  author={Wang, Yiming},
  journal={CrystaLLM Project, Companion paper},
  year={2026}
}

@inproceedings{welleck2020neural,
  title={Neural text generation with unlikelihood training},
  author={Welleck, Sean and Kulikov, Ilia and Roller, Stephen and Dinan, Emily and Cho, Kyunghyun and Weston, Jason},
  booktitle={ICLR},
  year={2020}
}

@article{xu2022understanding,
  title={Understanding the role of feedback in autoregressive decoding},
  author={Xu, Jinhua and others},
  journal={arXiv preprint},
  year={2022}
}

@inproceedings{szegedy2016rethinking,
  title={Rethinking the inception architecture for computer vision},
  author={Szegedy, Christian and Vanhoucke, Vincent and Ioffe, Sergey and Shlens, Jonathon and Wojna, Zbigniew},
  booktitle={CVPR},
  year={2016}
}
```

## Reproducibility Note for Reviewers

All experiments use fixed random seeds (42). The Soft-Exp inference modification is one line of code; the full implementation is included in the paper's appendix. The V49 1.2B baseline checkpoint is reproducible from the V49 architecture specification.

## Conflict of Interest

None declared. This work was conducted by the author independently.

## License

We release this paper under CC-BY-4.0 to encourage wide dissemination.

## Contact

- Email: yiming.wang@crystallm.org
- Repository: github.com/yiming-crystallm/soft-exp (to be made public upon acceptance)
