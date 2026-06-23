# §3 The CMT Campaign (30 Experiments)

This section narrates the full 30-round CMT (Continuous Manifold Transformer) campaign. We organize it into five sub-phases, each characterized by a refined hypothesis based on the previous phase's failure.

## 3.1 Pre-CMT Explorations (Exp 1-5)

Before committing to the CMT architecture, we conducted five exploratory experiments to scope the design space. These experiments did not test the CMT hypothesis directly but helped us rule out several alternative approaches.

\textbf{Exp 1 (Mamba3/SSD)} tested whether Mamba-style selective state-space models could replace attention. The hypothesis was that SSMs' continuous-time dynamics might naturally avoid the discrete-tokenization issues of Transformers. Result: perplexity was higher than the baseline Transformer; we abandoned this direction as it did not directly address the "continuous representation" question.

\textbf{Exp 2 (Complex KAN)} tested Kolmogorov-Arnold Networks with complex-valued weights as an alternative to GELU MLPs. KAN's theoretical parameter efficiency was attractive, and the complex extension added a phase dimension that we hoped could encode richer features. Result: complex KAN was strictly worse than GELU MLP in our setting, with no path forward.

\textbf{Exp 3 (FP8 mixed precision)} tested FP8 forward, FP32 backward training. This was orthogonal to the continuous-vs-discrete question; we ran it as a sanity check on training infrastructure.

\textbf{Exp 4 (8-bit AdamW)} tested 8-bit optimizer states (BitsAndBytes). Again orthogonal; we ran it to enable larger model experiments.

\textbf{Exp 5 (Curriculum learning)} tested easy-to-hard sample ordering. No improvement was observed.

The lesson from Phase 3.1: we should not pursue Mamba-style alternatives or KAN-style activation replacements independently. They do not address the fundamental question of whether Transformers can be made continuous.

## 3.2 CMT Main Campaign (Exp 6-15)

The first round of CMT experiments combined three "knives" into a single architecture:
\begin{itemize}
    \item \textbf{Knife 1}: KAN-based complex-valued FFN (replace GELU MLP)
    \item \textbf{Knife 2}: Complex-valued attention (replace standard softmax attention)
    \item \textbf{Knife 3}: Lie group rotation for position encoding (replace RoPE)
\end{itemize}

We tested these in combinations to identify which (if any) could survive training without degrading perplexity.

\textbf{Exp 6 (CMT-FFN only)}: we replaced only the FFN, leaving attention and position encoding standard. PPL rose from baseline 2.8 to 4.5. This was the first indication that complex-valued components were a liability.

\textbf{Exp 7 (CMT-full sanity)}: we combined all three knives. At 4k training steps, the model collapsed to a memorizer (PPL < 1.0). We suspected a bug; debugging began.

\textbf{Exp 8 (CMT-full refined)}: we fixed several bugs in complex multiplication and Lie algebra parameterization. The model still collapsed to a memorizer by 8k steps.

\textbf{Exp 9 (Baseline rerun)}: we re-ran the baseline to confirm it was stable. It was. The failure was specific to CMT.

\textbf{Exp 10-13} tested various surgical fixes: complex-softmax approximation, complex multiplication correctness, Lie algebra initialization, and removing context-aware position encoding. Each individual fix yielded marginal improvement, but the model still collapsed to memorization at 12-15k steps.

\textbf{Exp 14 (no context PE)}: removing the context-aware component of position encoding yielded a brief improvement (PPL 32.58 at peak), but the model still fell to memorization by 8k steps.

\textbf{Exp 15 (CMT-full v2)}: combining all fixes from Exp 10-14, the model trained for 12k steps before collapsing. Peak PPL was 1.01, identical to memorization.

\textbf{Diagnostic insight}: Exp 14's peak PPL of 32.58 was the highest the CMT architecture ever achieved. After 8k steps, PPL would drop to memorization values. We hypothesized this was a \emph{phase transition}: the model learns the training distribution's "average behavior" first, then sharpens into memorization.

## 3.3 CMT Engineering Audit (Exp 16-25)

Faced with the persistent memorization pattern, we launched a systematic audit to determine whether the failure was due to bugs or structural properties.

\textbf{Exp 16 (CMT-clean)}: we re-implemented CMT from scratch with all known bugs fixed. After 30k training steps, validation PPL reached 1.0097 (memorizer). Even a "bug-free" CMT failed. This was a turning point: we could no longer attribute the failure to implementation errors.

\textbf{Exp 17 (phase transition diagnosis)}: we trained three CMT variants at different learning rates, evaluating at 1k, 2k, 4k, and 8k steps. Two variants (A0, A2) collapsed to memorization by 4k steps. The third (A1, LR $3 \times 10^{-5}$) showed partial learning but was still underfit at 8k steps. The phase transition was real and LR-driven.

\textbf{Exp 18 (A1 long training)}: we extended A1 training to 8k steps. Four checkpoints all showed underfitting (val\_ppl 17.4 $\to$ 11.8, still decreasing). A1 was not converged; it had been misdiagnosed in Exp 17.

\textbf{Exp 19 (5k sanity)}: we ran a small-data sanity test (2M parameters, 2k samples). Val PPL dropped from 33 to 18 (a 38.5\% reduction), but the model also exhibited repetition (3/6 prompts) and zero coherence. The "dual-phase learning curve" was confirmed: initial learning, then collapse.

\textbf{Exp 20 (BPE sanity)}: switching from character-level to BPE tokenization (vocab=4100) yielded the first LM signal in the CMT campaign: coherence 0/6 $\to$ 2/6, bits-per-character 4.17 $\to$ 3.32. The BPE tokenizer seemed to help.

\textbf{Exp 21 (Baseline + BPE 5k)}: we ran a non-CMT baseline at the same scale. The baseline achieved coherence 1/6 at 5k steps. The CMT-BPE comparison at 5k was inconclusive.

\textbf{Exp 22 (CMT + BPE 16k)}: extending CMT+BPE to 16k steps yielded a surprising result: validation PPL \emph{rebounded} from 786 to 3860 (5$\times$ increase), while coherence rose from 0 to 5/6. We initially interpreted this as "structural learning" but suspected a memorization illusion.

\textbf{Exp 23 (CMT + BPE 10k + 16k)}: we re-ran the same architecture with $5\times$ more data and $5\times$ fewer epochs. The rebound disappeared (ratio $3.90 \to 0.00$), and val PPL dropped to 126. \textbf{Exp 22 was a memorization illusion}: 64 epochs over 2k samples caused the model to memorize specific sequences.

\textbf{Exp 24 (Cayley PE isolation)}: we isolated the Cayley position encoding from the rest of CMT, comparing it to RoPE and NoPE at matched parameter counts. Cayley PE was not better than RoPE, but not worse either. This was a partial success: the Cayley PE alone did not break the model.

\textbf{Exp 25 (CMT + BPE + 10k + 32k)}: a final CMT test with the best-known configuration. CMT reached PPL 1.40 at 12k (memorizer), while baseline reached PPL 50.92 at 32k (real LM). \textbf{CMT\_DEAD\_FINAL}.

\section{Summary of Phase 3.2--3.3}

Across 20 rounds of CMT experimentation, we observed:
\begin{itemize}
    \item Persistent memorization at 8--12k steps across all configurations.
    \item The phase transition was real and LR-driven, not architecture-specific.
    \item Even a bug-free CMT implementation (Exp 16) failed.
    \item BPE tokenization helped somewhat but did not save the architecture.
\end{itemize}

The CMT campaign confirmed that patching continuous components (KAN, complex attention, Cayley PE) onto a standard autoregressive Transformer does not work. The architecture either fails to learn (underfit) or collapses to memorization. We documented this as the \textbf{CMT\_DEAD\_FINAL} verdict and pivoted to the next strategy.

\section{3.4 The Soft-Exp Survivor (Exp 26-29)}

The pivot was not architectural but philosophical: instead of asking "how do we add continuous components to Transformers?", we asked "can we use continuous \emph{feedback} signals without changing the architecture?".

The answer was Soft-Exp: at inference time, replace the argmax token embedding with the continuous probability-weighted sum of all token embeddings.

This was a one-line code change. It did not require retraining. It did not change the training distribution.

We validated Soft-Exp across four scales:

\textbf{Exp 26 (2M)}: Soft-Exp yielded a $+81\%$ relative improvement over argmax. This was the largest gain we observed at any scale.

\textbf{Exp 27 (8M)}: improvement was $+53\%$. The scale-dependence was minimal.

\textbf{Exp 28 (16M/32M)}: at 16M parameters with a fixed LR schedule, improvement was $+48.6\%$. At 32M (which was underfit in our setup), improvement dropped to $+11\%$.

\textbf{Exp 29 (V49 1.2B)}: improvement was $+48.6\%$, identical to 16M. The improvement was scale-invariant across the $600\times$ range from 2M to 1.2B.

The result was striking: a one-line inference change yielded a $+48.6\%$ improvement on a 1.2B model, an improvement that did not diminish with scale. This was the first intervention in the entire 30-round campaign that combined three properties:
\begin{enumerate}
    \item Continuous feedback (the expected embedding is continuous)
    \item No architectural modification
    \item No retraining
\end{enumerate}

\section{3.5 The Final Test (Exp 30): Soft-Exp at Training}

If Soft-Exp worked at inference, what about training? We tested: replace the ground-truth embedding input during training with the continuous expected embedding, linearly warmed from 0 to $0.5$ over 1000 steps.

Result: catastrophic failure.

On V49 1.2B, validation PPL rose from $3.26$ (baseline) and even $33.27$ (inference Soft-Exp) to $8554$. The argmax-feedback PPL rose from $64.74$ to $10662$. Both rose, meaning the model was damaged in its core language modeling capacity, not just in its autoregressive inference.

This was the decisive experiment. It showed that Soft-Exp's success was specifically because it did not modify the training distribution. When the training input distribution was modified to include continuous expected embeddings, the model overfit to a distribution that was OOD with respect to any single inference distribution.

We discussed the mechanism in detail in \S5. The key insight: any change to the training input distribution introduces a covariate shift at inference time that the model cannot recover from.

\section{3.6 Lessons from Phase 3}

Across 30 experiments, we learned:
\begin{enumerate}
    \item \textbf{Patching continuous components into autoregressive Transformers does not work.} We tried 10+ combinations of KAN, complex attention, and Lie group PE. All failed.
    \item \textbf{Bug-fixing does not fix structural problems.} Even after 5+ rounds of code cleanup, CMT failed.
    \item \textbf{Continuous feedback at inference works; continuous feedback at training does not.} The asymmetry is real and structural.
    \item \textbf{The Soft-Exp one-line fix outperforms 30 rounds of architectural engineering.} This suggests that the right interventions are simpler than the architectural dreams.
\end{enumerate}

These lessons motivated the second campaign: instead of patching, we should design a continuous architecture from scratch. This became the CWF (Closed Waveformer) campaign, discussed in \S4.
