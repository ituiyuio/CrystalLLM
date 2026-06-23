# §1 Introduction

## 1.1 Motivation

The dominant paradigm in language modeling is the autoregressive Transformer: predict the next token given previous tokens, train with teacher forcing, and deploy by feeding back the model's own predictions. This paradigm has achieved remarkable empirical success---GPT-class models, LLaMA, and many others have demonstrated that scaling autoregressive Transformers yields increasingly capable language models.

Yet a structural concern persists: the autoregressive Transformer operates on \emph{discrete} tokens at the input and output, while the underlying signal (human language) has \emph{continuous} structure (phonetic, semantic, syntactic gradients). This discretization at the interface raises a question that the field has explored sporadically but never resolved: \emph{can we make Transformers continuous?}

The intuition is appealing. Continuous representations could:
\begin{itemize}
    \item Preserve the rich gradient information that tokenization discards
    \item Enable smooth interpolation in the representation space
    \item Allow multi-modal signals (audio, video, sensor data) to be processed without first quantizing to discrete units
    \item Eliminate the train-inference distribution mismatch that causes exposure bias
\end{itemize}

Over the past five years, several research communities have pursued this intuition through diverse approaches: complex-valued neural networks, Kolmogorov-Arnold representations, Lie group actions, neural ODEs, and diffusion language models. Each has had promising theoretical motivations. Each has, in our experience, failed to deliver on the autoregressive Transformer.

This paper documents \textbf{35 systematic experiments} we conducted to test whether continuous representations can be integrated into the autoregressive framework. Our finding is unambiguous: \textbf{patching continuity into the autoregressive Transformer is a dead end}. We prove this formally (§5) and confirm it empirically (§3, §6). The single positive result is a one-line code change that we did not originally set out to find: continuous expected embedding feedback at inference time.

## 1.2 Why We Wrote This Paper

The standard treatment of negative results in machine learning is to publish them only when they reveal something positive. We challenge this convention. Our 35 experiments collectively reveal a clear and useful boundary:

\begin{itemize}
    \item \textbf{What doesn't work}: continuous internal representations in autoregressive Transformers, regardless of whether they are implemented as complex-valued attention, KAN MLPs, Lie group position encodings, or fully closed wave architectures.

    \item \textbf{What does work (partially)}: continuous feedback at inference time, in the specific form of probability-weighted expected embeddings. This is the Soft-Exp method, which we report in companion work \citep{wang2026softexp}.

    \item \textbf{What might work (untested)}: a fully continuous architecture designed from scratch with ODE evolution machinery, recurrent state, and time-step embedding. This is the CWF Manifesto research charter \citep{wang2026cwf}.
\end{itemize}

By publishing this systematic record, we hope to save the community from repeating our 35 experiments and to provide a clear empirical foundation for future work on continuous language models.

## 1.3 Our Method

We conducted two research campaigns:

\textbf{CMT Campaign (Exp 1--30, $\sim$3 months)}: patch continuous components onto a standard autoregressive Transformer. We tested 10+ combinations of KAN FFNs, complex-valued attention, and Lie group position encodings across 5 model scales (2M to 1.2B parameters) and multiple tokenization schemes (character-level and BPE). The campaign concluded with the \textbf{CMT\_DEAD\_FINAL} verdict after 30 rounds.

\textbf{CWF Campaign (Exp 01--02, 2 weeks)}: design a fully closed continuous architecture from scratch. We formulated a closure conjecture, implemented a minimal CWF prototype with closure verified across training, and stress-tested it on the Lorenz chaotic system. The campaign concluded with a \textbf{TOTAL\_FAIL} at the GO/NO-GO gate.

Both campaigns were data-driven: each experiment's design was informed by the previous experiment's results, and the campaigns adapted as evidence accumulated. We did not pre-commit to a single architectural vision; we followed the evidence.

## 1.4 The 35 Experiments at a Glance

\begin{table}[h]
\centering
\small
\begin{tabular}{lcc}
\toprule
Verdict & Count & Percentage \\
\midrule
STRONG PASS & 1 & 2.9\% \\
PASS & 4 & 11.4\% \\
PARTIAL & 1 & 2.9\% \\
INCONCLUSIVE & 4 & 11.4\% \\
\textbf{FAIL} & \textbf{25} & \textbf{71.4\%} \\
\bottomrule
\end{tabular}
\caption{Distribution of verdicts across the 35 experiments.}
\end{table}

The single STRONG PASS is Soft-Exp on V49 1.2B (Exp 29). It is the only experiment that combines all three properties we sought:
\begin{enumerate}
    \item Continuous feedback (probability-weighted embedding)
    \item No architectural modification (one line of code)
    \item No retraining (model weights unchanged)
\end{enumerate}

The 25 FAILs span the entire design space we explored. No combination of continuous components, training schemes, hyperparameters, or tokenization methods produced a model that survived the autoregressive training regime without memorization or catastrophic performance loss.

## 1.5 The Three Independent Evidence Streams

We did not arrive at the "patching is futile" conclusion from a single experimental design. We arrived at it from three independent streams:

\textbf{Stream 1: CMT Campaign (30 experiments).} We patched KAN FFNs, complex-valued attention, and Lie group position encodings into autoregressive Transformers in 10+ combinations. All failed with a consistent pattern: training-inference distribution drift, expressed as memorization at 8--12k training steps.

\textbf{Stream 2: Soft-Exp Training Test (Exp 30).} We took the Soft-Exp inference method, which works (+48.6\%), and applied it at training time. The result was catastrophic ($\mathrm{PPL} = 8554$ vs baseline $3.26$). This is the cleanest possible demonstration that the training-inference gap must be closed from the inference side, not the training side.

\textbf{Stream 3: CWF Campaign (5 experiments).} We abandoned patching and designed a purpose-built closed continuous architecture. Despite closure being verified (max $\psi$ norm $< 1$ throughout training), the architecture failed to learn the Lorenz dynamics (effective prediction time of 2 steps vs Oracle's 100 steps). The root cause was the absence of ODE evolution machinery, not closure itself.

Three independent experimental designs, all reaching the same conclusion. This convergence is the strongest possible evidence for the patching-is-futile hypothesis.

## 1.6 The Closure Argument (Preview of §5)

The mathematical core of this paper is a formal argument for why patching fails. We define \emph{closure} as the property that every layer maps the representation space into itself. For continuous-internal models, we propose the open unit ball $\mathbb{D}^d$ as the representation space.

We prove:
\begin{theorem}[Patchwork continuity is impossible]
No architecture that has both (a) a discrete token input and (b) a continuous internal representation can be fully closed on $\mathbb{D}^d$.
\end{theorem}

The proof rests on the observation that a discrete token embedding maps a finite vocabulary to a finite subset of $\mathbb{D}^d$, while a continuous internal representation requires the entire $\mathbb{D}^d$. The resulting mismatch causes the model's training input distribution (supported on the finite subset) to differ from its inference input distribution (supported on a broader subset of $\mathbb{D}^d$). This is the formal statement of training-inference distribution drift.

The corollary explains Soft-Exp's success: Soft-Exp's expected embedding lies in the \emph{convex hull} of the discrete embedding image, which is a subset of the training distribution's support. The inference distribution is therefore narrower than (not broader than) the training distribution, avoiding the drift that doomed patchwork architectures.

## 1.7 Contributions

This paper makes the following contributions:

\begin{enumerate}
    \item \textbf{Systematic empirical record}: 35 experiments across two research campaigns, documented with full reproducibility information (Appendix B).

    \item \textbf{Five cross-cutting failure modes}: identified from the experiments, each with formal definition, first appearance, and empirical evidence (Appendix A).

    \item \textbf{Closure theorem}: a formal argument for why patchwork continuous architectures cannot be closed, and why Soft-Exp inference avoids the failure mode (§5).

    \item \textbf{Lorenz decisive experiment}: a clean stress test of continuous architectures on a canonical chaotic system, with three models (CWF, AR+VQ, Oracle) compared on six metrics (§6).

    \item \textbf{Research charter}: the CWF Manifesto, a documented open problem for the community, with concrete requirements for what a successful fully continuous architecture would need (§6.7, §8).
\end{enumerate}

## 1.8 Outline

§2 reviews the theoretical motivation for continuous representations. §3 narrates the CMT Campaign in detail. §4 introduces the CWF Manifesto and the pivot to redesign. §5 presents the closure argument. §6 reports the CWF Phase 1 decisive experiment. §7 discusses the Soft-Exp survivor. §8 concludes with the CWF Manifesto as a research challenge.

Appendix A presents the five cross-cutting failure modes. Appendix B provides reproducibility details. Appendix C lists all 35 experiments in a single table.

## 1.9 A Note on Tone

We have written this paper as a \emph{systematic record} rather than a celebration or a lamentation. The 30 failed CMT rounds were not wasted: each provided information that shaped the next experiment. The 5 CWF rounds were not futile: they falsified a hypothesis that would otherwise have remained untested. The single Soft-Exp survivor is not an accident: it is what survived 34 rounds of systematic elimination.

We hope this record will be useful to researchers considering the continuous-representation path. Our advice: do not patch; do not be surprised when patching fails; do design from scratch; do accept that the engineering investment is 1--2 person-years; and do not expect to be done in a paper.
