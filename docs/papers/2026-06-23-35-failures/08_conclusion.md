# §8 Conclusion and Research Challenge

## 8.1 Summary of 35 Experiments

Across two research campaigns (CMT and CWF), we conducted 35 systematic experiments to test whether continuous representations can be integrated into autoregressive Transformers. The results are unambiguous:

\begin{center}
\small
\begin{tabular}{lcc}
\toprule
Approach & Outcome & Mechanism \\
\midrule
Patch KAN FFN & FAIL (memorizer) & Closure violation \\
Patch complex attention & FAIL (memorizer) & Closure violation \\
Patch Lie group PE & PARTIAL (no benefit) & Architecture mismatch \\
Patch Soft-Exp at training & FAIL (catastrophic) & Train-inference drift \\
Soft-Exp at inference & \textbf{PASS} (+48.6\%) & Narrowed inference support \\
Closed architecture (CWF) & FAIL (no ODE) & Missing components \\
AR+VQ bottleneck & FAIL (catastrophic) & Information loss \\
\bottomrule
\end{tabular}
\end{center}

The single positive result is Soft-Exp inference, a one-line code modification that requires no retraining. It works because it narrows the inference distribution to the convex hull of the training distribution, avoiding the training-inference drift that doomed all other interventions.

The full closure argument (§5) explains why this is the only intervention that can work within the autoregressive framework. Patchwork architectures cannot be closed (Theorem 5.3). Continuous training-time feedback causes overfitting (§7.4). Continuous inference-time feedback is safe only if it stays within the training distribution's convex hull (§7.3).

\section{8.2 The Three Lessons}

From these 35 experiments, we extract three lessons that may be useful to the broader community.

\subsection{Lesson 1: Patching is structurally futile}

Adding continuous components to a discrete pipeline does not make the pipeline continuous. The discrete interfaces (token embedding, vocabulary output) create information bottlenecks that no amount of internal continuous machinery can overcome. This is a structural property of the architecture, not an implementation detail.

\subsection{Lesson 2: Closure is necessary but not sufficient}

A closed architecture maintains its representation throughout the network, avoiding the drift that doomed patchwork. But closure alone is not enough for continuous dynamics tasks. ODE evolution machinery, time-step embedding, and recurrent state are also required. The CWF Phase 1 prototype had closure but not these other components, and it failed the GO/NO-GO gate.

\subsection{Lesson 3: Simple interventions often beat architectural redesign}

Soft-Exp is one line of code. It outperforms 30 rounds of CMT engineering and 5 rounds of CWF architectural design. The lesson: before redesigning the architecture, look for targeted modifications that respect the existing framework. Soft-Exp respects the autoregressive framework (no retraining, no architectural change) and exploits a specific information-theoretic opportunity (the dark knowledge in non-maximal probabilities).

\section{8.3 The CWF Manifesto as Research Challenge}

The CWF Manifesto \citep{wang2026cwf} remains an open research problem. A truly continuous language model would require:

\begin{enumerate}
    \item \textbf{Continuous input}: no discrete tokenization. Inputs are raw signals (bytes, audio, video).
    \item \textbf{Continuous internal state}: all representations in $\mathbb{D}^d$ or another continuous manifold.
    \item \textbf{Closed layers}: every layer maps the representation into itself.
    \item \textbf{ODE evolution}: state evolves via a learned ODE, not step-by-step prediction.
    \item \textbf{Time-step embedding}: $\Delta t$ is a model input, enabling any temporal resolution.
    \item \textbf{Recurrent state}: the model maintains a state across rollout steps.
    \item \textbf{Continuous output}: no discrete token output. Output is decoded only at the final human-facing interface.
\end{enumerate}

Building such a model is a 1--2 person-year engineering investment. It requires:
\begin{itemize}
    \item Custom CUDA kernels for complex-valued matrix multiplication and ODE solvers
    \item New training algorithms (adjoint method for memory-efficient backpropagation through ODE)
    \item New evaluation protocols (no discrete perplexity; how do we measure continuous model quality?)
    \item New datasets (continuous signals, not tokenized text)
\end{itemize}

We do not know whether this investment will succeed. We offer the CWF Manifesto as a research charter to anyone willing to commit the engineering effort.

\section{8.4 What We Recommend}

Based on 35 experiments, our recommendations to the community are:

\begin{enumerate}
    \item \textbf{Do not patch} continuous components into autoregressive Transformers. The closure property is structurally violated.

    \item \textbf{Do consider} Soft-Exp inference for production deployments. It is a one-line change with no retraining cost and a $48\%$ PPL improvement at 1.2B scale.

    \item \textbf{Do investigate} fully continuous architectures from scratch, but only with realistic expectations. The engineering investment is 1--2 person-years, and the failure modes are subtle (closure without ODE evolution is insufficient).

    \item \textbf{Do share} negative results. The 30 failed CMT rounds were not wasted; each provided information that shaped the next experiment. The community benefits from systematic records of failure.

    \item \textbf{Do not assume} that mathematical elegance implies engineering viability. The closure conjecture was mathematically natural, but the CWF prototype still failed the Phase 1 gate. Mathematics is necessary but not sufficient.
\end{enumerate}

\section{8.5 Limitations of This Paper}

We acknowledge three limitations:

\begin{enumerate}
    \item \textbf{Single corpus}: all experiments used the v28 corpus subset. We did not test on diverse domains (code, multilingual, multimodal). Soft-Exp's scale-invariance suggests it should generalize, but we did not verify this.

    \item \textbf{Single chaos system}: the CWF Phase 1 test used the Lorenz system. We did not test on other chaotic systems (Rössler, double pendulum, etc.). The diagnostic about ODE evolution absence applies broadly, but specific results may differ.

    \item \textbf{Limited hyperparameter search}: each experiment had a specific hyperparameter configuration. We did not exhaustively tune hyperparameters for each model. The CWF failure on rollout may be partially addressable by longer training, but the architectural absence of ODE evolution is a structural limitation, not a hyperparameter issue.
\end{enumerate}

\section{8.6 A Note on the Companion Paper}

This paper is the long-form record. A companion paper documents the single positive result:

\begin{itemize}
    \item \textbf{Paper 1}: "Exposure Bias is Bigger Than You Think: Continuous Expected Embeddings as a Drop-In Fix for Autoregressive Decoding" \citep{wang2026softexp}.
\end{itemize}

Paper 1 is a 4-page ICLR-workshop-style paper that focuses on the Soft-Exp method, its empirical validation, and its theoretical motivation via the closure framework. Together, the two papers tell a complete story:

\begin{enumerate}
    \item \textbf{Paper 1}: the surviving fix (1 line of code, $+48.6\%$ PPL).
    \item \textbf{Paper 2}: the systematic context that produced the fix (35 experiments, closure argument, CWF research charter).
\end{enumerate}

We encourage readers interested in the practical method to read Paper 1 first. Readers interested in the broader research context should read Paper 2 (this paper). The two are designed to be complementary, not redundant.

\section{8.7 Final Acknowledgments}

This work represents the equivalent of three months of full-time work on autoregressive Transformer experimentation, followed by two weeks on the CWF campaign. The 30 CMT rounds were not wasted: each provided information that shaped the next experiment. The 5 CWF rounds were not futile: they falsified a hypothesis that would otherwise have remained untested. The single Soft-Exp survivor is not an accident: it is what survived 34 rounds of systematic elimination.

We thank the open-source community for the foundational tools (PyTorch, NumPy, SciPy) that made this work possible. We thank the reviewers who, in earlier stages of this work, pushed us to formalize the closure argument and to design the Phase 1 decisive experiment.

The CWF Manifesto remains an open research charter. We hope that the systematic record presented in this paper will save future researchers from repeating our 35 experiments, and will provide a clear empirical foundation for the next attempt to build a truly continuous language model.

\begin{center}
\itshape
"The best way to have a good idea is to have a lot of ideas."\\
--- Linus Pauling
\end{center}

\begin{center}
\itshape
"We must not cease from exploration, and the end of all our exploring\\
will be to arrive where we began, and to know the place for the first time."\\
--- T. S. Eliot
\end{center}
