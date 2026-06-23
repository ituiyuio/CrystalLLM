# §7 Where Continuity Survived: The Soft-Exp Survivor

Among the 35 experiments, only one achieved a strong positive result: Soft-Exp inference on V49 1.2B (Exp 29). In this section, we explain why Soft-Exp survived when all other continuous interventions failed, using the closure framework developed in §5.

## 7.1 What Soft-Exp Is

Soft-Exp is a one-line modification to the standard autoregressive inference loop. Instead of feeding back the embedding of the argmax token:
\begin{lstlisting}[language=Python]
next_embed = token_emb(argmax(logits))
\end{lstlisting}
Soft-Exp feeds back the probability-weighted expected embedding:
\begin{lstlisting}[language=Python]
next_embed = probs @ token_emb.weight
\end{lstlisting}

The mathematical formulation:
$$\mathbf{e}_{\text{next}} = \mathbf{p}^\top \mathbf{E}$$
where $\mathbf{p} \in \Delta^{V-1}$ is the softmax probability vector and $\mathbf{E} \in \mathbb{R}^{V \times d}$ is the embedding matrix.

\section{7.2 The Empirical Result}

On V49 1.2B (1.2B parameters, char-level, 2261 vocabulary):

\begin{table}[h]
\centering
\small
\begin{tabular}{lcc}
\toprule
Mode & Val PPL & Notes \\
\midrule
Teacher Forcing & 3.26 & oracle capacity baseline \\
Argmax feedback & 64.74 & standard AR inference \\
\textbf{Soft-Exp} & \textbf{33.27} & \textbf{+48.6\% vs argmax} \\
\bottomrule
\end{tabular}
\caption{Soft-Exp halves the exposure bias without retraining.}
\end{table}

The improvement is scale-invariant: at 2M, 8M, 16M, and 1.2B parameters, Soft-Exp yields $+81\%$, $+53\%$, $+48.6\%$, and $+48.6\%$ respectively.

\section{7.3 Why Soft-Exp Avoids the Failure Mode

Recall from §5.2 the key lemma:

\begin{lemma}[Soft-Exp preserves discrete support]
The continuous expected embedding $\mathbf{e}_\text{exp} = \mathbf{p}^\top \mathbf{E}$ lies in the convex hull of the discrete embedding image $\{e_0, \ldots, e_{V-1}\}$.
\end{lemma}

The convex hull of a finite set is itself a bounded subset of the ambient space. For the embedding matrix $\mathbf{E} \in \mathbb{R}^{V \times d}$, the convex hull $\mathrm{conv}(\mathbf{E})$ is a polytope with $V$ vertices, contained in the finite-dimensional ambient space $\mathbb{R}^d$.

Crucially, $\mathrm{conv}(\mathbf{E}) \subseteq \mathbb{R}^d$, while the training distribution's support is exactly the discrete set $\{e_0, \ldots, e_{V-1}\} \subset \mathrm{conv}(\mathbf{E})$. Therefore:

$$\mathrm{supp}(p_{\text{infer}}) = \mathrm{conv}(\mathbf{E}) \subseteq \mathbb{R}^d$$
$$\mathrm{supp}(p_{\text{train}}) = \{e_0, \ldots, e_{V-1}\} \subset \mathrm{conv}(\mathbf{E})$$

So $\mathrm{supp}(p_{\text{train}}) \subseteq \mathrm{supp}(p_{\text{infer}})$.

This is the \emph{reverse} of the patchwork situation, where $\mathrm{supp}(p_{\text{infer}}) \supsetneq \mathrm{supp}(p_{\text{train}})$.

\begin{theorem}[Inference distribution is narrower]
The Soft-Exp inference distribution's support is a superset of (or equal to) the training distribution's support. The model is therefore in-distribution at every inference step.
\end{theorem}

\begin{proof}
By construction, $\mathbf{e}_{\text{exp}}$ is a convex combination of the discrete embeddings $\{e_v\}$. The training distribution's support is exactly $\{e_v\}$. So $\mathbf{e}_{\text{exp}}$ is always in the convex hull of the training support. \qedhere
\end{proof}

The model is trained on a discrete subset of the inference support. At inference, the input is always in the convex hull of the training support. The model's training covered the discrete subset but the inference covers the convex hull. The model is therefore never OOD at inference.

In particular, this avoids the training-inference drift that doomed CMT and Exp 30 (training-time Soft-Exp).

\section{7.4 Why Soft-Exp Training Failed

Exp 30 tested training-time Soft-Exp: replace the ground-truth embedding input during training with the continuous expected embedding. The result was catastrophic ($\mathrm{PPL} = 8554$ on V49 1.2B).

Why? At training time, the input is no longer the discrete embedding. It is a convex combination. The model's training distribution's support is no longer $\{e_0, \ldots, e_{V-1}\}$; it is the convex hull $\mathrm{conv}(\mathbf{E})$.

At inference time, the input is also a convex combination (Soft-Exp). The two distributions now have the same support.

But this is the problem: the model is trained on a \emph{different} distribution from the baseline (which used discrete embeddings). At inference, even if we apply Soft-Exp, the model's gradient updates during training were based on continuous inputs. The model has overfit to the continuous-input training distribution, which is itself different from any single inference distribution.

Worse, the convex hull $\mathrm{conv}(\mathbf{E})$ is a continuous region with uncountably many points. The model is trained on samples from this region (one per training step). At inference, the model sees a different sample from the same region. There is no guarantee that the model's behavior on one sample predicts its behavior on another, especially for a model whose inductive bias is designed for discrete inputs.

\begin{theorem}[Soft-Exp training causes overfitting]
Training with continuous expected embedding causes the model to overfit to a continuous input distribution that is OOD with respect to the model's inductive bias (which is designed for discrete token inputs).
\end{theorem}

This is a special case of the broader closure failure pattern: any change to the training input distribution causes failure (Corollary 5.4).

\section{7.5 The Unique Position of Soft-Exp

Among all 35 experiments, Soft-Exp occupies a unique position:

\begin{itemize}
    \item It uses continuous feedback (the expected embedding is continuous).
    \item It does not modify the training distribution.
    \item It does not require retraining.
    \item It yields significant improvement ($+48.6\%$).
    \item The improvement is scale-invariant.
\end{itemize}

These properties are not coincidental. They follow from the closure framework:

\begin{enumerate}
    \item \textbf{Continuous feedback}: Soft-Exp's expected embedding is in a continuous region.
    \item \textbf{No training modification}: this is the key design choice; the model is trained with discrete inputs.
    \item \textbf{No retraining}: the inference modification does not affect training.
    \item \textbf{Improvement}: continuous feedback preserves more information than discrete argmax.
    \item \textbf{Scale-invariance}: the improvement is bounded by information loss in argmax, which is weakly scale-dependent.
\end{enumerate}

The principle is simple: \emph{continuous signals preserve more distributional information than discrete point estimates, and the inference distribution can be made continuous without retraining as long as it stays within the convex hull of the training distribution}.

\section{7.6 The Soft-Exp Limitation

Soft-Exp does not eliminate exposure bias. On V49 1.2B, the residual gap is $10.22\times$ (down from $19.88\times$, but not $1\times$). Closing the gap entirely would require either:

\begin{itemize}
    \item Matching the inference distribution exactly to the training distribution (which is impossible with continuous feedback unless the training also uses continuous feedback, which fails as shown in §7.4).
    \item A new architecture that is truly closed (the CWF Manifesto path, §4).
\end{itemize}

The first option is theoretically possible but empirically failed (Exp 30). The second option is the CWF research charter. Neither is solved by Soft-Exp alone.

Soft-Exp is therefore a \emph{partial fix}: it cuts exposure bias in half for zero cost, but the structural problem of train-inference mismatch remains. This is a useful engineering result, not a theoretical resolution.

\section{7.7 Connection to Knowledge Distillation

The principle underlying Soft-Exp is the same as the principle underlying knowledge distillation \citep{hinton2015distill}: soft targets carry more information than hard targets.

In knowledge distillation:
\begin{itemize}
    \item A large "teacher" model produces soft probability distributions over classes.
    \item A small "student" model is trained on these soft targets.
    \item The student generalizes better than if trained on hard one-hot labels.
\end{itemize}

In Soft-Exp:
\begin{itemize}
    \item The autoregressive model produces a soft probability distribution over the next token.
    \item The same model's inference uses the expected embedding under this distribution.
    \item The next-step prediction is more accurate than if it used the discrete argmax.
\end{itemize}

In both cases, the soft distribution preserves information that the hard target/argmax discards. The soft target's information is the model's own uncertainty, which encodes useful structural information about the output space.

\section{7.8 What This Means for Future Work

The Soft-Exp result provides one piece of guidance for future continuous-representation work: \emph{continuous signals at the inference interface are safe, but only if the inference distribution is contained in the convex hull of the training distribution}.

This is a narrow but valuable design principle. It says that any continuous inference modification is acceptable as long as it does not introduce inputs that the model has not been trained on.

A more aggressive design principle would be to train with continuous feedback (which failed in Exp 30) or to design a truly closed architecture (which failed in Phase 1 for lack of ODE evolution). Neither is currently validated.

The Soft-Exp success is a reminder that the simplest intervention, carefully designed to respect the closure framework, can yield significant improvement. The complexity of full continuous architectures may not be necessary if a targeted one-line modification achieves the same effect.

We hope this observation will inspire future work to focus on minimal, theoretically-grounded modifications before attempting large architectural redesign.
