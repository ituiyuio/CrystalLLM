# §5 Why "Patching" Fails: A Mathematical Argument

This section presents the core theoretical contribution of this paper: a formal argument for why patchwork architectures (adding continuous components to discrete pipelines) cannot achieve continuous representation. The argument rests on the concept of \emph{closure}---a property the CMT campaign empirically violated but did not formally require.

## 5.1 The Closure Property

Let $\mathcal{X}$ denote the space of valid representations in a neural network layer, and $f_\theta: \mathcal{X} \to \mathcal{X}$ a parameterized transformation.

\begin{definition}[Closure]
A neural network layer $f_\theta: \mathcal{X} \to \mathcal{X}$ is \emph{closed} on $\mathcal{X}$ if $\forall x \in \mathcal{X}, f_\theta(x) \in \mathcal{X}$.
\end{definition}

For standard Transformers, $\mathcal{X} = \mathbb{R}^d$ and every component (linear, attention, LayerNorm) maps $\mathbb{R}^d \to \mathbb{R}^d$ trivially. Closure is automatic.

For continuous-internal models, we propose $\mathcal{X} = \mathbb{D}^d = \{z \in \mathbb{C}^d : \|z\| < 1\}$ (the open unit ball). Closure requires every layer to map $\mathbb{D}^d$ into $\mathbb{D}^d$.

\begin{definition}[Full closure]
A neural network is \emph{fully closed} on $\mathbb{D}^d$ if every layer (including token embedding and output head) maps $\mathbb{D}^d \to \mathbb{D}^d$.
\end{definition}

\begin{lemma}[Discrete-continuous interface breaks closure]
Let $E: \{0, 1, \ldots, V-1\} \to \mathbb{D}^d$ be a token embedding (discrete input $\to$ continuous representation) and $D: \mathbb{D}^d \to \mathbb{R}^V$ be an output head. If $E$ is bijective (one-to-one) onto a finite subset of $\mathbb{D}^d$, then for any $z \in \mathbb{D}^d$ not in the image of $E$, no layer downstream of $E$ can recover the discrete structure without leaving $\mathbb{D}^d$.
\end{lemma}

\begin{proof}
$E$ maps the discrete vocabulary $\{0, \ldots, V-1\}$ to a finite set $\{e_0, \ldots, e_{V-1}\} \subset \mathbb{D}^d$. Any $z \in \mathbb{D}^d$ not in this set has no discrete preimage. Downstream layers cannot determine which discrete token $z$ corresponds to without a decoding function. The decoding function $D: \mathbb{D}^d \to \mathbb{R}^V$ is necessarily non-injective (it has at most $V$ distinct outputs for an infinite input domain). Therefore, the composed map $D \circ E$ loses information about $z$ outside the finite image.

In practice, this means: any internal continuous representation is constrained to lie in a finite subset of $\mathbb{D}^d$ (the image of $E$) at the input boundary. The internal continuous machinery operates on this discrete set, not on a true continuous distribution.
\end{proof}

\begin{theorem}[Patchwork continuity is impossible]
No architecture that has both (a) a discrete token input and (b) a continuous internal representation can be fully closed on $\mathbb{D}^d$.
\end{theorem}

\begin{proof}
By Lemma 5.1, the input boundary $E$ maps discrete tokens to a finite subset of $\mathbb{D}^d$. For closure to hold, all internal layers must map this finite subset to $\mathbb{D}^d$. This is achievable (every layer is bounded). However, the \emph{converse} closure property fails: not every $z \in \mathbb{D}^d$ has a discrete preimage under $E$. Therefore, internal layers cannot guarantee that their outputs lie in the finite image of $E$, and the input layer cannot recover discrete structure from arbitrary $z$.

This means: any internal continuous representation is at risk of being OOD with respect to the discrete vocabulary. The model can learn to map any internal $z$ to a discrete token (via $D$), but it cannot ensure that the resulting discrete token corresponds to a coherent language model input.
\end{proof}

\begin{corollary}[Training-inference distribution drift]
In a patchwork architecture, the model's training input distribution is supported on $\{e_0, \ldots, e_{V-1}\}$ (the discrete embedding image), while the inference input distribution is supported on a subset of $\mathbb{D}^d$ that may include points not in this finite set. The two distributions are therefore not identical; the inference distribution is \emph{broader} than the training distribution.

This is the formal statement of the training-inference distribution drift observed empirically in Exp 30 (Soft-Exp training) and in the CMT memorization pattern.
\end{corollary}

## 5.2 Why Soft-Exp Inference Avoids Drift

Soft-Exp inference (Exp 29) avoids training-inference drift by a subtle construction:

\begin{lemma}[Soft-Exp preserves discrete support]
The continuous expected embedding $\mathbf{e}_\text{exp} = \mathbf{p}^\top \mathbf{E}$ lies in the convex hull of the discrete embedding image $\{e_0, \ldots, e_{V-1}\}$.
\end{lemma}

\begin{proof}
By definition, $\mathbf{e}_\text{exp} = \sum_v p(v) e_v$ where $\mathbf{p}$ is a probability vector (non-negative entries summing to 1). The result is a convex combination of the $e_v$'s, which lies in their convex hull. \qedhere
\end{proof}

The convex hull of the discrete embedding image is a \emph{strict subset} of $\mathbb{D}^d$. Soft-Exp inference constrains the inference input distribution to this convex hull, which is a subset of the training distribution's support (the discrete image itself).

Therefore, the inference distribution is \emph{narrower} than (or equal to) the training distribution, not broader. This is the opposite of patchwork architectures, where the inference distribution is broader.

The asymmetry explains why Soft-Exp works: it reduces the inference distribution's support rather than expanding it.

\begin{theorem}[Inference narrowing is safe]
If the inference input distribution's support is a subset of the training input distribution's support, then the model is in-distribution at inference time, and standard maximum likelihood training provides PAC-style guarantees on inference performance.
\end{theorem}

\begin{proof}[Sketch]
This is a corollary of standard generalization theory. The training distribution is supported on $\mathcal{S}_\text{train}$; the inference distribution is supported on $\mathcal{S}_\text{infer} \subseteq \mathcal{S}_\text{train}$. The model's loss on $\mathcal{S}_\text{infer}$ is bounded by its loss on $\mathcal{S}_\text{train}$, which is bounded by the training loss. \qedhere
\end{proof}

\section{5.3 The Closure Verdict}

Combining the above:

\begin{itemize}
    \item Patchwork architectures (CMT, Exp 6--25) violate closure (Theorem 5.1).
    \item Continuous internal representations in patchwork architectures cause training-inference drift (Corollary 5.2).
    \item Drift causes memorization or catastrophic failure (empirical, Exp 7--25, 30).
    \item Soft-Exp inference preserves the discrete embedding support (Lemma 5.3), avoiding drift.
    \item Soft-Exp therefore works (empirical, Exp 29).
\end{itemize}

The conclusion: \emph{continuous feedback is safe at inference time only if it stays within the discrete training distribution's convex hull}. Soft-Exp is the unique simple operation that satisfies this constraint while preserving more distributional information than argmax.

\section{5.4 What Would a Truly Closed Architecture Require?}

Theorem 5.1 shows that patchwork is impossible. A truly closed architecture requires:

\begin{enumerate}
    \item \textbf{Continuous input}: no discrete tokens. Inputs are continuous signals (e.g., raw text bytes, audio waveforms).
    \item \textbf{Closed internal layers}: every layer maps $\mathbb{D}^d \to \mathbb{D}^d$.
    \item \textbf{Continuous output}: no discrete token output. The output is a continuous signal, decoded only at the final human-facing interface.
\end{enumerate}

This is a significant architectural departure from the autoregressive Transformer. It is the subject of the CWF Manifesto \citep{wang2026cwf}, a companion research charter. We do not implement it in this paper; we note that it is an open research problem.

\section{5.5 Implications for the Field}

The closure argument has three implications for the broader community:

\begin{enumerate}
    \item \textbf{Patching is futile}. Researchers proposing to add "continuous components" to existing autoregressive Transformers should expect failure. The closure property is structurally violated.

    \item \textbf{Continuous feedback is safe but limited}. Soft-Exp shows that continuous inference-time feedback can be added without breaking the model, but only if it stays within the discrete distribution's convex hull. This is a narrow but valuable design space.

    \item \textbf{Truly closed models require redesign}. A new architecture from scratch, with continuous inputs and outputs, is needed. This is a 1--2 person-year research investment, not a paper-level contribution.
\end{enumerate}

We hope that this formal analysis, combined with the empirical evidence from 30 CMT rounds, will save the community from repeating the patchwork path.
