# CrystaLLM

> **Information-Crystallization Language Model** — diffusion localizes the semantic domain, autoregression routes the tokens.

A personal research repo that iterates on one bet:

> **Pure next-token prediction is not the path to AGI. Diffusion-conditioned generation with global semantic planning is.**

It records the full journey — from a 50-line character-level toy to a 570M-parameter hybrid model — together with the scaffolding, design docs, and per-version experiment logs accumulated along the way.

---

## TL;DR

- **Premise.** Text generation is reframed as a two-stage physical process: high-entropy noise → low-entropy latent semantic intent → concrete tokens.
- **Architecture.** Phase I: 5–10 diffusion steps in a low-dim latent space produce a `z` that carries global intent. Phase II: a causal Transformer decoder generates tokens conditioned on `z`.
- **Status (v36, 2026-06-19).** 570M-param cross-attn decoder, val PPL 2.81, 14 min single-GPU training. **Honest read: this regresses on PPL vs the v25 baseline (2.47)** but resolves the v35 collapse-to-spaces failure mode — see [`crystalllm/versions/v36/v36_results.md`](./crystalllm/versions/v36/v36_results.md) for the full autopsy. v25 remains the current SOTA.
- **Trajectory.** 50 versions in [`crystalllm/versions/`](./crystalllm/versions/), from M1 toy world to real-corpus diffusion + AR hybrid. Full timeline in [`crystalllm/TIMELINE.md`](./crystalllm/TIMELINE.md).

---

## The two-stage pipeline

```
                        Phase I                          Phase II
                    ┌──────────────────┐           ┌────────────────────┐
  noise             │  diffusion       │    z      │  AR decoder        │   tokens
  z_T ~ N(0, I) ──► │  (5–10 steps)    │ ────────► │  p(xₜ | x<ₜ, z)   │ ──► x₁ x₂ … x_N
                    │  global intent   │  semantic │  token-by-token    │
                    └──────────────────┘  domain   └────────────────────┘
```

- **Why hybrid.** Pure AR has no global plan ("next-token pathfinding"). Pure diffusion has weak long-sequence precision. Hybrid splits the work along the axis each does best.
- **Why this might matter.** If you believe generation needs both *layout* and *realization*, then a planner that runs in latent space — rather than a hidden state the decoder must learn to compress — is a more honest decomposition.

---

## Where things stand

### Headline numbers

| Model         | Params | Val PPL | Non-space rate | Code-structure samples | Note                            |
|---------------|-------:|--------:|---------------:|-----------------------:|---------------------------------|
| v3 (pure AR)  |    2 M |   24.7  |             —  |                    —   | end-to-end pipeline sanity      |
| v6 (prefix-LM)| 11.8 M |    7.2  |             —  |                    —   | first hybrid that beats AR PPL  |
| v25 (BAD-DP)  |  476 M |  **2.47** |          ~70 % |                  ~5/10 | **current SOTA baseline**       |
| v28.5 (BAD-DP)|  555 M |    2.39 |           ~0 % |                   0/10 | collapsed to spaces             |
| **v36 (cross-attn)** | **570 M** | **2.81** |      **85 %** |              **6/10** | fixed collapse, regressed PPL   |

Reading the v36 row honestly: it solves the collapse problem the prior cross-attn attempts hit, but PPL gets worse — likely because the latent `z` is high-KL noise that cross-attn forces every layer to consume. The detailed diagnosis and next-step plan (v37 prefix-tuning, v38 fix `z` distribution) are in the v36 report.

### What was actually learned

- **Entropy collapse works in toy worlds.** After diffusion, `‖z_clean‖` converges to the anchor norm regardless of initial noise strength (see `phase_transition.png`).
- **Prefix-LM beats pure AR at 12M scale.** v6 (with `z`) at PPL 7.2 < v3 (without `z`) at PPL 9.1.
- **Scaling widens the gap, it doesn't close it.** v5 confirmed the PPL gap between with-z and without-z is design-driven, not data-driven.
- **Warm-start is fragile.** Adding 94M randomly-initialized cross-attn tensors to a 476M converged decoder (4000 steps total) was not enough to out-perform the baseline — random-init portion stays under-trained.
- **Batch val PPL is a weak signal.** The real generalization number is full-val PPL; batch numbers have 2× variance.
- **v25 is the best AR-routing baseline we have.** Any new variant has to beat 2.47 to be worth keeping.

---

## Repository layout

```
CrystaLLM/
├── README.md                # this file
├── LICENSE                  # Apache 2.0
├── pyproject.toml           # uv-managed deps (PyTorch 2.9.1 + cu128)
│
├── crystalllm/              # core research direction: diffusion + AR hybrid
│   ├── README.md            # detailed technical write-up
│   ├── TIMELINE.md          # complete timeline M0 → v36 (zh)
│   ├── goal.md              # OKRs
│   ├── design.md            # architecture spec
│   │
│   ├── versions/            # per-version scripts + artifacts (50 versions)
│   │   ├── _common/         #   version-less shared scripts
│   │   ├── v1 … v9/         #   M1 toy world + early prototypes
│   │   ├── v10 … v22/       #   VAE / KL / cross-attn exploration
│   │   ├── v23/             #   data-ingest stage (2467 sessions, ~12 GB)
│   │   ├── v24 … v32/       #   diffusion localization + KV-cache compression
│   │   ├── v34a / v34b / v34d/  # cross-attn stage sub-versions
│   │   └── v35 / v36/       #   latest cross-attn + warm-start
│   │
│   ├── autoresearch/        # experiment scaffolding (Karpathy-style self-edit loop)
│   │   └── nanochat/        #   simplified port for infra reference
│   ├── docs/                # design notes + meeting notes
│   ├── tests/               # pytest suite
│   ├── scripts/             # shell entry points + version-organizer
│   ├── logs/                # training logs
│   └── data/                # training data (gitignored, generated locally)
│
└── docs/                    # top-level plans + specs
```

Each `versions/v<N>/` keeps four function subdirs — `training/`, `evaluation/`, `pipeline/`, `benchmarks/` — so every experiment is self-contained and rerunnable. Variants with the same major version (`v19_5/`, `v20a/`, `v23_BAD/`, `v34a/`, `v34b/`, `v34d/`) are flattened into independent subdirectories.

---

## Quick start

```bash
# 1. Install
uv sync

# 2. Re-run v36 end-to-end evaluation
uv run python crystalllm/versions/v36/evaluation/eval_v36_e2e.py

# 3. Replay the toy prototypes (Phase I proof of concept)
uv run python crystalllm/versions/_common/training/prototype.py     # v1
uv run python crystalllm/versions/v2/training/proto_v2.py           # v2

# 4. Run the pytest suite
uv run pytest crystalllm/tests/

# 5. Try the simplified Karpathy nanochat (single-GPU GPT pretraining)
uv run python crystalllm/autoresearch/nanochat/train.py
```

Hardware target: a single CUDA GPU with ~16 GB VRAM is enough for the v36 eval; full v25/v36 training fits in ~14 min on a 24 GB card.

---

## Data

Training data is ingested from local session snapshots under `~/.claude/projects/` — 2467 jsonl files, ~12 GB. This directory is **not** in git and is rebuilt locally by the scripts under [`crystalllm/versions/v23/pipeline/`](./crystalllm/versions/v23/pipeline/).

The corpus is **not** redistributable; no personal data leaves the machine.

---

## Documentation map

| Doc                                              | What's inside                                            |
|--------------------------------------------------|----------------------------------------------------------|
| [`crystalllm/README.md`](./crystalllm/README.md) | detailed prototype write-up, verified hypotheses         |
| [`crystalllm/goal.md`](./crystalllm/goal.md)     | OKRs (project goals + key results)                       |
| [`crystalllm/design.md`](./crystalllm/design.md) | architecture, training, evaluation, risks                |
| [`crystalllm/TIMELINE.md`](./crystalllm/TIMELINE.md) | complete M0 → v36 timeline, per-version findings     |
| [`crystalllm/versions/v36/v36_results.md`](./crystalllm/versions/v36/v36_results.md) | the latest experiment, in full                       |
| [`crystalllm/autoresearch/README.md`](./crystalllm/autoresearch/README.md) | scaffolding layout + naming conventions           |

---

## Roadmap (where this is going)

- **v37 — prefix-tuning z → M memory tokens.** Cheaper than cross-attn, lets the decoder selectively use `z` instead of being forced to consume it at every layer.
- **v38 — fix the `z` distribution itself.** Current KL ≈ 303 nats is too high — `z` is high-entropy and hard to use. Either weaken the KL constraint (free_bits 1.0 → 5.0) so the encoder can learn a tighter `z`, or drop `z` and accept a pure prefix decoder.
- **M3 milestone.** 1.5B-param joint-trained diffusion + decoder per the original OKRs.

---

## Credits & license

- [Karpathy nanochat](https://github.com/karpathy/nanochat) — `autoresearch/nanochat/` is a simplified port (MIT, upstream copyright preserved).
- [Karpathy autoresearch](https://x.com/karpathy/status/2029701092347630069) — the "AI agent self-edits train.py to do experiments" paradigm that inspired the `versions/` layout.
- Training corpus: local `~/.claude/projects/` sessions (personal, never uploaded).

**Apache License 2.0** — see [`LICENSE`](./LICENSE). The `train.py` and `prepare.py` under `autoresearch/nanochat/` retain their upstream MIT notice.

---

*"The future of AI is not just next-token prediction. It's diffusion-conditioned generation with global semantic planning."*
