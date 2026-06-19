# CrystaLLM — Prototype

> Information-Crystallization Language Model: from high-entropy noise to low-entropy semantics, then per-token routing to concrete text.

## Overview

Four progressively-scaled prototypes + real corpus ingestion layer:

| Path | Vocabulary | Training corpus | Scale | Demonstrated capability |
|---|---|---|---|---|
| `prototype.py` | 9 chars | 3 words (hand-crafted) | D_Z=12 | Intra-cluster generation + information phase transition |
| `proto_v2.py` | 26 letters | 15 words (hand-crafted) | D_Z=3 | + Inter-cluster interpolation + 3D latent space + phase-transition curve |
| `proto_v3.py` | 788 chars | **100 real sessions** | 2M params | **End-to-end pipeline**: jsonl → parquet → train → generate |
| `proto_v4.py` | 788 chars | 100 real sessions | 2.1M params | **AR + diffusion localization + z reconstruction**: z-space structure + z interpolation controlling language/topic |
| `proto_v5.py` | 1701 chars | **1317 real sessions** | 11.4-11.8M params | **Scaled comparison**: v3 (no z) vs v4 (with z) at 12M / 13× data |
| `proto_v6.py` | 1701 chars | 1317 real sessions | 11.78M params | **Prefix-LM paradigm**: fixes v4 design flaws, **PPL 7.2 < v3 9.1** + pure-diffusion generation demo |

v1/v2 share the same architecture: 5-step diffusion → latent z → z-prefix GRU decoder.
v3 is a pure-AR baseline (no diffusion), used to verify the data → train → inference pipeline end-to-end.
v4 ports the v1/v2 diffusion-localization idea onto v3's real data.

## Quick start

```bash
uv run python crystalllm/versions/_common/training/prototype.py   # v1: 50-line minimal prototype
uv run python crystalllm/versions/v2/training/proto_v2.py         # v2: controllable generation + visualization
```

v2 additionally outputs `crystalllm/phase_transition.png`: left panel shows 3D latent space (15 anchors + interpolation path), right panel shows the ‖z‖ phase-transition curve (all 3 clusters converge to anchor norm ≈4).

## Verified hypotheses

- **Entropy collapse**: after diffusion, ‖z_clean‖ converges to the anchor norm regardless of initial noise strength (see v2 phase-transition plot).
- **Semantic alignment**: z's neighbors in cluster K decode to words from cluster K (v1: all 3 anchors correct; v2: same-cluster words are mutual neighbors).
- **Controllable interpolation**: linearly interpolating z between two cluster anchors produces a smooth transition in decoded words (v2 cat→red 5-step demo).
- **Pipeline reach-through (v3)**: 100 real sessions / 2M params / 12 s train → val PPL 24.7; generated text preserves the structure of the training distribution (code syntax, mixed CN/EN, markdown headings). See [`versions/v3/training_log_v3.md`](./versions/v3/training_log_v3.md).
- **z-space structure (v4)**: z reconstruction input → prevents collapse; z scatter forms a 1D manifold curve; z interpolation yields smooth language/topic transitions. See [`versions/v4/training_log_v4.md`](./versions/v4/training_log_v4.md) and `z_space.png`.
- **PPL gap is design-driven (v5)**: scaling to 12M / 13× data, v3 PPL 9.1, v4 PPL 35 — gap widens rather than shrinks. See [`versions/v5/training_log_v5.md`](./versions/v5/training_log_v5.md) and [`versions/v5/v5_*.png`](./versions/v5/).
- **Prefix-LM fix (v6)**: single forward + z-required signal → val PPL **7.2 < v3 9.1**; effective z rank 28/64; **pure-diffusion generation demo** (z from N(0,I) → 5-step denoise → multilingual / code text). See [`versions/v6/training_log_v6.md`](./versions/v6/training_log_v6.md) and [`versions/v6/v6_z_space.png`](./versions/v6/v6_z_space.png).

## Training corpus

`data/` ingests local snapshots from `~/.claude/projects/` — 16 projects, 2467 jsonl sessions, ~12 GB. **git-excluded**, see [`data/README.md`](./data/README.md).

Subset (100 short sessions, ~109K tokens) is extracted via `versions/v16/pipeline/make_v16_subset.py`; vocabulary is built by the same script and written to `processed/char_vocab.json` (git-tracked).

## Design roadmap

Full design in [`goal.md`](./goal.md) (OKRs) and [`design.md`](./design.md) (architecture, training, evaluation, risks).
- v1 / v2 correspond to the **M1 (minimal prototype)** stage: validating the two-stage pipeline on a toy world.
- M2 / M3 stages: see milestones.

## Relationship with `autoresearch/`

`autoresearch/` is the **module entry point for the project's own experiment scaffolding** (`__init__.py` + `tests/` + `nanochat/`). The 140 `.py` scripts formerly under `pipeline/`, `training/`, `evaluation/`, `benchmarks/` have been migrated into [`versions/`](./versions/) by version + function. The naming borrows from the "short-and-fast training loop" idea (each version keeps a rerunnable training entry + eval scripts + training logs), but all scripts, models, and evaluation methods are written from scratch around CrystaLLM's "diffusion localization + autoregressive decoding" paradigm — they are NOT a port of Karpathy nanochat or any other public repo.

See [`autoresearch/README.md`](./autoresearch/README.md) and the migration script [`scripts/organize_to_versions.py`](./scripts/organize_to_versions.py).

## Directory organization

- **`autoresearch/`** — module entry + inline sanity tests + nanochat port (140 version scripts have been migrated out)
- **`versions/v<N>/`** — every artifact of one experiment for a given version (training scripts, eval scripts, training logs, result reports, model weights, z caches, visualizations, etc.). Variants are flattened into independent subdirectories: `v19_5/`, `v20a/`, `v23_BAD/`, `v26_5/`, `v28_5/`, `v34a/`, `v34b/`, `v34d/`. Each version subdirectory is further split by function: `training/`, `evaluation/`, `pipeline/`, `benchmarks/`
- **`versions/_common/`** — version-less shared scripts (`prototype.py`, `train.py`, `check_*.py`, `bench_speed_quality.py`, etc.)
- **`data/`** — training corpus and vocabulary (git-excluded)
- **`scripts/`** — top-level shell entry points + `organize_to_versions.py` migration tool
- **Cross-version files kept at the project root**: `bench_*` (benchmarks), `scaling_results.md`, `results_v23.tsv`, `kv_cache_train.npz`, `pca_basis.npz`, `phase_transition.png`, `z_space.png`
- **Top-level docs**: `README.md`, `TIMELINE.md`, `design.md`, `goal.md`, `.env.v23.example`

> Some hard-coded paths inside eval scripts (`versions/v*/evaluation/`) — e.g. `crystalllm/cached_v18_z.npz`, `v25_decoder.pt` — still point to old locations and need their path constants updated separately after the migration.