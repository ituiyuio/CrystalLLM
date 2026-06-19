# autoresearch — CrystaLLM experiment scaffolding (migrated to versions/)

> **Migration note (2026-06-19)**: 140 `.py` scripts formerly under `autoresearch/{training,evaluation,pipeline,benchmarks}/` have been moved into [`../versions/`](../versions/) grouped by version + function. This directory now only keeps the module entry point, `tests/` sanity checks, and the `nanochat/` 3rd-party port.

## What this is

`autoresearch/` is the **module entry point for `crystalllm/`'s own experiment scaffolding**. It is not a port of Karpathy nanochat or any other public repository.

- **Historical arc**: from `versions/v2/training/proto_v2.py` (26 letters / 15 words, toy world) all the way to `versions/v36/training/train_v36_decoder.py` (570M params, end-to-end diffusion + AR)
- **Version style**: each version (v10, v22, v23, …, v36) has a `train_v{NN}_*.py` entry + `eval_v{NN}_*.py` evaluation + `versions/v{NN}/*_train_log.json` log
- **Research thread**: introduces "diffusion localization" as a global planning layer on top of autoregression (see root [`../README.md`](../README.md))

## Current directory layout

```
autoresearch/
├── __init__.py
├── README.md              # this file
├── tests/                 # inline sanity tests  (6 scripts) — kept in place
│   ├── test_v28_speed.py
│   ├── test_v32_speed.py
│   ├── test_v36_model.py
│   ├── test_v36_warmstart.py
│   ├── test_collect_v29.py
│   └── test_collect_v30.py
└── nanochat/              # simplified Karpathy nanochat port  (2 scripts) — kept in place
    ├── train.py
    └── prepare.py
```

> Note: the former `training/`, `evaluation/`, `pipeline/`, `benchmarks/` subdirectories are now empty. All 140 `.py` scripts have moved to [`../versions/`](../versions/).

## Where the version scripts now live

Organized by version number; each version subdirectory keeps the four function subdirs `training/`, `evaluation/`, `pipeline/`, `benchmarks/`:

```
versions/
├── _common/                              # version-less shared scripts
│   ├── training/{prototype.py, train.py}
│   ├── evaluation/{check_expand_upper.py, check_unused_sessions.py, check_unused_tokens.py}
│   └── benchmarks/{bench_speed_quality.py, speed_benchmark.py}
├── v2/training/proto_v2.py               # M1 toy world
├── v3/.../v9/                            # early prototypes (v3-v9)
├── v10/training/proto_v10.py
├── v10/evaluation/eval_v10.py
├── v11/.../v18/                          # VAE / KL / cross-attn exploration
├── v19/
│   ├── training/proto_v19_diffusion_prior.py
│   ├── evaluation/{eval_v19_e2e.py, smoke_v19.py}
│   └── benchmarks/benchmark_v19.py
├── v19_5/training/proto_v195_pure_ar.py
├── v20a/.../v22/                         # decoder capacity scaling
├── v23/{training,evaluation,pipeline}/   # data-ingest stage (16 scripts, the largest)
├── v24/.../v32/                          # diffusion localization + KV cache compression
├── v34a/  v34b/  v34d/                   # cross-attn stage sub-versions
└── v35/  v36/                            # latest cross-attn + warm-start
```

Full 140-file distribution: see the "per-version stats" output from `scripts/organize_to_versions.py`, or simply:

```bash
find crystalllm/versions -name "*.py" | sort
```

## Version-detection rules

Scripts tagged with `v<N>` in their filename are placed under `versions/v<N>/`. Special cases:

| Filename token | Resolved version | Notes |
|---|---|---|
| `v2`, `v10`, `v36` | matching dir | integer versions auto-create a dir |
| `v195`, `v215` | `v19_5`, `v21_5` | 3-digit ending in `5` denotes a `.5` sub-version |
| `v20a`, `v22a`, `v23_BAD`, `v26_5`, `v28_5`, `v34a`, `v34b`, `v34d` | same-named dir | dedicated dir already exists |
| `v19b`, `v15_3`, `v34c` | main version or `_common/` | falls back to `_common/` if the main version dir is missing (e.g. `v34c` → `_common/` because `v34/` doesn't exist) |
| no `v<num>` token | `_common/` | `prototype.py`, `train.py`, `bench_speed_quality.py`, etc. |

Migration script: `crystalllm/scripts/organize_to_versions.py` (idempotent, rerunnable).

## Naming conventions

| Prefix | Meaning |
|---|---|
| `train_v{NN}_*.py` | Training entry for version NN (N = 22–36) |
| `eval_v{NN}_*.py` | Evaluation script for the corresponding version |
| `proto_v{N}_*.py` | Early prototype exploration (N = 2–23) |
| `debug_v{NN}_*.py` | Throw-away debug script (usually tied to one specific bug fix) |
| `check_*.py` | Data / consistency checks (rerunnable) |
| `smoke_v{NN}.py` | End-to-end smoke test (under 10 seconds) |
| `*_train_log.json` | Training metric log (val PPL / step time / LR) |
| `cached_v{NN}_*.npz` | Pre-computed feature cache (git-ignored, see root `.gitignore`) |
| `proto_v{N}_model.pt` / `v{NN}_*.pt` | Model checkpoint (git-ignored) |

## Quick start (new paths)

```bash
# 1. Train v36 (latest, 570M params)
uv run python crystalllm/versions/v36/training/train_v36_decoder.py

# 2. Evaluate v36 end-to-end PPL + speed
uv run python crystalllm/versions/v36/evaluation/eval_v36_e2e.py

# 3. Check v25 → v36 warm-start loading
uv run python crystalllm/autoresearch/tests/test_v36_warmstart.py

# 4. Run all sanity tests under autoresearch/
uv run python crystalllm/autoresearch/tests/

# 5. Re-run the migration script (idempotent)
uv run python crystalllm/scripts/organize_to_versions.py
```

## Data dependency

The scripts under `pipeline/` (now at `versions/v*/pipeline/`) read session data from `crystalllm/data/` by default. That directory is **not in git** (see root `.gitignore`); rebuild locally from `~/.claude/projects/` via `versions/v23/pipeline/download_v23_streaming.py`.

See root [`README.md`](../README.md) and [`../design.md`](../design.md).

## Relationship with the rest of the project

- `crystalllm/versions/` — **experiment scripts collection** (organized vertically by version)
- `crystalllm/autoresearch/` — module entry point (`__init__.py` + `tests/` + `nanochat/`)
- `crystalllm/autoresearch/nanochat/` — **simplified port** of Karpathy nanochat (a separate sub-project, used as infrastructure reference and performance baseline)
- `crystalllm/` root — **core code + docs** (README / TIMELINE / goal / design / logs)
- `crystalllm/docs/` — design docs and meeting notes produced during the research
- `crystalllm/tests/` — pytest-style formal tests (distinct from `autoresearch/tests/` inline sanity checks)

## Evolution roadmap

| Version | Stage | Major changes |
|---|---|---|
| v1/v2 | M1 | Toy world (9/26 chars) validating the diffusion-localization idea |
| v3-v6 | Early | Real-corpus ingestion, proto series up to 12M params |
| v10-v17 | Exploration | VAE / KL / cross-attn mechanism trials |
| v22-v23 | v23 stage | 2467 real sessions ingested; V23_MODE flag switches datasets |
| v28-v32 | Diffusion-localization | KV cache compression + drafter speedups |
| v34-v36 | cross-attn stage | cross-attn decoder + warm-start |

Training logs per version: see `crystalllm/versions/v{NN}/*_train_log.json` (git-tracked) and `TIMELINE.md` (chronological).