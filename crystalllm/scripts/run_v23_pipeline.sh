#!/bin/bash
# run_v23_pipeline.sh — v23 数据准备 5 步 driver
# 用法: bash scripts/run_v23_pipeline.sh [--skip-train] [--code-quota 20G] [--wiki-quota 10G]
set -euo pipefail

cd "$(dirname "$0")/.."

SKIP_TRAIN=0
CODE_QUOTA=20000000000
WIKI_QUOTA=10000000000
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-train) SKIP_TRAIN=1; shift;;
        --code-quota) CODE_QUOTA="$2"; shift 2;;
        --wiki-quota) WIKI_QUOTA="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "=== Step 0: discover schema ==="
python discover_v23_schema.py --source armand0e/claude-fable-5-claude-code --sample-mb 5
python discover_v23_schema.py --source Glint-Research/Fable-5-traces --sample-mb 5
python discover_v23_schema.py --source lazarus19/Vibe-Coding-Claude-Fable-5 --sample-mb 5
python discover_v23_schema.py --source ZhipuAI/humaneval-x --split test --sample-mb 1
python discover_v23_schema.py --source swift/github-code --subset-name default --sample-mb 5
python discover_v23_schema.py --source swift/wikipedia --subset-name en --sample-mb 5
python discover_v23_schema.py --source swift/wikipedia --subset-name zh --sample-mb 5

echo "=== Step 1a: agentic download ==="
python download_v23_agentic.py

echo "=== Step 1b: streaming download ==="
python download_v23_streaming.py --code-quota-chars $CODE_QUOTA --wiki-quota-chars $WIKI_QUOTA

echo "=== Step 2: clean ==="
for d in agentic code wiki; do
    mkdir -p data/clean_v23/$d
    for f in data/raw_v23/$d/*.jsonl; do
        [ -f "$f" ] || continue
        out="data/clean_v23/$d/$(basename $f)"
        python clean_v23_data.py "$f" "$out"
    done
done

echo "=== Step 3: dedup ==="
for d in agentic code wiki; do
    mkdir -p data/dedup_v23/$d
    for f in data/clean_v23/$d/*.jsonl; do
        [ -f "$f" ] || continue
        out="data/dedup_v23/$d/$(basename $f .jsonl).dedup.jsonl"
        python dedup_v23_data.py "$f" "$out"
    done
done

echo "=== Step 4: pack ==="
python pack_v23_data.py \
    --in-paths data/dedup_v23/agentic/*.jsonl \
                data/dedup_v23/code/*.jsonl \
                data/dedup_v23/wiki/*.jsonl \
                data/processed/v23_train.parquet \
    --out data/processed/extended_v23.parquet

echo "=== Step 5: train (optional) ==="
if [ $SKIP_TRAIN -eq 0 ]; then
    CRYSTALLM_V23=1 python proto_v23_decoder.py
    CRYSTALLM_V23=1 python eval_v23_e2e.py
fi

echo "=== Done ==="
echo "Output: data/processed/extended_v23.parquet"
