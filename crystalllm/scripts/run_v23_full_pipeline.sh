#!/bin/bash
# run_v23_full_pipeline.sh — clean + dedup + pack 全量（232 个 code 文件 + agentic + eval）
set -euo pipefail

cd D:/CrystaLLM/crystalllm
PYTHON=D:/CrystaLLM/.venv/Scripts/python.exe

# 创建输出目录
mkdir -p data/clean_v23/code data/clean_v23/agentic data/clean_v23/eval
mkdir -p data/dedup_v23/code data/dedup_v23/agentic data/dedup_v23/eval
mkdir -p data/processed logs_v23

# ============================
# Step 2: clean (per-file, can parallelize)
# ============================
echo "=== Step 2: clean ===" | tee logs_v23/clean.log

# Clean agentic + eval first (small)
for d in agentic eval; do
    for f in data/raw_v23/$d/*.jsonl; do
        [ -f "$f" ] || continue
        out="data/clean_v23/$d/$(basename $f)"
        if [ -f "$out" ]; then
            echo "  [skip] $out exists" | tee -a logs_v23/clean.log
        else
            $PYTHON clean_v23_data.py "$f" "$out" 2>&1 | tail -1 | tee -a logs_v23/clean.log
        fi
    done
done

# Clean code (232 files) - run in parallel with 8 workers
echo "  [code] starting 232 files in parallel (8 workers)..." | tee -a logs_v23/clean.log
ls data/raw_v23/code/*.jsonl | xargs -I {} -P 8 bash -c '
    f="{}"
    out="data/clean_v23/code/$(basename $f)"
    if [ -f "$out" ]; then
        echo "  [skip] $out"
    else
        '"$PYTHON"' clean_v23_data.py "$f" "$out" 2>&1 | tail -1
    fi
' 2>&1 | tee -a logs_v23/clean.log

n_cleaned=$(ls data/clean_v23/code/*.jsonl 2>/dev/null | wc -l)
echo "  [done] $n_cleaned cleaned code files" | tee -a logs_v23/clean.log

# ============================
# Step 3: dedup (per-file exact + minhash)
# ============================
echo "=== Step 3: dedup ===" | tee logs_v23/dedup.log

# Dedup agentic + eval first (use --no-minhash for the 1.1M-doc Vibe-Coding file)
for d in agentic eval; do
    for f in data/clean_v23/$d/*.jsonl; do
        [ -f "$f" ] || continue
        out="data/dedup_v23/$d/$(basename $f .jsonl).dedup.jsonl"
        if [ -f "$out" ]; then
            echo "  [skip] $out exists" | tee -a logs_v23/dedup.log
        else
            $PYTHON dedup_v23_data.py "$f" "$out" --no-minhash 2>&1 | tail -1 | tee -a logs_v23/dedup.log
        fi
    done
done

# Dedup code (232 files) - parallel, --no-minhash for speed
echo "  [code] starting 232 files in parallel (4 workers, --no-minhash for speed)..." | tee -a logs_v23/dedup.log
ls data/clean_v23/code/*.jsonl | xargs -I {} -P 4 bash -c '
    f="{}"
    out="data/dedup_v23/code/$(basename $f .jsonl).dedup.jsonl"
    if [ -f "$out" ]; then
        echo "  [skip] $out"
    else
        '"$PYTHON"' dedup_v23_data.py "$f" "$out" --no-minhash 2>&1 | tail -1
    fi
' 2>&1 | tee -a logs_v23/dedup.log

n_deduped=$(ls data/dedup_v23/code/*.jsonl 2>/dev/null | wc -l)
echo "  [done] $n_deduped deduped code files" | tee -a logs_v23/dedup.log

# ============================
# Step 4: pack
# ============================
echo "=== Step 4: pack ===" | tee logs_v23/pack.log

# Build the file list
PACK_INPUTS=(
    data/dedup_v23/agentic/*.jsonl
    data/dedup_v23/code/*.jsonl
    data/dedup_v23/eval/*.jsonl
    data/processed/v23_train.parquet   # local sessions anchor
)

# Run pack
$PYTHON pack_v23_data.py --in-paths "${PACK_INPUTS[@]}" --out data/processed/extended_v23.parquet 2>&1 | tee -a logs_v23/pack.log

# Summary
echo ""
echo "=== Final output ==="
ls -la data/processed/extended_v23.parquet
$PYTHON -c "
import pandas as pd
df = pd.read_parquet('data/processed/extended_v23.parquet')
print(f'rows: {len(df)}')
print(f'max text len: {df[\"text\"].str.len().max()}')
print(f'avg text len: {df[\"text\"].str.len().mean():.0f}')
print(f'domains: {df[\"domain\"].value_counts().to_dict()}')
print(f'sources: {df[\"source\"].value_counts().head(5).to_dict()}')
"
