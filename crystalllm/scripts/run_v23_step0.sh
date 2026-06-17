#!/bin/bash
# run_v23_step0.sh — 并行跑 7 个源的 schema 探查
set -uo pipefail

cd D:/CrystaLLM/crystalllm
mkdir -p data/schema_v23
mkdir -p D:/tmp_v23_dl
export MODELSCOPE_CACHE=D:/tmp_v23_dl/

PYTHON=D:/CrystaLLM/.venv/Scripts/python.exe
LOG_DIR=data/schema_v23/logs
mkdir -p $LOG_DIR

probe() {
    local name=$1; shift
    local logfile=$LOG_DIR/${name}.log
    echo "[$(date +%H:%M:%S)] START  $name -> $logfile"
    $PYTHON discover_v23_schema.py "$@" > $logfile 2>&1
    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "[$(date +%H:%M:%S)] DONE   $name (rc=$rc)"
    else
        echo "[$(date +%H:%M:%S)] FAIL   $name (rc=$rc) -- see $logfile"
    fi
}

# 3 个 Fable 5 agentic 源
probe fable5_1 --source armand0e/claude-fable-5-claude-code --sample-mb 5 --trust-remote-code &
probe fable5_2 --source Glint-Research/Fable-5-traces --split test --sample-mb 5 --trust-remote-code &
probe fable5_3 --source lazarus19/Vibe-Coding-Claude-Fable-5 --subset-name default --sample-mb 5 --trust-remote-code &
# HumanEval-X
probe humaneval --source ZhipuAI/humaneval-x --split test --sample-mb 1 --trust-remote-code &
# github-code (用 Python-all, 不用 default)
probe ghcode_py --source swift/github-code --subset-name "Python-all" --sample-mb 5 --trust-remote-code &
# wikipedia (中英)
probe wiki_en --source swift/wikipedia --subset-name en --sample-mb 5 --trust-remote-code &
probe wiki_zh --source swift/wikipedia --subset-name zh --sample-mb 5 --trust-remote-code &

wait
echo ""
echo "=== Summary ==="
ls -la data/schema_v23/*.json 2>/dev/null
