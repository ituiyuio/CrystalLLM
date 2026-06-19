#!/bin/bash
# run_all_v46.sh — 依次训练 v46 Phase 0 的 A / B / C 三个变体
# 总耗时: ~90 min on RTX 5090

set -e

cd D:/CrystaLLM

echo "=== v46 Phase 0 完整训练启动 ==="
echo "时间: $(date)"
echo ""

# Variant A: dense AR baseline
echo ""
echo "######################################################################"
echo "## Variant A (dense AR baseline)"
echo "## 启动时间: $(date)"
echo "######################################################################"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe crystalllm/versions/v46/pipeline/train_v46.py \
    --variant A \
    --steps 5000 \
    --batch_size 8 \
    --T 512 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --eval_every 250 \
    --eval_batches_train 32 \
    --eval_batches_final 254

echo ""
echo "######################################################################"
echo "## Variant A 完成: $(date)"
echo "######################################################################"

# Variant B: MoE AR
echo ""
echo "######################################################################"
echo "## Variant B (MoE AR)"
echo "## 启动时间: $(date)"
echo "######################################################################"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe crystalllm/versions/v46/pipeline/train_v46.py \
    --variant B \
    --steps 5000 \
    --batch_size 8 \
    --T 512 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --eval_every 250 \
    --eval_batches_train 32 \
    --eval_batches_final 254

echo ""
echo "######################################################################"
echo "## Variant B 完成: $(date)"
echo "######################################################################"

# Variant C: full framework
echo ""
echo "######################################################################"
echo "## Variant C (full framework: MoE + per-block z + 0.5 AR + 0.5 diff)"
echo "## 启动时间: $(date)"
echo "######################################################################"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe crystalllm/versions/v46/pipeline/train_v46.py \
    --variant C \
    --steps 5000 \
    --batch_size 8 \
    --T 512 \
    --lr 3e-4 \
    --warmup_steps 500 \
    --alpha 0.5 \
    --eval_every 250 \
    --eval_batches_train 32 \
    --eval_batches_final 254

echo ""
echo "######################################################################"
echo "## Variant C 完成: $(date)"
echo "######################################################################"
echo ""
echo "=== v46 Phase 0 全部训练完成 ==="