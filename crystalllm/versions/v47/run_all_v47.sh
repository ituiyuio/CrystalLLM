#!/bin/bash
# run_all_v47.sh — 依次训练 v47 Phase 1 的 A / B / C 三个变体
# 总耗时: ~3-4 小时 on RTX 5090

set -e

cd D:/CrystaLLM

echo "=== v47 Phase 1 完整训练启动 ==="
echo "时间: $(date)"
echo ""

# Variant A: dense AR baseline
echo ""
echo "######################################################################"
echo "## Variant A (200M dense AR baseline)"
echo "## 启动时间: $(date)"
echo "######################################################################"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe crystalllm/versions/v47/pipeline/train_v47.py \
    --variant A \
    --steps 10000 \
    --batch_size 4 \
    --T 512 \
    --lr 1.5e-4 \
    --warmup_steps 1000 \
    --eval_every 500 \
    --eval_batches_train 32 \
    --eval_batches_final 254

echo ""
echo "######################################################################"
echo "## Variant A 完成: $(date)"
echo "######################################################################"

# Variant B: MoE AR
echo ""
echo "######################################################################"
echo "## Variant B (200M MoE AR)"
echo "## 启动时间: $(date)"
echo "######################################################################"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe crystalllm/versions/v47/pipeline/train_v47.py \
    --variant B \
    --steps 10000 \
    --batch_size 4 \
    --T 512 \
    --lr 1.5e-4 \
    --warmup_steps 1000 \
    --eval_every 500 \
    --eval_batches_train 32 \
    --eval_batches_final 254

echo ""
echo "######################################################################"
echo "## Variant B 完成: $(date)"
echo "######################################################################"

# Variant C: full framework + sparse attention
echo ""
echo "######################################################################"
echo "## Variant C (200M MoE + per-block z + sparse attn + 0.5 L_AR + 0.5 L_diff)"
echo "## 启动时间: $(date)"
echo "######################################################################"
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe crystalllm/versions/v47/pipeline/train_v47.py \
    --variant C \
    --steps 10000 \
    --batch_size 4 \
    --T 512 \
    --lr 1.5e-4 \
    --warmup_steps 1000 \
    --alpha 0.5 \
    --eval_every 500 \
    --eval_batches_train 32 \
    --eval_batches_final 254

echo ""
echo "######################################################################"
echo "## Variant C 完成: $(date)"
echo "######################################################################"
echo ""
echo "=== v47 Phase 1 全部训练完成 ==="