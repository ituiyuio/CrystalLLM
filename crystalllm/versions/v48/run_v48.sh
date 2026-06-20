#!/bin/bash
# run_v48.sh — v48 Phase 2 集成模型训练 (~22h on RTX 5090)

set -e

cd D:/CrystaLLM

echo "=== v48 Phase 2 集成模型训练启动 ==="
echo "时间: $(date)"
echo ""
echo "## 配置:"
echo "  - 1.2B active params (MoE 8 experts Top-2)"
echo "  - T=1024, 64 blocks"
echo "  - Sparse attention (global z + window ±2)"
echo "  - Per-block z injection (位置条件化)"
echo "  - 0.5 L_AR + 0.5 L_diff (α=0.5)"
echo "  - 从零训练 (无 warm-start)"
echo "  - 10000 steps, batch=1, LR=1e-4"
echo "  - Adafactor + grad checkpoint"
echo "  - 数据: 1M+ samples (v24 + v28 + extended_v23)"
echo ""

PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe crystalllm/versions/v48/pipeline/train_v48.py \
    --steps 10000 \
    --batch_size 1 \
    --T 1024 \
    --lr 1e-4 \
    --warmup_steps 1000 \
    --alpha 0.5 \
    --eval_every 500 \
    --eval_batches_train 8 \
    --eval_batches_final 128 \
    --use_grad_checkpoint \
    --optimizer adafactor

echo ""
echo "## 训练完成: $(date)"
echo ""
echo "下一步: python eval_v48.py → PPL 对比 v25/v47 baseline"