#!/bin/bash
# Direct run with unbuffered output to file (avoid tee buffering issues)
cd D:/CrystaLLM

# Use python -u for unbuffered, redirect stdout/stderr directly to file
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -u crystalllm/versions/v48/pipeline/train_v48.py \
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
    --optimizer adafactor \
    > crystalllm/versions/v48/v48_train_all.log 2>&1

echo "Exit code: $?" >> crystalllm/versions/v48/v48_train_all.log
