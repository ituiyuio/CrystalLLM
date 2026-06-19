# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""test_v36_model.py — v36 model forward pass + param count sanity checks"""
import torch
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.path.insert(0, ".")
from v36_model import DecoderCrossAttn, BlockCrossAttn

# v25 / v36 配置
V = 2261
T = 512
D_Z = 256
DEC_LAYER = 24
DEC_HEAD = 20
DEC_EMBD = 1280
BOS_ID = 1

torch.manual_seed(42)
decoder = DecoderCrossAttn(V, T, DEC_LAYER, DEC_HEAD, DEC_EMBD, D_Z, BOS_ID).to("cuda")
n_params = sum(p.numel() for p in decoder.parameters())
print(f"v36 decoder params: {n_params/1e6:.2f}M")
assert 560e6 < n_params < 580e6, f"param count {n_params/1e6:.2f}M not in expected 560-580M range"

# BlockCrossAttn 子层存在性
block = decoder.blocks[0]
required = ["ln1", "qkv", "proj", "ln_cross", "q_cross", "k_cross", "v_cross", "proj_cross", "ln2", "mlp"]
for name in required:
    assert hasattr(block, name), f"BlockCrossAttn missing {name}"
print(f"BlockCrossAttn has all required sublayers: {required}")

# 前向 shape 校验
B = 4
z = torch.randn(B, D_Z, device="cuda")
x = torch.randint(0, V, (B, T), device="cuda")
logits = decoder(z, x)
print(f"logits shape: {logits.shape}, expected: ({B}, {T+1}, {V})")
assert logits.shape == (B, T + 1, V), f"unexpected logits shape {logits.shape}"

# 梯度反传校验
loss = logits.sum()
loss.backward()
has_grad = sum(1 for p in decoder.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
total = sum(1 for p in decoder.parameters())
print(f"params with non-zero grad: {has_grad}/{total}")
assert has_grad == total, "not all params received gradient"

print("\n✓ All sanity checks passed")
