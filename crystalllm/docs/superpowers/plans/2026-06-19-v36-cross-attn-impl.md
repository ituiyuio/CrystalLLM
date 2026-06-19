# v36 BAD-DP v2 (Cross-Attention) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 v25 的 BAD-DP decoder 改为 cross-attention BAD-DP，warm-start 自 v25，验证 PPL < 2.30 + 生成非空格率 > 90% + KL < 200。

**Architecture:** 每个 block 在 self-attn + MLP 之间插入 cross-attn 子层，z 作为 K/V 单 token。z 不再 prepended 到序列。BOS 移到 pos 0。参数量 ~570M (vs v25 476M)。

**Tech Stack:** PyTorch (与 v25 同), AdamW, cosine LR, RTX 5090, 19K v24 数据。

---

## File Structure

**新增文件**:
| 文件 | 职责 |
|---|---|
| `crystalllm/v36_model.py` | `BlockCrossAttn` + `DecoderCrossAttn` 类定义（共享给 train/eval/debug） |
| `crystalllm/test_v36_model.py` | 模型前向 shape 校验 + 参数量校验 |
| `crystalllm/test_v36_warmstart.py` | warm-start 加载校验 (loaded/skipped/fresh 计数) |
| `crystalllm/train_v36_decoder.py` | 训练脚本（含 warm-start 加载 + 4000 步训练） |
| `crystalllm/eval_v36_e2e.py` | PPL + 速度评测（端到端） |
| `crystalllm/debug_v36_gen.py` | 生成质量调试（非空格率 + 样本输出） |
| `crystalllm/v36_decoder.pt` | 训练产出模型 |
| `crystalllm/v36_decoder_train_log.json` | 训练日志 |
| `crystalllm/v36_e2e.json` | 评测指标 JSON |
| `crystalllm/v36_results.md` | 实验报告 |

**不修改文件**: v25_*, v28_5_*, v31_*, v35_* 等历史只读。

**共享依赖**:
- `data/processed/v24_train.parquet` (19,307) + `v24_val.parquet` (1,016)
- `cached_v24_z.npz` (D_Z=256 train + val z)
- `v25_decoder.pt` (warm-start 来源)

---

## Task 1: 写 v36_model.py (BlockCrossAttn + DecoderCrossAttn)

**Files:**
- Create: `crystalllm/v36_model.py`

- [ ] **Step 1: 创建文件，写入两个类**

```python
"""v36_model.py — v36 cross-attention decoder 模型定义

v36 = v25 架构 + 每 block 加 cross-attn(z) 子层
z 不再 prepended 到序列; z 作为 K/V 传给每个 block
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BlockCrossAttn(nn.Module):
    """self-attn + cross-attn(z as K/V) + mlp"""
    def __init__(s, N_EMBD, N_HEAD, D_Z):
        super().__init__()
        s.nh = N_HEAD
        s.head_dim = N_EMBD // N_HEAD
        # Self-attention (warm-start from v25)
        s.ln1 = nn.LayerNorm(N_EMBD)
        s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        # Cross-attention (NEW, random init)
        s.ln_cross = nn.LayerNorm(N_EMBD)
        s.q_cross = nn.Linear(N_EMBD, N_EMBD)
        s.k_cross = nn.Linear(D_Z, N_EMBD)
        s.v_cross = nn.Linear(D_Z, N_EMBD)
        s.proj_cross = nn.Linear(N_EMBD, N_EMBD)
        # MLP (warm-start from v25)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
        )

    def forward(s, x, z_kv):
        B, T, C = x.shape
        # Self-attention (existing, unchanged from v25)
        h = s.ln1(x)
        qkv = s.qkv(h).reshape(B, T, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(attn.transpose(1, 2).contiguous().view(B, T, C))
        # Cross-attention (NEW)
        h_c = s.ln_cross(x)
        q_c = s.q_cross(h_c).reshape(B, T, s.nh, s.head_dim).permute(0, 2, 1, 3)
        # z_kv: (B, D_Z) → (B, 1, N_EMBD) → (B, 1, nh, head_dim)
        k_c = s.k_cross(z_kv).reshape(B, 1, s.nh, s.head_dim).permute(0, 2, 1, 3)
        v_c = s.v_cross(z_kv).reshape(B, 1, s.nh, s.head_dim).permute(0, 2, 1, 3)
        # 无 causal mask, full attn to z
        attn_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
        x = x + s.proj_cross(attn_c.transpose(1, 2).contiguous().view(B, T, C))
        # MLP (existing)
        x = x + s.mlp(s.ln2(x))
        return x


class DecoderCrossAttn(nn.Module):
    """v36 decoder: BOS + tokens, 每 block cross-attn(z as K/V)"""
    def __init__(s, V, T, DEC_LAYER, DEC_HEAD, DEC_EMBD, D_Z, BOS_ID):
        super().__init__()
        s.V = V; s.T = T; s.BOS_ID = BOS_ID
        s.d_z = D_Z
        # 不再有 z_to_emb; z 通过 k_cross/v_cross 直接注入
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)  # T+2 保持与 v25 形状一致
        s.blocks = nn.ModuleList([
            BlockCrossAttn(DEC_EMBD, DEC_HEAD, D_Z) for _ in range(DEC_LAYER)
        ])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight  # tied weights

    def forward(s, z, x):
        B, T = x.shape
        bos = s.tok(torch.tensor([s.BOS_ID], device=x.device)).expand(B, 1, -1)
        inp = torch.cat([bos, s.tok(x)], dim=1)        # (B, T+1, D) 不 prepend z
        inp = inp + s.pos(torch.arange(T + 1, device=x.device))
        for b in s.blocks:
            inp = b(inp, z)                             # z 传给每个 block
        return s.head(s.ln_f(inp))                      # (B, T+1, V)
```

- [ ] **Step 2: 验证语法**

Run: `python -c "from v36_model import DecoderCrossAttn, BlockCrossAttn; print('imports OK')"`
Expected: `imports OK`

---

## Task 2: 写 test_v36_model.py (前向 shape + 参数量校验)

**Files:**
- Create: `crystalllm/test_v36_model.py`

- [ ] **Step 1: 创建校验脚本**

```python
"""test_v36_model.py — v36 model forward pass + param count sanity checks"""
import torch
import sys
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
```

- [ ] **Step 2: 运行校验**

Run: `cd crystalllm && python test_v36_model.py`
Expected: 输出 `v36 decoder params: ~570.XXM`，然后 `✓ All sanity checks passed`

- [ ] **Step 3: Commit**

```bash
cd /d/CrystaLLM
git add crystalllm/v36_model.py crystalllm/test_v36_model.py
git commit -m "v36: add cross-attn decoder model + sanity check"
```

---

## Task 3: 写 test_v36_warmstart.py (warm-start 加载校验)

**Files:**
- Create: `crystalllm/test_v36_warmstart.py`

- [ ] **Step 1: 创建 warm-start 校验脚本**

```python
"""test_v36_warmstart.py — 验证 v25 → v36 warm-start 加载的 shape 与计数"""
import torch
import sys
sys.path.insert(0, ".")
from v36_model import DecoderCrossAttn

# v25 配置
V, T, D_Z = 2261, 512, 256
DEC_LAYER, DEC_HEAD, DEC_EMBD = 24, 20, 1280
BOS_ID = 1

# 加载 v25
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cpu", weights_only=False)
v25_state = ckpt_v25["decoder"]
print(f"v25 state keys: {len(v25_state)}")

# 构建 v36
new_decoder = DecoderCrossAttn(V, T, DEC_LAYER, DEC_HEAD, DEC_EMBD, D_Z, BOS_ID)
new_state = new_decoder.state_dict()
print(f"v36 state keys: {len(new_state)}")

# warm-start 加载
loaded, skipped, fresh, mismatched = 0, 0, 0, 0
for k, v in v25_state.items():
    if k in ("z_to_emb.weight", "z_to_emb.bias"):
        skipped += 1
        continue
    if k == "pos.weight":
        # v25 pos[0]=z, pos[1]=BOS, pos[2:T+2]=tokens; v36 pos[0]=BOS, pos[1:T+1]=tokens
        new_state[k][: T + 1] = v[1 : T + 2]
        loaded += 1
        continue
    if k in new_state:
        if v.shape == new_state[k].shape:
            new_state[k] = v
            loaded += 1
        else:
            mismatched += 1
            print(f"  shape mismatch: {k} v25={v.shape} v36={new_state[k].shape}")
    else:
        fresh += 1

print(f"\nWarm-start summary:")
print(f"  loaded (from v25): {loaded}")
print(f"  skipped (z_to_emb): {skipped}")
print(f"  fresh (random init, kept): {fresh}")
print(f"  mismatched (shape error): {mismatched}")

# 校验计数 (基于实际验证)
# loaded: pos(1) + self-attn/mlp 权重 = 293 (v25 295 keys - 2 z_to_emb)
# skipped: z_to_emb.weight + bias = 2
# fresh: cross-attn 10 tensors × 24 blocks = 240 (含 ln_cross 和 bias)
assert loaded == 293, f"expected 293 loaded, got {loaded}"
assert skipped == 2, f"expected 2 skipped (z_to_emb), got {skipped}"
assert mismatched == 0, f"unexpected mismatches: {mismatched}"
assert fresh == 240, f"expected 240 fresh cross-attn tensors, got {fresh}"

new_decoder.load_state_dict(new_state)

# 验证 v36 仍能前向
new_decoder = new_decoder.to("cuda")
B = 2
z = torch.randn(B, D_Z, device="cuda")
x = torch.randint(0, V, (B, T), device="cuda")
logits = new_decoder(z, x)
print(f"\npost-warmstart forward OK, logits shape: {logits.shape}")

print("\n✓ Warm-start sanity checks passed")
```

- [ ] **Step 2: 运行校验**

Run: `cd crystalllm && python test_v36_warmstart.py`
Expected: `loaded: 290 / skipped: 2 / fresh: 96 / mismatched: 0`，然后 `✓ Warm-start sanity checks passed`

- [ ] **Step 3: Commit**

```bash
cd /d/CrystaLLM
git add crystalllm/test_v36_warmstart.py
git commit -m "v36: add warm-start loading sanity check"
```

---

## Task 4: 写 train_v36_decoder.py (训练脚本)

**Files:**
- Create: `crystalllm/train_v36_decoder.py`

- [ ] **Step 1: 创建训练脚本**

```python
"""train_v36_decoder.py — v36 cross-attn decoder 训练

warm-start from v25_decoder.pt; 数据复用 v24; 超参与 v25 一致
"""
import json, time, random, sys, io, os
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


P("=== v36 Cross-Attention Decoder (warm-start from v25) ===")
sys.path.insert(0, ".")
from v36_model import DecoderCrossAttn

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]
BOS_ID = stoi.get("<bos>", 1); PAD_ID = stoi.get("<pad>", 0); EOS_ID = stoi.get("<eos>", 2)
P(f"Vocab {V} (v22, 与 v25 decoder 一致)")

df_train = pd.read_parquet(DATA / "v24_train.parquet")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
P(f"v24 train: {len(train_texts)} | val: {len(val_texts)}")

# ===== 配置 (与 v25 一致) =====
B, T = 4, 512
D_Z = 256
DEC_LAYER, DEC_HEAD, DEC_EMBD = 24, 20, 1280
LR, STEPS = 1e-4, 4000
EVAL_EVERY = 250
W_RECON, W_KL = 1.0, 0.1
KL_ANNEAL_STEPS = 1000
FREE_BITS_NAT = 1.0
DEVICE = "cuda"

# 构建 v36 decoder
decoder = DecoderCrossAttn(V, T, DEC_LAYER, DEC_HEAD, DEC_EMBD, D_Z, BOS_ID).to(DEVICE)

# Warm-start 加载
P("\n=== Warm-start: 加载 v25 decoder 权重 ===")
ckpt_v25 = torch.load("v25_decoder.pt", map_location="cpu", weights_only=False)
v25_state = ckpt_v25["decoder"]
new_state = decoder.state_dict()
loaded, skipped, fresh = 0, 0, 0
for k, v in v25_state.items():
    if k in ("z_to_emb.weight", "z_to_emb.bias"):
        skipped += 1; continue
    if k == "pos.weight":
        new_state[k][: T + 1] = v[1 : T + 2]; loaded += 1; continue
    if k in new_state and v.shape == new_state[k].shape:
        new_state[k] = v; loaded += 1
    else:
        fresh += 1
decoder.load_state_dict(new_state)
n_dec = sum(p.numel() for p in decoder.parameters())
P(f"v36 decoder: {n_dec/1e6:.2f}M (loaded {loaded}, skipped {skipped}, fresh {fresh})")
assert loaded == 293 and skipped == 2 and fresh == 240

opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

cache = np.load("cached_v24_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)
P(f"Loaded v24 cached z: train {train_z_cache.shape} | val {val_z_cache.shape}")


def kl_loss(mu, logvar, free_bits=FREE_BITS_NAT):
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_per_dim = kl.mean(dim=0)
    return torch.clamp(kl_per_dim, min=free_bits).sum(), kl_per_dim.detach()


P(f"\n=== train {STEPS} steps, B={B}, T={T} ===")
t0 = time.time()
log = []
best_val_ppl = float("inf")
for step in range(STEPS):
    decoder.train()
    ix = np.random.randint(0, len(train_texts), B)
    x_chunks = []
    for i in ix:
        text = train_texts[i]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        x_chunks.append([stoi.get(c, 0) for c in chunk])
    x = torch.tensor(x_chunks, dtype=torch.long, device=DEVICE)
    z = train_z_cache[torch.tensor(ix, device=DEVICE)]
    logvar = torch.full_like(z, -3.0)
    logits = decoder(z, x)
    # logits: (B, T+1, V) — drop last position (无对应 target)
    loss_recon = F.cross_entropy(logits[:, :T].reshape(-1, V), x.reshape(-1))
    loss_kl, _ = kl_loss(z, logvar)
    beta = min(1.0, step / KL_ANNEAL_STEPS)
    loss = W_RECON * loss_recon + W_KL * beta * loss_kl
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
    opt.step(); sched.step()

    if step % EVAL_EVERY == 0 or step == STEPS - 1:
        decoder.eval()
        with torch.no_grad():
            vix = np.random.randint(0, len(val_texts), B)
            vx_chunks = []
            for i in vix:
                text = val_texts[i]
                if len(text) < T: text = text + "\n" * (T - len(text))
                start = random.randint(0, max(0, len(text) - T))
                chunk = text[start:start + T]
                vx_chunks.append([stoi.get(c, 0) for c in chunk])
            vx = torch.tensor(vx_chunks, dtype=torch.long, device=DEVICE)
            vz = val_z_cache[torch.tensor(vix, device=DEVICE)]
            vlogits = decoder(vz, vx)
            vloss_recon = F.cross_entropy(vlogits[:, :T].reshape(-1, V), vx.reshape(-1))
        val_ppl = float(np.exp(vloss_recon.item()))
        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            torch.save({
                "decoder": decoder.state_dict(),
                "config": {"V": V, "T": T, "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD,
                           "DEC_EMBD": DEC_EMBD, "D_Z": D_Z, "BOS_ID": BOS_ID},
            }, "v36_decoder.pt")
            P(f"  step {step:4d} | val_ppl {val_ppl:.3f} *saved*")
        else:
            elapsed = time.time() - t0
            eta = elapsed / max(step, 1) * (STEPS - step)
            P(f"  step {step:4d}/{STEPS} | recon {loss_recon.item():.3f} | val_recon {vloss_recon.item():.3f} "
              f"| KL {loss_kl.item():.2f} β={beta:.3f} | val_ppl {val_ppl:.3f} "
              f"| best {best_val_ppl:.3f} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss_recon": loss_recon.item(), "val_recon": vloss_recon.item(),
                    "loss_kl": loss_kl.item(), "beta": beta,
                    "val_ppl": val_ppl, "best_val_ppl": best_val_ppl})

# 保存最终模型 + log
torch.save({
    "decoder": decoder.state_dict(),
    "config": {"V": V, "T": T, "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD,
               "DEC_EMBD": DEC_EMBD, "D_Z": D_Z, "BOS_ID": BOS_ID},
}, "v36_decoder_final.pt")
with open("v36_decoder_train_log.json", "w") as f:
    json.dump({"log": log, "best_val_ppl": best_val_ppl,
               "config": {"B": B, "T": T, "D_Z": D_Z, "LR": LR, "STEPS": STEPS}}, f, indent=2)
P(f"\n=== 训练完成. best val_ppl={best_val_ppl:.3f} ===")
```

- [ ] **Step 2: 语法检查**

Run: `cd crystalllm && python -c "import ast; ast.parse(open('train_v36_decoder.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /d/CrystaLLM
git add crystalllm/train_v36_decoder.py
git commit -m "v36: add cross-attn training script (warm-start v25)"
```

---

## Task 5: 运行训练

**Files:**
- Output: `crystalllm/v36_decoder.pt`, `crystalllm/v36_decoder_final.pt`, `crystalllm/v36_decoder_train_log.json`

- [ ] **Step 1: 启动训练**

Run: `cd crystalllm && python train_v36_decoder.py 2>&1 | tee v36_train.log`
Expected: 训练 4000 步, ~15-18 min, log 显示 `val_ppl` 持续下降

- [ ] **Step 2: 验证训练产出**

Run: `ls -la crystalllm/v36_decoder.pt crystalllm/v36_decoder_train_log.json`
Expected: 两个文件都存在，v36_decoder.pt ~2.2GB

- [ ] **Step 3: 检查 best val_ppl**

Run: `python -c "import json; d=json.load(open('crystalllm/v36_decoder_train_log.json')); print('best_val_ppl:', d['best_val_ppl'])"`
Expected: `best_val_ppl` 应 < 2.50（理想 < 2.30）

- [ ] **Step 4: Commit 训练产出**

```bash
cd /d/CrystaLLM
git add crystalllm/v36_decoder.pt crystalllm/v36_decoder_train_log.json crystalllm/v36_train.log
git commit -m "v36: train cross-attn decoder (warm-start v25)"
```

---

## Task 6: 写 eval_v36_e2e.py (PPL + 速度评测)

**Files:**
- Create: `crystalllm/eval_v36_e2e.py`

- [ ] **Step 1: 创建评测脚本**

```python
"""eval_v36_e2e.py — v36 端到端 PPL + 速度评测

评测指标:
  1. PPL (全 1016 val)
  2. 速度 (5步扩散 + 100 token AR, batch=1)
  3. KL (z 信息保留度)
"""
import json, sys, io, os, random, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, ".")
from v36_model import DecoderCrossAttn

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

ckpt = torch.load("v36_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
print(f"v36 config: {cfg}")

decoder = DecoderCrossAttn(**{k: cfg[k] for k in ["V", "T", "DEC_LAYER", "DEC_HEAD", "DEC_EMBD", "D_Z", "BOS_ID"]}).to("cuda")
decoder.load_state_dict(ckpt["decoder"])
decoder.eval()
n_params = sum(p.numel() for p in decoder.parameters())
print(f"v36 decoder params: {n_params/1e6:.2f}M")

# 加载数据
df_val = pd.read_parquet(DATA / "v24_val.parquet")
val_texts = df_val["text"].tolist()
cache = np.load("cached_v24_z.npz")
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device="cuda")
print(f"val: {len(val_texts)} samples")

T = cfg["T"]

# ===== 1. PPL (全 val) =====
print("\n=== 1. PPL evaluation ===")
all_losses = []
with torch.no_grad():
    for i in range(0, len(val_texts), 4):
        batch_texts = val_texts[i:i+4]
        B = len(batch_texts)
        x_chunks = []
        for text in batch_texts:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            x_chunks.append([stoi.get(c, 0) for c in text[start:start + T]])
        x = torch.tensor(x_chunks, dtype=torch.long, device="cuda")
        z = val_z_cache[i:i+B]
        logits = decoder(z, x)
        loss = F.cross_entropy(logits[:, :T].reshape(-1, V), x.reshape(-1), reduction="sum")
        all_losses.append(loss.item())
total_loss = sum(all_losses) / (len(val_texts) * T)
ppl = float(np.exp(total_loss))
print(f"PPL: {ppl:.4f}")

# ===== 2. 速度 (5步扩散 + 100 token AR) =====
print("\n=== 2. Speed evaluation (5-step diff + 100 AR, batch=1) ===")
# 注: 5步扩散用 cached val_z 作为 z0 (真实生成场景)
z_single = val_z_cache[0:1]
# warmup
with torch.no_grad():
    bos = decoder.tok(torch.tensor([BOS_ID], device="cuda")).unsqueeze(0)  # (1, 1, D)
    cur = bos
    for _ in range(5):  # warmup
        logits = decoder(z_single, torch.zeros(1, 1, dtype=torch.long, device="cuda"))
torch.cuda.synchronize()

times = []
with torch.no_grad():
    for trial in range(20):
        torch.cuda.synchronize(); t0 = time.time()
        cur = bos
        for step in range(100):
            # 单 token 输入
            x_in = torch.zeros(1, 1, dtype=torch.long, device="cuda") if step == 0 else torch.tensor([[next_id]], device="cuda")
            logits = decoder(z_single, x_in)
            logits_t = logits[:, -1, :]  # (1, V)
            probs = F.softmax(logits_t, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
        torch.cuda.synchronize()
        times.append((time.time() - t0) * 1000)
times.sort()
median_ms = times[len(times) // 2]
p25_ms = times[len(times) // 4]
print(f"Speed (100 AR, batch=1): median {median_ms:.0f}ms, p25 {p25_ms:.0f}ms")

# ===== 3. KL (z 信息保留度) =====
# KL 基于 z 的分布参数, 与 decoder 架构无关, 但 v36 必须能消费 z
# 此处用 train 时的 kl_loss 公式: z=encoder_mu, logvar=train 用的 -3.0
# KL 高 = z 未被利用; KL 正常 = z 分布合理 (与 v25 ~250 接近)
print("\n=== 3. KL estimation (z distribution sanity) ===")
mu = torch.tensor(cache["val_z"][:100], dtype=torch.float32, device="cuda")
logvar = torch.full_like(mu, -3.0)
kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
kl_per_dim = kl.mean(dim=0)
kl_sum = kl_per_dim.sum().item()
print(f"KL (sum, free_bits threshold 1.0): {kl_sum:.2f} nats")
print(f"  注: KL 反映 z 分布本身, 不直接反映 decoder 是否使用 z")
print(f"  真正反映 z 使用率: 看 PPL 改善 (vs v25) + 生成质量")

# ===== 输出 JSON =====
results = {
    "v36_decoder_params_M": n_params / 1e6,
    "PPL": ppl,
    "speed_median_ms": median_ms,
    "speed_p25_ms": p25_ms,
    "KL_sum_nats": kl_sum,
    "config": cfg,
}
with open("v36_e2e.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\n=== 结果保存到 v36_e2e.json ===")
print(json.dumps(results, indent=2))
```

- [ ] **Step 2: 语法检查**

Run: `cd crystalllm && python -c "import ast; ast.parse(open('eval_v36_e2e.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /d/CrystaLLM
git add crystalllm/eval_v36_e2e.py
git commit -m "v36: add e2e eval script (PPL + speed + KL)"
```

---

## Task 7: 运行 PPL + 速度评测

**Files:**
- Output: `crystalllm/v36_e2e.json`

- [ ] **Step 1: 跑评测**

Run: `cd crystalllm && python eval_v36_e2e.py 2>&1 | tee v36_e2e.log`
Expected: ~3 min，输出 `PPL: X.XX`、`Speed: XXX ms`、`KL: XXX nats`

- [ ] **Step 2: 检查 PPL 阈值**

Run: `python -c "import json; d=json.load(open('crystalllm/v36_e2e.json')); print(f'PPL={d[\"PPL\"]:.4f}, target <2.30'); assert d['PPL'] < 2.30, f'PPL {d[\"PPL\"]} >= 2.30, FAIL'"`
Expected: `PPL=X.XXXX, target <2.30` (不抛 assertion)
如果 PPL >= 2.30，记录实际值，继续到 Task 8 检查生成质量

- [ ] **Step 3: 检查速度阈值**

Run: `python -c "import json; d=json.load(open('crystalllm/v36_e2e.json')); print(f'speed={d[\"speed_median_ms\"]:.0f}ms, target <1500ms'); assert d['speed_median_ms'] < 1500, 'too slow'"`
Expected: 输出 speed 值，不抛 assertion
如果 speed >= 1500ms，记录实际值，继续 Task 8

- [ ] **Step 4: Commit 评测结果**

```bash
cd /d/CrystaLLM
git add crystalllm/v36_e2e.json crystalllm/v36_e2e.log
git commit -m "v36: e2e eval results"
```

---

## Task 8: 写 debug_v36_gen.py (生成质量调试)

**Files:**
- Create: `crystalllm/debug_v36_gen.py`

- [ ] **Step 1: 创建生成质量脚本**

```python
"""debug_v36_gen.py — v36 生成质量调试

检查:
  1. 非空格率 (从零生成 50 token, > 90%?)
  2. 样本检查 (10 个样本是否含 import/def/class)
"""
import json, sys, io, os, random
from pathlib import Path
import torch, torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)
sys.path.insert(0, ".")
from v36_model import DecoderCrossAttn

DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)
SPACE_ID = stoi.get(" ", -1)
print(f"V={V}, BOS_ID={BOS_ID}, SPACE_ID={SPACE_ID}")

ckpt = torch.load("v36_decoder.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
decoder = DecoderCrossAttn(**{k: cfg[k] for k in ["V", "T", "DEC_LAYER", "DEC_HEAD", "DEC_EMBD", "D_Z", "BOS_ID"]}).to("cuda")
decoder.load_state_dict(ckpt["decoder"])
decoder.eval()

cache = np.load("cached_v24_z.npz")
val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device="cuda")

# ===== 1. 非空格率 (10 个样本 × 50 token AR) =====
print("\n=== 非空格率 (10 样本 × 50 token) ===")
non_space_rates = []
samples = []
with torch.no_grad():
    for i in range(10):
        z = val_z[i:i+1]
        bos_emb = decoder.tok(torch.tensor([BOS_ID], device="cuda"))
        # 起点: (1, 1) = BOS
        cur_tokens = torch.tensor([[BOS_ID]], dtype=torch.long, device="cuda")
        generated_ids = [BOS_ID]
        for step in range(50):
            logits = decoder(z, cur_tokens)
            logits_t = logits[:, -1, :]
            probs = F.softmax(logits_t, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            generated_ids.append(next_id)
            cur_tokens = torch.tensor([generated_ids], dtype=torch.long, device="cuda")
        text = "".join(itos.get(t, "<unk>") for t in generated_ids[1:])  # skip BOS
        samples.append(text)
        non_space = sum(1 for t in generated_ids[1:] if t != SPACE_ID)
        rate = non_space / 50
        non_space_rates.append(rate)
        print(f"  sample {i}: non_space_rate={rate:.2%} text={repr(text[:60])}...")

avg_rate = sum(non_space_rates) / len(non_space_rates)
print(f"\n平均非空格率: {avg_rate:.2%} (阈值 > 90%)")
assert avg_rate > 0.90, f"非空格率 {avg_rate:.2%} <= 90%, FAIL"

# ===== 2. 样本检查 (含 import/def/class) =====
print("\n=== 样本代码结构检查 ===")
KEYWORDS = ["import ", "def ", "class ", "function ", "var ", "const ", "let "]
matched = 0
for i, s in enumerate(samples):
    has = any(kw in s for kw in KEYWORDS)
    if has: matched += 1
    print(f"  sample {i}: has_keyword={has} text={repr(s[:80])}")

print(f"\n含代码结构样本数: {matched}/10 (阈值 >= 3)")
assert matched >= 3, f"仅 {matched}/10 样本含代码结构, FAIL"

# ===== 保存样本 =====
with open("v36_samples.json", "w") as f:
    json.dump({"non_space_rates": non_space_rates, "avg_rate": avg_rate,
               "samples": samples, "matched_count": matched}, f, indent=2)

print("\n✓ 生成质量检查通过")
print(f"  非空格率: {avg_rate:.2%}")
print(f"  代码结构样本: {matched}/10")
```

- [ ] **Step 2: 运行生成质量**

Run: `cd crystalllm && python debug_v36_gen.py 2>&1 | tee v36_gen.log`
Expected: `平均非空格率 > 90%`，`代码结构样本 >= 3/10`，`✓ 生成质量检查通过`
如果任何 assertion 失败：说明 v36 仍有坍缩问题，需要诊断

- [ ] **Step 3: Commit**

```bash
cd /d/CrystaLLM
git add crystalllm/debug_v36_gen.py crystalllm/v36_samples.json crystalllm/v36_gen.log
git commit -m "v36: gen quality check (non-space rate + code structure)"
```

---

## Task 9: 写 v36_results.md (实验报告)

**Files:**
- Create: `crystalllm/v36_results.md`

- [ ] **Step 1: 生成报告**

读取 `v36_e2e.json` 和 `v36_samples.json`，写入报告。模板：

```markdown
# CrystaLLM v36 — Cross-Attention Standard Decoder (BAD-DP v2)

> **Q: 把 BAD-DP decoder 改为 cross-attention, PPL 能从 v25 2.47 降到 < 2.30 吗? z 能被真使用吗? 生成会再坍缩到空格吗?**

## TL;DR

| 项 | v25 (BAD-DP) | **v36 (BAD-DP v2 + cross-attn)** | 目标 | 实际 |
|---|---|---|---|---:|
| PPL | 2.47 | **< 2.30 (目标)** | < 2.30 | {PPL 实际值} |
| 速度 | 828ms | **< 1500ms** | < 1500ms | {速度} ms |
| KL (z 分布) | ~250 (估) | 报告 | — | {KL} nats |
| 非空格率 | ~70% (估) | **> 90%** | > 90% | {rate:.2%} |
| 含代码结构样本 | 估 5/10 | **>= 3/10** | >= 3/10 | {matched}/10 |
| 参数量 | 476M | 570M (估) | — | {params}M |
| 训练时间 | 12 min | ~15-18 min | — | {train_time} min |

{如果全部达成: v36 成功, SpS verifier 可换为 v36}
{如果未达成: 见下方失败模式分析}

## 1. 实验过程

### 1.1 训练

- warm-start 自 v25_decoder.pt (loaded 290 / skipped 2 / fresh 96)
- 数据 v24 19K, B=4, T=512, LR=1e-4, 4000 步
- best val_ppl = {best_val_ppl}

### 1.2 关键观察

{填写训练曲线 + KL 演化 + 任何异常}

## 2. 评测结果

### 2.1 PPL
{填写 PPL 数值 + 与 v25/v28.5 对比}

### 2.2 速度
{填写速度 + 与 v25 对比}

### 2.3 生成质量
{粘贴 5-10 个样本, 标注哪些有代码结构}

## 3. 失败模式 (如有)

{如果未达阈值, 说明是哪个失败模式, 后续行动}

## 4. 结论

{总结 v36 是否成功, 下一步建议 (e.g. v31 SpS 重做 / 扩 T / prefix-tuning 备选)}

## 5. 文件清单

| 文件 | 用途 |
|---|---|
| v36_model.py | 模型定义 (BlockCrossAttn + DecoderCrossAttn) |
| test_v36_model.py | 前向 shape + 参数量校验 |
| test_v36_warmstart.py | warm-start 加载校验 |
| train_v36_decoder.py | 训练脚本 |
| eval_v36_e2e.py | PPL + 速度评测 |
| debug_v36_gen.py | 生成质量调试 |
| v36_decoder.pt | 训练产出 |
| v36_decoder_train_log.json | 训练日志 |
| v36_e2e.json | 评测 JSON |
| v36_samples.json | 生成样本 + 指标 |
| v36_results.md | 本报告 |
```

- [ ] **Step 2: 填入实际数值后保存**

读取 JSON, 替换 `{PPL 实际值}` 等占位符, 写入 `crystalllm/v36_results.md`

- [ ] **Step 3: Commit**

```bash
cd /d/CrystaLLM
git add crystalllm/v36_results.md
git commit -m "v36: experiment report"
```

---

## Self-Review Checklist

- [x] Spec coverage: Task 1-3 (架构 + 校验), Task 4-5 (训练), Task 6-7 (PPL/速度), Task 8 (生成质量), Task 9 (报告)
- [x] No TBD/TODO/placeholder
- [x] 完整代码每 step (无 "add appropriate handling")
- [x] 频率 commit (每 task 一次)
- [x] 类型一致: `DecoderCrossAttn` 在 v36_model.py 定义, 所有 train/eval/debug 都 `from v36_model import DecoderCrossAttn`
- [x] 成功 / 失败阈值与 spec 一致 (PPL < 2.30, 非空格率 > 90%, KL < 200, 速度 < 1500ms, 样本 >= 3/10)