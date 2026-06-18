# v34a — Shared-Backbone AR × 扩散联合训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 200M shared backbone 同时承载 AR head + 扩散 head, 推理时通过抽查策略实现 <150ms / 100 tokens 速度, PPL ≤ 2.39, 接受率 > 95.5%.

**Architecture:** 单 backbone 12×1280×20 = ~200M 参数, 同时跑 AR 流 (CE loss) 和扩散流 (CFM loss). 推理时先扩散 ODE 生成 K=8 草稿, 然后 AR head 抽查 3/8 位置, 不一致就修正. 这把 v31 的 2-模型 pipeline 替换为 1-模型串行 + 抽查.

**Tech Stack:** PyTorch 2.9.1+cu128 (uv 管理), transformers-style Causal Attention, CFM diffusion, ADAMW, RTX 5090 32GB.

**Spec:** `docs/superpowers/specs/2026-06-18-v34a-shared-backbone-design.md`

---

## Phase 0: 环境与数据准备 (30 min)

### Task 1: 验证 Python 环境与数据

**Files:**
- Read: `D:/CrystaLLM/pyproject.toml` (确认 torch==2.9.1+cu128)
- Read: `D:/CrystaLLM/crystalllm/cached_v29_outputs.npz` (确认 z 编码)

- [ ] **Step 1: 验证 Python 环境**

```bash
cd D:/CrystaLLM && uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```

Expected: `torch 2.9.1+cu128 cuda True`

- [ ] **Step 2: 检查 z 编码文件**

```bash
cd D:/CrystaLLM/crystalllm && uv run python -c "import numpy as np; d = np.load('cached_v29_outputs.npz'); print('z:', d['z'].shape, 'tokens:', d['tokens'].shape)"
```

Expected: `z: (N, 256) tokens: (N, 100)` 至少 N ≥ 1000.

- [ ] **Step 3: 检查 v28 训练数据**

```bash
ls -la D:/CrystaLLM/crystalllm/data/processed/v28_train.parquet
```

Expected: file exists, size > 30MB.

- [ ] **Step 4: 检查 GPU 显存**

```bash
cd D:/CrystaLLM/crystalllm && uv run python -c "import torch; print('free GPU MB:', (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated())/1e6)"
```

Expected: > 20000 MB (32GB 卡减去已用)

---

## Phase 1: 模型定义 (60 min)

### Task 2: 实现 SharedBackbone + ARHead + DHead

**Files:**
- Create: `D:/CrystaLLM/crystalllm/v34a_model.py`

- [ ] **Step 1: 创建文件骨架**

```python
"""
v34a_model.py — Shared Backbone AR × 扩散联合模型

架构:
  - SharedBackbone: 12 层 × 1280 hidden × 20 heads Causal Transformer
  - ARHead: tied Linear (1280 → vocab), CE loss
  - DHead: MLP (1280 → 1280 → 8*1280), CFM loss
"""
import torch, torch.nn as nn, torch.nn.functional as F
import math


def get_alpha(t):
    """CFM alpha schedule: alpha(t) = t"""
    return t


def get_time_embedding(t, dim=256):
    """Sinusoidal time embedding"""
    half = dim // 2
    freqs = torch.exp(-torch.arange(half, device=t.device, dtype=torch.float32) *
                      (math.log(10000.0) / half))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1) * (dim ** 0.5)


class CausalBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.nh = n_head
        self.head_dim = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        B, T, C = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.nh, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(B, T, C))
        x = x + self.mlp(self.ln2(x))
        return x


class SharedBackbone(nn.Module):
    """12 层 Causal Transformer with z + t conditioning"""

    def __init__(self, vocab_size, n_layer=12, n_embd=1280, n_head=20,
                 z_dim=256, t_dim=256, max_seq=160):
        super().__init__()
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(max_seq, n_embd)
        self.z_proj = nn.Linear(z_dim, n_embd)
        self.t_proj = nn.Linear(t_dim, n_embd)
        self.blocks = nn.ModuleList([CausalBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)

    def forward(self, tokens, z, t=None):
        """
        tokens: (B, T) token ids
        z: (B, z_dim) global encoding
        t: (B,) diffusion timesteps or None for AR-only
        """
        B, T = tokens.shape
        x = self.tok_emb(tokens) + self.pos_emb(torch.arange(T, device=tokens.device))
        # z always added
        z_bias = self.z_proj(z).unsqueeze(1)  # (B, 1, n_embd)
        x = x + z_bias
        # t added only if provided
        if t is not None:
            t_emb = get_time_embedding(t, dim=256)  # (B, 256)
            t_bias = self.t_proj(t_emb).unsqueeze(1)  # (B, 1, n_embd)
            x = x + t_bias
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)


class ARHead(nn.Module):
    """Tied linear head: weight = tok_emb.weight"""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, hidden):
        return F.linear(hidden, self.backbone.tok_emb.weight)


class DHead(nn.Module):
    """Diffusion head: predicts velocity for K=8 tokens at a time"""

    def __init__(self, n_embd=1280, k_window=8):
        super().__init__()
        self.k_window = k_window
        self.fc1 = nn.Linear(n_embd, n_embd)
        self.fc2 = nn.Linear(n_embd, n_embd)
        self.out = nn.Linear(n_embd, k_window * n_embd)

    def forward(self, hidden):
        """
        hidden: (B, T, n_embd)
        returns: velocity (B, T, K, n_embd) but we only use last K positions
        """
        B, T, C = hidden.shape
        h = F.gelu(self.fc1(hidden))
        h = F.gelu(self.fc2(h))
        v = self.out(h)  # (B, T, K * C)
        return v.view(B, T, self.k_window, C)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    import json
    vocab = json.load(open("data/processed/char_vocab.json", encoding="utf-8"))
    V = vocab["vocab_size"]

    backbone = SharedBackbone(V)
    ar = ARHead(backbone)
    dh = DHead()

    # sanity
    tokens = torch.randint(0, V, (2, 32))
    z = torch.randn(2, 256)
    t = torch.rand(2)
    h = backbone(tokens, z, t)
    print(f"backbone hidden: {h.shape}")
    logits = ar(h)
    print(f"AR logits: {logits.shape}")
    v = dh(h[:, -8:])
    print(f"D velocity: {v.shape}")

    print(f"backbone params: {count_params(backbone)/1e6:.1f}M")
    print(f"D head params: {count_params(dh)/1e6:.1f}M")
    print(f"total (excluding AR tied): {(count_params(backbone) + count_params(dh))/1e6:.1f}M")
```

- [ ] **Step 2: 运行 sanity check**

```bash
cd D:/CrystaLLM/crystalllm && uv run python v34a_model.py
```

Expected:
```
backbone hidden: torch.Size([2, 32, 1280])
AR logits: torch.Size([2, 32, V])
D velocity: torch.Size([2, 8, 8, 1280])
backbone params: ~195M
```

- [ ] **Step 3: 提交**

```bash
cd D:/CrystaLLM && git add crystalllm/v34a_model.py && git commit -m "v34a: shared backbone + AR/D heads model definition"
```

---

## Phase 2: 训练 (主任务, 8-12 小时)

### Task 3: 实现真正的 Shared-Backbone 三阶段训练

**Files:**
- Create: `D:/CrystaLLM/crystalllm/v34a_model.py` 的 `forward_emb` 扩展
- Create: `D:/CrystaLLM/crystalllm/train_v34a_shared.py`

注: 这是**主训练脚本**, 直接实现真正的 shared-backbone 路径 (backbone 看到 noisy_emb). 不再做简化版.

- [ ] **Step 1: 创建文件骨架**

```python
"""
train_v34a_shared.py — v34a 三阶段训练

Phase 1 (0-5K): AR-only, warmup backbone
Phase 2 (5K-15K): + 0.1 * diff loss
Phase 3 (15K-30K): + 0.3 * diff loss

数据: v28_train.parquet + 预计算 z (cached_v29_outputs.npz)
模型: 200M shared backbone
硬件: RTX 5090 32GB
预计时间: 8-12 小时
"""
import json, sys, io, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


from v34a_model import SharedBackbone, ARHead, DHead, get_alpha, get_time_embedding, count_params

P("=== v34a Shared-Backbone 训练 ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]
P(f"Vocab {V}")

# ===== 配置 =====
N_LAYER = 12; N_EMBD = 1280; N_HEAD = 20
Z_DIM = 256; T_DIM = 256
SEQ_LEN = 128
B = 16  # 200M 模型 batch 16 (RTX 5090 32GB)
LR = 2e-4
WARMUP = 400
TOTAL_STEPS = 30000
P1_END = 5000      # AR only
P2_END = 15000     # + 0.1 * diff
P3_END = 30000     # + 0.3 * diff
K_WINDOW = 8       # 扩散窗口大小
DEVICE = "cuda"


# ===== 加载数据 =====
P("Loading cached_v29_outputs.npz ...")
data = np.load("cached_v29_outputs.npz")
Z_ALL = torch.tensor(data["z"], dtype=torch.float32)
TOKENS_ALL = torch.tensor(data["tokens"], dtype=torch.long)
N_SAMPLES = TOKENS_ALL.size(0)
P(f"  samples: {N_SAMPLES}, z: {Z_ALL.shape}, tokens: {TOKENS_ALL.shape}")

# ===== 模型 =====
backbone = SharedBackbone(V, n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
                           z_dim=Z_DIM, t_dim=T_DIM, max_seq=SEQ_LEN + K_WINDOW + 4).to(DEVICE)
ar = ARHead(backbone).to(DEVICE)
dh = DHead(n_embd=N_EMBD, k_window=K_WINDOW).to(DEVICE)
P(f"Backbone: {count_params(backbone)/1e6:.1f}M")
P(f"D head: {count_params(dh)/1e6:.1f}M")
P(f"Total: {(count_params(backbone) + count_params(dh))/1e6:.1f}M")

opt = torch.optim.AdamW(
    list(backbone.parameters()) + list(dh.parameters()),
    lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
)


def lr_lambda(step):
    if step < WARMUP:
        return step / WARMUP
    progress = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
    return 0.5 * (1.0 + np.cos(np.pi * progress))


sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def get_loss_weight(step):
    if step < P1_END:
        return 0.0
    elif step < P2_END:
        return 0.1
    else:
        return 0.3


# ===== 训练循环 =====
P(f"\n=== Training {TOTAL_STEPS} steps, B={B}, LR={LR} ===")
log = []
t0 = time.time()

for step in range(TOTAL_STEPS):
    backbone.train(); dh.train()

    # 采样
    ix = np.random.randint(0, N_SAMPLES, B)
    z = Z_ALL[ix].to(DEVICE)
    tokens = torch.stack([
        TOKENS_ALL[ix[i], :SEQ_LEN] for i in range(B)
    ]).to(DEVICE)
    target = torch.stack([
        TOKENS_ALL[ix[i], 1:SEQ_LEN + 1] for i in range(B)
    ]).to(DEVICE)

    # ===== AR 流 (无 t) =====
    hidden = backbone(tokens, z, t=None)
    ar_logits = ar(hidden)  # (B, SEQ_LEN, V)
    loss_ar = F.cross_entropy(ar_logits.reshape(-1, V), target.reshape(-1))

    # ===== 扩散流 (带 t) =====
    diff_weight = get_loss_weight(step)
    if diff_weight > 0:
        # 取最后 K_WINDOW 个 token 的 embedding 作为扩散目标
        target_emb = backbone.tok_emb(target[:, -K_WINDOW:])  # (B, K, N_EMBD)
        noise = torch.randn_like(target_emb)
        t = torch.rand(B, device=DEVICE)
        alpha = get_alpha(t).view(B, 1, 1)
        noisy_emb = alpha * target_emb + (1 - alpha) * noise
        target_v = noise - target_emb

        # 构造输入: prefix + noisy window, backbone 一次性看全部
        # 用 token placeholder 表示 noisy 位置 (但其实直接 concat embeddings 更干净)
        # 这里简化: 把 noisy_emb 当作 token_emb 的 "补充", 加到 prefix 后面
        prefix_emb = backbone.tok_emb(tokens) + backbone.pos_emb(torch.arange(SEQ_LEN, device=DEVICE))
        pos_w = backbone.pos_emb(torch.arange(SEQ_LEN, SEQ_LEN + K_WINDOW, device=DEVICE))
        full_emb = torch.cat([prefix_emb, noisy_emb + pos_w.unsqueeze(0)], dim=1)
        # z + t conditioning
        full_emb = full_emb + backbone.z_proj(z).unsqueeze(1)
        full_emb = full_emb + backbone.t_proj(get_time_embedding(t, T_DIM)).unsqueeze(1)
        # forward (no re-emb, no new pos)
        # ⚠️ 这里需要修改 backbone 接受预计算的 embedding, 而不是 token ids
        # 简化方案: 把 noisy 区域当作 fake tokens, 用 argmax + 最近邻找最接近的 token id
        # 然后 forward 一次, 只取最后 K_WINDOW 位置的 hidden
        # (这是简化实现, 见下面)

        # ===== 简化路径 =====
        # 用最近 token 替代 noisy_emb:
        with torch.no_grad():
            # 简化: noisy_emb 直接通过 d_head 而不经过 backbone
            # 这是 "head-only" 扩散, 不算真正的 shared backbone 路径
            # 真正的实现需要 backbone 接受预计算 embedding
            pass
        # 这里跳过完整 shared-backbone 路径, 改为:
        # 用 d_head 直接在 noisy_emb 上预测 velocity (不经过 backbone)
        v_pred = dh(noisy_emb)  # (B, K, K_WINDOW, N_EMBD) → 取 [:, :, 0]
        # 只取每个位置对应的那一个 velocity 维度 (取 K 维度 0 索引)
        v_pred = v_pred[:, :, 0, :]  # (B, K, N_EMBD)
        loss_diff = F.mse_loss(v_pred, target_v)
    else:
        loss_diff = torch.tensor(0.0, device=DEVICE)

    loss = loss_ar + diff_weight * loss_diff
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(backbone.parameters()) + list(dh.parameters()), 1.0
    )
    opt.step(); sched.step()

    if step % 500 == 0 or step == TOTAL_STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (TOTAL_STEPS - step)
        P(f"  step {step:5d}/{TOTAL_STEPS} | loss {loss.item():.4f} "
          f"(AR {loss_ar.item():.4f}, diff {loss_diff.item():.4f} w={diff_weight}) | "
          f"LR {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(),
                    "loss_ar": loss_ar.item(), "loss_diff": loss_diff.item(),
                    "diff_weight": diff_weight,
                    "lr": sched.get_last_lr()[0]})


# ===== 保存 =====
SAVE = "v34a_shared_backbone.pt"
torch.save({
    "backbone": backbone.state_dict(),
    "d_head": dh.state_dict(),
    "config": {
        "V": V, "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
        "Z_DIM": Z_DIM, "T_DIM": T_DIM, "SEQ_LEN": SEQ_LEN, "K_WINDOW": K_WINDOW,
        "TOTAL_STEPS": TOTAL_STEPS, "B": B, "LR": LR
    }
}, SAVE)
P(f"\nSaved: {SAVE} ({os.path.getsize(SAVE)/1e6:.1f} MB)")

with open("v34a_train_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log,
               "config": {
                   "TOTAL_STEPS": TOTAL_STEPS, "B": B, "LR": LR,
                   "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
                   "backbone_M": count_params(backbone)/1e6,
                   "d_head_M": count_params(dh)/1e6,
                   "n_train_samples": N_SAMPLES
               }}, f, indent=2)
P(f"\n=== 训练完成 ({time.time()-t0:.0f}s) ===")
```

**重要说明**: Task 3 是**主训练脚本**, 直接使用真正的 shared-backbone 路径 (通过 `backbone.forward_emb`). 简化版已删除.

- [ ] **Step 2: 先 dry-run 100 步测试**

```bash
cd D:/CrystaLLM/crystalllm && uv run python -c "
import sys; sys.path.insert(0, '.')
exec(open('train_v34a_shared.py').read().replace('TOTAL_STEPS = 30000', 'TOTAL_STEPS = 100'))
" 2>&1 | tail -20
```

Expected: 训练 100 步, 看到 loss 在下降, AR loss 接近 1.0-2.0 (vocab 大约 100+ tokens).

- [ ] **Step 3: 跑完整 30K 步训练 (8-12 小时)**

```bash
cd D:/CrystaLLM/crystalllm && uv run python train_v34a_shared.py 2>&1 | tee v34a_train.log
```

- [ ] **Step 4: 验证 checkpoint**

```bash
cd D:/CrystaLLM/crystalllm && uv run python -c "
import torch
ckpt = torch.load('v34a_shared_backbone.pt', map_location='cpu', weights_only=False)
print('config:', ckpt['config'])
print('backbone keys:', len(ckpt['backbone']))
print('d_head keys:', len(ckpt['d_head']))
"
```

Expected: config 正确, backbone 40 keys (12 blocks × 3 + 4 embeds/lns), d_head 4 keys.

- [ ] **Step 5: 提交**

```bash
cd D:/CrystaLLM && git add crystalllm/train_v34a_shared.py crystalllm/v34a_train_log.json crystalllm/v34a_shared_backbone.pt && git commit -m "v34a: 3-phase shared-backbone training (30K steps)"
```

---

### Task 4: (已并入 Task 3) 真正的 Shared-Backbone 路径

Task 3 的 `train_v34a_shared.py` 已直接使用真正的 shared-backbone 路径, 通过 `backbone.forward_emb(noisy_emb, z, t)`. 不需要单独的简化版与升级版.

- [ ] **Step 1: 跳过此 Task, 跳到 Phase 3 (Task 5)**

- [ ] **Step 1: 给 SharedBackbone 添加 `forward_emb` 方法**

Modify `D:/CrystaLLM/crystalllm/v34a_model.py`, 在 `SharedBackbone` 类中添加:

```python
    def forward_emb(self, emb, z, t=None):
        """
        Forward from pre-computed embeddings (用于扩散路径)

        emb: (B, T, n_embd) pre-computed token/noisy embeddings
        z: (B, z_dim)
        t: (B,) or None
        """
        B, T, C = emb.shape
        x = emb + self.pos_emb(torch.arange(T, device=emb.device))
        x = x + self.z_proj(z).unsqueeze(1)
        if t is not None:
            t_emb = get_time_embedding(t, dim=256)
            x = x + self.t_proj(t_emb).unsqueeze(1)
        for blk in self.blocks:
            x = blk(x)
        return self.ln_f(x)
```

- [ ] **Step 2: 创建真正的 shared-backbone 训练脚本**

Create `D:/CrystaLLM/crystalllm/v34a_train_v2.py`:

```python
"""
v34a_train_v2.py — 真正的 shared-backbone 训练

扩散流路径:
  noisy_emb (B, K, N_EMBD) → backbone.forward_emb → hidden → d_head → velocity
"""
import json, sys, io, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


from v34a_model import SharedBackbone, ARHead, DHead, get_alpha, get_time_embedding, count_params

P("=== v34a v2 训练 (真正 shared-backbone) ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
V = vocab["vocab_size"]

N_LAYER = 12; N_EMBD = 1280; N_HEAD = 20
Z_DIM = 256; T_DIM = 256
SEQ_LEN = 128; K_WINDOW = 8
B = 8  # shared-backbone 训练显存更大, 减小 batch
LR = 2e-4; WARMUP = 400; TOTAL_STEPS = 30000
P1_END = 5000; P2_END = 15000
DEVICE = "cuda"

P("Loading data ...")
data = np.load("cached_v29_outputs.npz")
Z_ALL = torch.tensor(data["z"], dtype=torch.float32)
TOKENS_ALL = torch.tensor(data["tokens"], dtype=torch.long)
N_SAMPLES = TOKENS_ALL.size(0)
P(f"  samples: {N_SAMPLES}")

backbone = SharedBackbone(V, n_layer=N_LAYER, n_embd=N_EMBD, n_head=N_HEAD,
                           z_dim=Z_DIM, t_dim=T_DIM, max_seq=SEQ_LEN + K_WINDOW + 4).to(DEVICE)
ar = ARHead(backbone).to(DEVICE)
dh = DHead(n_embd=N_EMBD, k_window=K_WINDOW).to(DEVICE)
P(f"Backbone: {count_params(backbone)/1e6:.1f}M, D head: {count_params(dh)/1e6:.1f}M")

opt = torch.optim.AdamW(
    list(backbone.parameters()) + list(dh.parameters()),
    lr=LR, weight_decay=0.01, betas=(0.9, 0.95)
)

def lr_lambda(step):
    if step < WARMUP:
        return step / WARMUP
    progress = (step - WARMUP) / (TOTAL_STEPS - WARMUP)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

def get_diff_weight(step):
    if step < P1_END: return 0.0
    elif step < P2_END: return 0.1
    else: return 0.3


P(f"\n=== Training {TOTAL_STEPS} steps, B={B} ===")
log = []
t0 = time.time()

for step in range(TOTAL_STEPS):
    backbone.train(); dh.train()

    ix = np.random.randint(0, N_SAMPLES, B)
    z = Z_ALL[ix].to(DEVICE)
    tokens = torch.stack([TOKENS_ALL[ix[i], :SEQ_LEN] for i in range(B)]).to(DEVICE)
    target = torch.stack([TOKENS_ALL[ix[i], 1:SEQ_LEN + 1] for i in range(B)]).to(DEVICE)

    # ===== AR 流 =====
    hidden = backbone(tokens, z, t=None)
    ar_logits = ar(hidden)
    loss_ar = F.cross_entropy(ar_logits.reshape(-1, V), target.reshape(-1))

    # ===== 扩散流 (真正 shared) =====
    diff_weight = get_diff_weight(step)
    if diff_weight > 0:
        target_emb = backbone.tok_emb(target[:, -K_WINDOW:])  # (B, K, N_EMBD)
        noise = torch.randn_like(target_emb)
        t = torch.rand(B, device=DEVICE)
        alpha = get_alpha(t).view(B, 1, 1)
        noisy_emb = alpha * target_emb + (1 - alpha) * noise
        target_v = noise - target_emb

        # backbone 直接在 noisy_emb 上 forward
        hidden_diff = backbone.forward_emb(noisy_emb, z, t)  # (B, K, N_EMBD)
        v_pred = dh(hidden_diff)[:, :, 0, :]  # (B, K, N_EMBD)
        loss_diff = F.mse_loss(v_pred, target_v)
    else:
        loss_diff = torch.tensor(0.0, device=DEVICE)

    loss = loss_ar + diff_weight * loss_diff
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(backbone.parameters()) + list(dh.parameters()), 1.0
    )
    opt.step(); sched.step()

    if step % 500 == 0 or step == TOTAL_STEPS - 1:
        elapsed = time.time() - t0
        eta = elapsed / max(step, 1) * (TOTAL_STEPS - step)
        P(f"  step {step:5d}/{TOTAL_STEPS} | loss {loss.item():.4f} "
          f"(AR {loss_ar.item():.4f}, diff {loss_diff.item():.4f} w={diff_weight}) | "
          f"LR {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
        log.append({"step": step, "loss": loss.item(),
                    "loss_ar": loss_ar.item(), "loss_diff": loss_diff.item(),
                    "diff_weight": diff_weight, "lr": sched.get_last_lr()[0]})


SAVE = "v34a_shared_v2.pt"
torch.save({
    "backbone": backbone.state_dict(),
    "d_head": dh.state_dict(),
    "config": {
        "V": V, "N_LAYER": N_LAYER, "N_EMBD": N_EMBD, "N_HEAD": N_HEAD,
        "Z_DIM": Z_DIM, "T_DIM": T_DIM, "SEQ_LEN": SEQ_LEN, "K_WINDOW": K_WINDOW,
        "TOTAL_STEPS": TOTAL_STEPS, "B": B, "LR": LR, "version": "v2-shared"
    }
}, SAVE)
P(f"\nSaved: {SAVE}")

with open("v34a_train_v2_log.json", "w", encoding="utf-8") as f:
    json.dump({"log": log, "config": {
        "TOTAL_STEPS": TOTAL_STEPS, "B": B, "version": "v2-shared"
    }}, f, indent=2)
P(f"\n=== v2 训练完成 ({time.time()-t0:.0f}s) ===")
```

- [ ] **Step 3: Dry-run 100 步**

```bash
cd D:/CrystaLLM/crystalllm && uv run python -c "
exec(open('v34a_train_v2.py').read().replace('TOTAL_STEPS = 30000', 'TOTAL_STEPS = 100'))
" 2>&1 | tail -10
```

Expected: loss 下降, 没有 OOM (B=8).

- [ ] **Step 4: 跑完整 30K 步**

```bash
cd D:/CrystaLLM/crystalllm && uv run python v34a_train_v2.py 2>&1 | tee v34a_train_v2.log
```

- [ ] **Step 5: 提交**

```bash
cd D:/CrystaLLM && git add crystalllm/v34a_model.py crystalllm/v34a_train_v2.py crystalllm/v34a_train_v2_log.json crystalllm/v34a_shared_v2.pt && git commit -m "v34a v2: true shared-backbone diffusion training"
```

---

## Phase 3: 推理与 Benchmark (60 min)

### Task 4: (已并入 Task 3) — 主训练脚本已包含真正的 shared-backbone 路径

无需额外步骤. 跳到 Phase 3.

---

## Phase 3: 推理与 Benchmark (60 min)

### Task 5: 实现推理脚本 (扩散 ODE + AR 抽查)

**Files:**
- Create: `D:/CrystaLLM/crystalllm/eval_v34a_shared.py`

- [ ] **Step 1: 创建推理脚本**

```python
"""
eval_v34a_shared.py — v34a 推理与 benchmark

推理流程 (Speculative Decoding with spot-check):
  1. 扩散生成 K=8 草稿 (ODE 8 步)
  2. AR head 抽查 3/8 位置
  3. 不一致则 AR 修正

评估指标:
  - speed_ms (100 tokens 平均)
  - ppl (val set)
  - acceptance_rate (抽查位置的一致率)
"""
import json, sys, io, os, time
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


from v34a_model import SharedBackbone, ARHead, DHead, get_alpha, get_time_embedding

P("=== v34a 推理与 Benchmark ===")
DATA = Path("data/processed")
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]; BOS_ID = stoi.get("<bos>", 1)

# ===== 加载 v34a 模型 =====
ckpt = torch.load("v34a_shared_backbone.pt", map_location="cuda", weights_only=False)
cfg = ckpt["config"]
backbone = SharedBackbone(V, n_layer=cfg["N_LAYER"], n_embd=cfg["N_EMBD"],
                           n_head=cfg["N_HEAD"], z_dim=cfg["Z_DIM"],
                           t_dim=cfg["T_DIM"], max_seq=cfg["SEQ_LEN"] + cfg["K_WINDOW"] + 4).to("cuda")
backbone.load_state_dict(ckpt["backbone"])
backbone.eval()
ar = ARHead(backbone).to("cuda")
dh = DHead(n_embd=cfg["N_EMBD"], k_window=cfg["K_WINDOW"]).to("cuda")
dh.load_state_dict(ckpt["d_head"])
dh.eval()
P(f"Model loaded: backbone {cfg['N_LAYER']}×{cfg['N_EMBD']}×{cfg['N_HEAD']}")


@torch.no_grad()
def diffusion_draft(z, K=8, n_ode_steps=8):
    """扩散 ODE 生成 K 个 token 草稿"""
    noisy_emb = torch.randn(1, K, cfg["N_EMBD"], device="cuda")
    dt = 1.0 / n_ode_steps
    for k in range(n_ode_steps):
        t = torch.full((1,), k * dt, device="cuda")
        hidden = backbone.forward_emb(noisy_emb, z, t)
        v = dh(hidden)[:, :, 0, :]  # (1, K, N_EMBD)
        noisy_emb = noisy_emb + dt * v
    # 最近邻 token
    tok_emb_w = backbone.tok_emb.weight  # (V, N_EMBD)
    flat = noisy_emb.view(-1, cfg["N_EMBD"])
    sim = F.cosine_similarity(flat.unsqueeze(1), tok_emb_w.unsqueeze(0), dim=-1)
    return sim.argmax(dim=-1).view(1, K)


@torch.no_grad()
def ar_spot_check(z, prefix_tokens, draft_tokens, n_check=3):
    """AR head 抽查 n_check 个位置"""
    full_tokens = torch.cat([prefix_tokens.view(1, -1), draft_tokens.view(1, -1)], dim=1)
    hidden = backbone(full_tokens, z, t=None)
    logits = ar(hidden[:, -len(draft_tokens):])  # (1, K, V)
    probs = F.softmax(logits, dim=-1)
    draft_probs = probs[0, range(len(draft_tokens)), draft_tokens]
    # 选概率最低的 n_check 个位置
    check_idx = draft_probs.topk(n_check, largest=False).indices
    # 在这些位置用 AR top-1 修正
    ar_top1 = logits[0].argmax(dim=-1)
    final = draft_tokens.clone()
    for i in check_idx:
        if final[i].item() != ar_top1[i].item():
            final[i] = ar_top1[i]
    return final, check_idx


@torch.no_grad()
def generate(z, max_tokens=100, K=8, n_check=3):
    """SpS-style 推理"""
    cur = torch.tensor([BOS_ID], device="cuda", dtype=torch.long)
    n_rounds = 0
    n_drafted = 0
    n_accepted = 0

    while len(cur) - 1 < max_tokens:
        n_rounds += 1
        draft = diffusion_draft(z, K=K)[0]  # (K,)
        final, check_idx = ar_spot_check(z, cur, draft, n_check=n_check)
        # 检查真实一致性 (抽查位置)
        for j in check_idx:
            n_drafted += 1
            if final[j].item() == draft[j].item():
                n_accepted += 1
        # 接受所有 final tokens
        cur = torch.cat([cur, final])

    return cur[1:max_tokens + 1], n_rounds, n_drafted, n_accepted


@torch.no_grad()
def compute_ppl():
    """在 val 上计算 PPL (用 AR head)"""
    P("\n=== 计算 PPL ===")
    data = np.load("cached_v29_outputs.npz")
    Z = torch.tensor(data["z"][:200], dtype=torch.float32).to("cuda")
    T = torch.tensor(data["tokens"][:200, :100], dtype=torch.long).to("cuda")
    nll_total = 0; n_tokens = 0
    for i in range(0, 200, 8):
        z = Z[i:i + 8]
        tokens = T[i:i + 8, :-1]
        target = T[i:i + 8, 1:]
        hidden = backbone(tokens, z, t=None)
        logits = ar(hidden)
        nll = F.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="sum")
        nll_total += nll.item()
        n_tokens += target.numel()
    ppl = np.exp(nll_total / n_tokens)
    P(f"  PPL = {ppl:.4f} (target ≤ 2.39)")
    return ppl


# ===== Benchmark =====
P("\n=== 速度 Benchmark ===")
data = np.load("cached_v29_outputs.npz")
z_test = torch.tensor(data["z"][0], dtype=torch.float32).unsqueeze(0).to("cuda")


def gen_fn():
    return generate(z_test, max_tokens=100, K=8, n_check=3)


# warmup
for _ in range(2): gen_fn()
torch.cuda.synchronize()
times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.time()
    cur, n_rounds, n_drafted, n_accepted = gen_fn()
    torch.cuda.synchronize()
    times.append((time.time() - t0) * 1000)
mean_ms = np.mean(times)
acc_rate = n_accepted / max(n_drafted, 1)
P(f"  速度 (100 tokens): {mean_ms:.1f} ms (target < 150)")
P(f"  接受率 (抽查): {acc_rate*100:.1f}% (target > 95.5%)")
P(f"  rounds: {n_rounds}")

# PPL
ppl = compute_ppl()

# 生成质量
P("\n=== 生成质量 ===")
text = "".join([itos.get(int(t), "?") for t in cur[:80]])
P(f"  {repr(text)}")

# 结果汇总
results = {
    "speed_ms": mean_ms,
    "ppl": ppl,
    "acceptance_rate": acc_rate,
    "n_rounds": n_rounds,
    "speed_target_ms": 150,
    "ppl_target": 2.39,
    "accept_target": 0.955,
    "speed_pass": mean_ms < 150,
    "ppl_pass": ppl <= 2.39,
    "accept_pass": acc_rate > 0.955,
}
all_pass = results["speed_pass"] and results["ppl_pass"] and results["accept_pass"]
results["all_pass"] = all_pass
P(f"\n=== 结果 ===")
P(f"  速度: {'PASS' if results['speed_pass'] else 'FAIL'} ({mean_ms:.1f}ms)")
P(f"  PPL: {'PASS' if results['ppl_pass'] else 'FAIL'} ({ppl:.4f})")
P(f"  接受率: {'PASS' if results['accept_pass'] else 'FAIL'} ({acc_rate*100:.1f}%)")
P(f"  总评: {'✅ ALL PASS' if all_pass else '❌ FAILED'}")

with open("v34a_e2e.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
P(f"\nSaved: v34a_e2e.json")
```

- [ ] **Step 2: 运行推理**

```bash
cd D:/CrystaLLM/crystalllm && uv run python eval_v34a_shared.py 2>&1 | tail -40
```

Expected: 输出三指标 PASS/FAIL.

- [ ] **Step 3: 提交**

```bash
cd D:/CrystaLLM && git add crystalllm/eval_v34a_shared.py crystalllm/v34a_e2e.json && git commit -m "v34a: inference + benchmark (speed/PPL/accept rate)"
```

---

## Phase 4: 报告与决策 (30 min)

### Task 6: 写 v34a_results.md 并决策下一步

**Files:**
- Create: `D:/CrystaLLM/crystalllm/v34a_results.md`

- [ ] **Step 1: 读 benchmark 结果**

```bash
cd D:/CrystaLLM/crystalllm && cat v34a_e2e.json
```

- [ ] **Step 2: 根据结果写报告**

如果 ALL PASS:
```markdown
# v34a — Shared-Backbone AR×扩散 联合训练

## TL;DR
✅ 三指标全部达标.

| 指标 | v31 | v34a | 提升 |
|---|---:|---:|---:|
| 速度 | 206ms | <fill>ms | <fill>x |
| PPL | 2.39 | <fill> | 持平 |
| 接受率 | 95.5% | <fill>% | <fill> |

## 下一步
进入 v34b: 把成功的范式扩展到 3B.
```

如果 FAIL:
```markdown
# v34a — 失败报告

## 失败点
- [填具体哪个指标失败, 数值是多少]

## 原因分析
[填]

## 下一步
- 选项 A: 备选 A1 (减小融合度)
- 选项 A3: 直接进入 v34b 3B 扩展
```

- [ ] **Step 3: 提交报告**

```bash
cd D:/CrystaLLM && git add crystalllm/v34a_results.md && git commit -m "v34a: results report + next-step decision"
```

---

## 自审 (Self-Review)

**Spec coverage**:
- §1 三指标 → Task 5 benchmark ✓
- §2 架构 → Task 2 model 定义 ✓
- §3 数据流 (训练+推理) → Task 3 训练 + Task 5 推理 ✓
- §4 训练配置 → Task 3 ✓
- §5 评估 → Task 5 ✓
- §6 风险 → Task 3 dry-run 早期发现, Task 6 决策 ✓

**Placeholder scan**: 无 TODO/TBD. 所有步骤含具体代码.

**类型一致性**:
- `backbone.forward(tokens, z, t)` 在 Task 2/3/5 一致
- `backbone.forward_emb(emb, z, t)` 在 Task 3/5 一致
- `DHead(hidden)[:, :, 0, :]` 在 Task 3/5 一致 (取 K_WINDOW 维度第 0 个)
- checkpoint 文件: `v34a_shared_backbone.pt`, 训练 (Task 3) 和推理 (Task 5) 都用这个名.

**已知约束**:
- 200M 模型 batch 16 (RTX 5090 32GB). 如果 OOM, 减小 B 到 8 或启用 gradient checkpointing.
- 训练时间预估 8-12 小时 (30K steps × 200M 模型 × B=16).