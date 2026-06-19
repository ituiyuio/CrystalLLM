# v37 Zero-z Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 通过 zero-z ablation 量化回答 "CrystaLLM decoder 是否真消费 z 信号",基于此决定 v37+ 走向 (战略重定位 / 二次 brainstorm / v37 prefix-tuning).

**Architecture:** 复用 v25_decoder.pt 与 v36_decoder.pt 现有 checkpoint,编写统一的 `zero_z_eval.py` 接受 `--z_mode {encoded, zero}` 参数。zero 模式下,decoder forward 入口把 z 强制替换为 `torch.zeros(B, D_Z)`,其他信号(pos embed/KV cache/head)不动。测量 4 个数据点 (A1-A4) 派生 ΔPPL,基于决策矩阵分流.

**Tech Stack:** PyTorch 2.9.1 (cu128), 项目既有 eval 脚本风格 (pandas + parquet + numpy cache + torch.no_grad).

**Spec:** `docs/superpowers/specs/2026-06-19-v37-zero-z-ablation-design.md`

---

## File Structure

| 文件 | 角色 | 状态 |
|---|---|---|
| `crystalllm/versions/v37/README.md` | v37 目的说明 | 新建 |
| `crystalllm/versions/v37/pipeline/zero_z_eval.py` | 统一 eval 脚本 (--checkpoint, --z_mode) | 新建 |
| `crystalllm/versions/v37/evaluation/gen_samples.py` | 生成样本 + 非空格率 + 代码结构 | 新建 |
| `crystalllm/versions/v37/v37_e2e.json` | 4 PPL + 派生指标输出 | 由脚本生成 |
| `crystalllm/versions/v37/v37_samples.json` | 4 × 10 生成样本输出 | 由脚本生成 |
| `crystalllm/versions/v37/v37_decision.md` | 决策报告 (实验结果 + 推荐) | 新建 |

**不创建新 checkpoint. 不创建新训练脚本. 不修改 v25/v36 现有 eval 脚本.**

---

## Task 1: 创建 v37 目录与 README

**Files:**
- Create: `crystalllm/versions/v37/README.md`

- [ ] **Step 1.1: 创建 v37 目录**

```bash
mkdir -p crystalllm/versions/v37/pipeline crystalllm/versions/v37/evaluation
```

- [ ] **Step 1.2: 写 README.md**

写入 `crystalllm/versions/v37/README.md`:

```markdown
# v37 — Zero-z Ablation (决策门)

## 目的
通过 zero-z ablation 量化回答 "decoder 是否真消费 z 信号", 基于此分流 v37+ 走向.

## 不做什么
- ❌ 不训练新模型
- ❌ 不修任何架构
- ❌ 不动 v25/v36 现有 checkpoint

## 复用资产
- `crystalllm/versions/v25/v25_decoder.pt` (476M, PPL 2.47)
- `crystalllm/versions/v36/v36_decoder.pt` (570M, PPL 2.81)
- `crystalllm/data/processed/cached_v24_z.npz` (val_z, n=1016)
- `crystalllm/data/processed/v24_val.parquet` (val_texts)
- `crystalllm/data/processed/char_vocab.json` (vocab)

## 决策矩阵
参见 spec §5.

## 实验矩阵
| 编号 | ckpt | z_mode | 用途 |
|---|---|---|---|
| A1 | v25 | encoded | baseline (复用 v25_e2e.json 2.47) |
| A2 | v25 | zero | 主要测量 |
| A3 | v36 | encoded | baseline (复用 v36_e2e.json 2.81) |
| A4 | v36 | zero | cross-attn 验证 |
```

- [ ] **Step 1.3: 验证文件已创建**

```bash
ls -la crystalllm/versions/v37/
ls -la crystalllm/versions/v37/pipeline/ crystalllm/versions/v37/evaluation/
```
预期: README.md 存在, 两个子目录为空.

- [ ] **Step 1.4: Commit**

```bash
git add crystalllm/versions/v37/README.md
git commit -m "v37: init directory + README (zero-z ablation decision gate)"
```

---

## Task 2: 编写 zero_z_eval.py 脚本骨架

**Files:**
- Create: `crystalllm/versions/v37/pipeline/zero_z_eval.py`

- [ ] **Step 2.1: 复制 v25 eval 的 decoder 类定义**

写入 `crystalllm/versions/v37/pipeline/zero_z_eval.py` (完整文件):

```python
# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""zero_z_eval.py — v37 zero-z ablation 统一评估脚本

测量 decoder 在 z 强制为零向量时的 PPL 退化. 复用 v25/v36 checkpoint,
接受 --checkpoint {v25, v36} --z_mode {encoded, zero}.

zero 模式下, 在 decoder forward 入口把 z 替换为 torch.zeros(B, D_Z).
其他信号 (pos embed, KV cache, head) 不动.

预期:
  - v25 encoded: PPL 2.47 (已有)
  - v25 zero:    PPL 接近 2.49-2.55 (训练日志估算)
  - v36 encoded: PPL 2.81 (已有)
  - v36 zero:    与 v25 zero 对比, 验证 cross-attn 是否真用 z
"""
import argparse
import json
import sys
import io
import os
import random
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42); random.seed.seed(42) if hasattr(random, 'seed') else random.seed(42)

V37_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = V37_DIR.parent.parent  # crystalllm/versions/v37 -> crystalllm/versions -> crystalllm -> CrystaLLM
DATA = PROJECT_ROOT / "data" / "processed"

# ============================================================
# v25 Decoder (BAD-DP, z as pos 0 token)
# ============================================================
class BlockCausal(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x):
        B_, T_, C = x.shape
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        x = x + s.mlp(s.ln2(x)); return x


class DecoderV25(nn.Module):
    """v25 BAD-DP: z as pos 0 single token."""
    def __init__(s, V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID):
        super().__init__()
        s.T, s.BOS_ID = T, BOS_ID
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, z, x):
        B_, T_ = x.shape
        z_emb = s.z_to_emb(z).unsqueeze(1)
        bos_emb = s.tok(torch.tensor([s.BOS_ID], device=x.device)).expand(B_, 1, -1)
        x_emb = s.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        for b in s.blocks: inp = b(inp)
        logits = s.head(s.ln_f(inp))
        return logits[:, 1:T_ + 1]


# ============================================================
# v36 Decoder (cross-attn, z as K/V per block)
# ============================================================
class DecoderV36(nn.Module):
    """v36 BAD-DP v2: z consumed via cross-attn in every block."""
    def __init__(s, V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID):
        super().__init__()
        s.T, s.BOS_ID, s.D_Z = T, BOS_ID, D_Z
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCrossAttn(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight
    def forward(s, z, x):
        B_, T_ = x.shape
        z_emb = s.z_to_emb(z).unsqueeze(1)  # [B, 1, D]
        # broadcast z_emb to [B, T+1, D] for cross-attn K/V
        z_kv = z_emb.expand(-1, T_ + 1, -1)
        bos_emb = s.tok(torch.tensor([s.BOS_ID], device=x.device)).expand(B_, 1, -1)
        x_emb = s.tok(x)
        inp = torch.cat([bos_emb, x_emb], dim=1)
        inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        for b in s.blocks: inp = b(inp, z_kv)
        logits = s.head(s.ln_f(inp))
        return logits[:, 1:T_ + 1]


class BlockCrossAttn(nn.Module):
    def __init__(s, N_EMBD, N_HEAD):
        super().__init__()
        s.nh = N_HEAD; s.head_dim = N_EMBD // N_HEAD
        # self-attn
        s.ln1 = nn.LayerNorm(N_EMBD); s.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        s.proj = nn.Linear(N_EMBD, N_EMBD)
        # cross-attn (NEW vs BlockCausal)
        s.ln_cross = nn.LayerNorm(N_EMBD)
        s.q_cross = nn.Linear(N_EMBD, N_EMBD)
        s.k_cross = nn.Linear(N_EMBD, N_EMBD)
        s.v_cross = nn.Linear(N_EMBD, N_EMBD)
        s.proj_cross = nn.Linear(N_EMBD, N_EMBD)
        # mlp
        s.ln2 = nn.LayerNorm(N_EMBD)
        s.mlp = nn.Sequential(nn.Linear(N_EMBD, 4 * N_EMBD), nn.GELU(), nn.Linear(4 * N_EMBD, N_EMBD))
    def forward(s, x, z_kv):
        B_, T_, C = x.shape
        # self-attn
        h = s.ln1(x); qkv = s.qkv(h).reshape(B_, T_, 3, s.nh, s.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + s.proj(y.transpose(1, 2).contiguous().view(B_, T_, C))
        # cross-attn (z_kv: [B, T+1, D])
        h2 = s.ln_cross(x)
        q_c = s.q_cross(h2).reshape(B_, T_, s.nh, s.head_dim).permute(0, 2, 1, 3)
        k_c = s.k_cross(z_kv).reshape(B_, -1, s.nh, s.head_dim).permute(0, 2, 1, 3)
        v_c = s.v_cross(z_kv).reshape(B_, -1, s.nh, s.head_dim).permute(0, 2, 1, 3)
        y_c = F.scaled_dot_product_attention(q_c, k_c, v_c)
        x = x + s.proj_cross(y_c.transpose(1, 2).contiguous().view(B_, T_, C))
        # mlp
        x = x + s.mlp(s.ln2(x))
        return x


# ============================================================
# 加载 checkpoint
# ============================================================
def load_decoder(checkpoint_name: str, device="cuda"):
    """返回 (decoder, ckpt_dir, D_Z, T, V)"""
    if checkpoint_name == "v25":
        ckpt = torch.load(V37_DIR / "v25_decoder.pt", map_location=device, weights_only=False)
        cfg = ckpt["config"]
        D_Z = cfg["D_Z"]; T = cfg["T"]
        DEC_LAYER = cfg["DEC_LAYER"]; DEC_HEAD = cfg["DEC_HEAD"]; DEC_EMBD = cfg["DEC_EMBD"]
        vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
        V = vocab["vocab_size"]; BOS_ID = vocab["stoi"].get("<bos>", 1)
        decoder = DecoderV25(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID).to(device)
        decoder.load_state_dict(ckpt["decoder"])
        return decoder, V, D_Z, T
    elif checkpoint_name == "v36":
        # v36 复用 v36_model.py 定义 (与训练一致)
        sys.path.insert(0, str(V37_DIR.parent))  # crystalllm/versions
        from v36_model import DecoderCrossAttn
        ckpt = torch.load(V37_DIR / "v36_decoder.pt", map_location=device, weights_only=False)
        cfg = ckpt["config"]
        D_Z = cfg["D_Z"]; T = cfg["T"]
        decoder = DecoderCrossAttn(
            V=cfg["V"], T=T, DEC_LAYER=cfg["DEC_LAYER"], DEC_HEAD=cfg["DEC_HEAD"],
            DEC_EMBD=cfg["DEC_EMBD"], D_Z=D_Z, BOS_ID=cfg["BOS_ID"]
        ).to(device)
        decoder.load_state_dict(ckpt["decoder"])
        return decoder, cfg["V"], D_Z, T
    else:
        raise ValueError(f"Unknown checkpoint: {checkpoint_name}")


# ============================================================
# 数据加载
# ============================================================
def load_val_data(device="cuda"):
    df_val = pd.read_parquet(DATA / "v24_val.parquet")
    val_texts = df_val["text"].tolist()
    cache = np.load(DATA / "cached_v24_z.npz")
    val_z = torch.tensor(cache["val_z"], dtype=torch.float32, device=device)
    vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
    stoi = vocab["stoi"]; itos = {int(k): v for k, v in vocab["itos"].items()}
    V = vocab["vocab_size"]
    return val_texts, val_z, stoi, itos, V


def get_val_batches(val_texts, stoi, T, B=4):
    batches = []
    for i in range(0, len(val_texts), B):
        batch = val_texts[i:i + B]
        chunks = []
        for text in batch:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi.get(c, 0) for c in chunk])
        batches.append((torch.tensor(chunks, dtype=torch.long, device="cuda"), i))
    return batches


# ============================================================
# 主评估
# ============================================================
@torch.no_grad()
def eval_ppl(decoder, val_batches, val_z, D_Z, V, z_mode):
    """z_mode: 'encoded' or 'zero'"""
    total_loss = 0.0; n = 0
    for x, i in val_batches:
        B = x.size(0)
        if z_mode == "encoded":
            z = val_z[i:i + B]
        elif z_mode == "zero":
            z = torch.zeros(B, D_Z, device=x.device)
        else:
            raise ValueError(f"Unknown z_mode: {z_mode}")
        logits = decoder(z, x)
        loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item(); n += x.numel()
    avg_loss = total_loss / n
    ppl = float(np.exp(avg_loss))
    return ppl, avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["v25", "v36"], required=True)
    parser.add_argument("--z_mode", choices=["encoded", "zero"], required=True)
    parser.add_argument("--output_json", type=str, default=None,
                        help="Optional path to save {checkpoint, z_mode, PPL, avg_loss} JSON")
    args = parser.parse_args()

    print(f"=== v37 zero-z ablation ===")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  z_mode:     {args.z_mode}")

    # 加载
    decoder, V, D_Z, T = load_decoder(args.checkpoint, device="cuda")
    decoder.eval()
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  decoder params: {n_params/1e6:.2f}M, D_Z={D_Z}, T={T}")

    val_texts, val_z, stoi, itos, V_check = load_val_data()
    assert V_check == V, f"vocab mismatch: ckpt V={V}, data V={V_check}"
    val_batches = get_val_batches(val_texts, stoi, T, B=4)
    print(f"  val_batches: {len(val_batches)} (B=4, T={T})")

    # 评估
    ppl, avg_loss = eval_ppl(decoder, val_batches, val_z, D_Z, V, args.z_mode)
    print(f"\n  [{args.checkpoint} + {args.z_mode}] PPL = {ppl:.4f} (avg_loss {avg_loss:.4f})")

    # 保存
    if args.output_json:
        result = {"checkpoint": args.checkpoint, "z_mode": args.z_mode,
                  "PPL": ppl, "avg_loss": avg_loss,
                  "decoder_params_M": n_params / 1e6, "D_Z": D_Z, "T": T,
                  "n_val_samples": len(val_texts), "n_batches": len(val_batches)}
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  saved: {args.output_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: 验证脚本可被 Python 解析 (无运行时)**

```bash
cd crystalllm/versions/v37/pipeline && python -c "import ast; ast.parse(open('zero_z_eval.py').read()); print('AST OK')"
```
预期: `AST OK`

- [ ] **Step 2.3: Commit**

```bash
git add crystalllm/versions/v37/pipeline/zero_z_eval.py
git commit -m "v37: zero_z_eval.py scaffold (v25/v36 unified eval with --z_mode)"
```

---

## Task 3: 验证 v25 + encoded 模式 (Sanity check, baseline 复现)

**Files:**
- Run: `crystalllm/versions/v37/pipeline/zero_z_eval.py`

- [ ] **Step 3.1: 复制 v25_decoder.pt 到 v37 目录**

```bash
cp crystalllm/versions/v25/v25_decoder.pt crystalllm/versions/v37/
ls -la crystalllm/versions/v37/v25_decoder.pt
```
预期: 文件存在 (约 1.8GB).

- [ ] **Step 3.2: 运行 v25 + encoded**

```bash
cd crystalllm/versions/v37 && python pipeline/zero_z_eval.py --checkpoint v25 --z_mode encoded --output_json v25_encoded.json 2>&1 | tail -20
```
预期输出 (最后几行):
```
  decoder params: 475.71M, D_Z=256, T=512
  val_batches: 254 (B=4, T=512)

  [v25 + encoded] PPL = 2.4xxx (avg_loss 0.90xx)
  saved: v25_encoded.json
```
PPL 应在 2.46-2.50 范围 (与已有 v25_e2e.json PPL=2.47 一致, ±0.02 容差来自 random chunk 采样).

- [ ] **Step 3.3: 验证 JSON 输出**

```bash
cat crystalllm/versions/v37/v25_encoded.json
```
预期字段: `checkpoint: "v25"`, `z_mode: "encoded"`, `PPL` ≈ 2.47, `n_val_samples: 1016`.

- [ ] **Step 3.4: Commit 输出**

```bash
git add crystalllm/versions/v37/v25_encoded.json
git commit -m "v37: sanity check - v25 encoded PPL reproduces baseline (2.47)"
```

---

## Task 4: 测量 v25 + zero (A2 — 主要测量)

**Files:**
- Run: `crystalllm/versions/v37/pipeline/zero_z_eval.py`

- [ ] **Step 4.1: 运行 v25 + zero**

```bash
cd crystalllm/versions/v37 && python pipeline/zero_z_eval.py --checkpoint v25 --z_mode zero --output_json v25_zero.json 2>&1 | tail -10
```
预期输出:
```
  [v25 + zero] PPL = 2.4xxx 或 2.5xxx (avg_loss ~0.90-0.95)
  saved: v25_zero.json
```
**关键观察**:
- 若 PPL ≤ 2.55 (退化 < 4%): 强烈支持"z 是 dead weight"
- 若 PPL 在 2.55-2.65 (退化 4-7%): z 有弱信号, 需二次 brainstorm
- 若 PPL > 2.65 (退化 > 7%): z 真有用, 应走 prefix-tuning

- [ ] **Step 4.2: 计算 ΔPPL_v25**

```bash
cd crystalllm/versions/v37 && python -c "
import json
enc = json.load(open('v25_encoded.json'))
zero = json.load(open('v25_zero.json'))
delta_pct = (zero['PPL'] - enc['PPL']) / enc['PPL'] * 100
print(f'v25 encoded PPL: {enc[\"PPL\"]:.4f}')
print(f'v25 zero PPL:    {zero[\"PPL\"]:.4f}')
print(f'ΔPPL_v25:        {delta_pct:+.3f}%')
print()
print('Decision:')
if abs(delta_pct) < 1: print('  → z 是 dead weight (走 C 战略重定位)')
elif delta_pct < 5:    print('  → z 有微弱信号 (二次 brainstorm)')
else:                  print('  → z 真有用 (走 B v37 prefix-tuning)')
"
```

- [ ] **Step 4.3: Commit 输出**

```bash
git add crystalllm/versions/v37/v25_zero.json
git commit -m "v37: measure v25 + zero PPL (A2 - main measurement)"
```

---

## Task 5: 验证 v36 + encoded (Sanity check)

**Files:**
- Run: `crystalllm/versions/v37/pipeline/zero_z_eval.py`

- [ ] **Step 5.1: 复制 v36_decoder.pt 到 v37 目录**

```bash
cp crystalllm/versions/v36/v36_decoder.pt crystalllm/versions/v37/
ls -la crystalllm/versions/v37/v36_decoder.pt
```
预期: 文件存在 (约 2.2GB).

- [ ] **Step 5.2: 运行 v36 + encoded**

```bash
cd crystalllm/versions/v37 && python pipeline/zero_z_eval.py --checkpoint v36 --z_mode encoded --output_json v36_encoded.json 2>&1 | tail -10
```
预期输出:
```
  [v36 + encoded] PPL = 2.8xxx (avg_loss ~1.03)
  saved: v36_encoded.json
```
PPL 应在 2.80-2.82 范围 (与 v36_e2e.json PPL=2.81 一致).

- [ ] **Step 5.3: Commit 输出**

```bash
git add crystalllm/versions/v37/v36_encoded.json
git commit -m "v37: sanity check - v36 encoded PPL reproduces baseline (2.81)"
```

---

## Task 6: 测量 v36 + zero (A4 — cross-attn 验证)

**Files:**
- Run: `crystalllm/versions/v37/pipeline/zero_z_eval.py`

- [ ] **Step 6.1: 运行 v36 + zero**

```bash
cd crystalllm/versions/v37 && python pipeline/zero_z_eval.py --checkpoint v36 --z_mode zero --output_json v36_zero.json 2>&1 | tail -10
```
预期输出:
```
  [v36 + zero] PPL = 2.xxxx (avg_loss ~?)
  saved: v36_zero.json
```
**关键对比**:
- `v36_zero - v25_zero` 的差异 = cross-attn 真实贡献
- 若差异 < 0.05 (即 v36 zero 也 ≈ v25 zero): cross-attn 在 z=0 时也接近 z=encoded 一样"无 z 可用", 但 PPL 仍比 v25 高 → 修复坍缩但 PPL 退化的根因不是 z, 而是 cross-attn 参数开销
- 若差异 > 0.20: cross-attn 部分用 z, 但 PPL 仍差 → v37 prefix-tuning 路径有意义

- [ ] **Step 6.2: 计算交叉对比**

```bash
cd crystalllm/versions/v37 && python -c "
import json
v25_z = json.load(open('v25_zero.json'))
v36_z = json.load(open('v36_zero.json'))
v25_e = json.load(open('v25_encoded.json'))
v36_e = json.load(open('v36_encoded.json'))

delta_v25 = (v25_z['PPL'] - v25_e['PPL']) / v25_e['PPL'] * 100
delta_v36 = (v36_z['PPL'] - v36_e['PPL']) / v36_e['PPL'] * 100
cross = v36_z['PPL'] - v25_z['PPL']

print(f'v25: enc={v25_e[\"PPL\"]:.4f}  zero={v25_z[\"PPL\"]:.4f}  Δ={delta_v25:+.3f}%')
print(f'v36: enc={v36_e[\"PPL\"]:.4f}  zero={v36_z[\"PPL\"]:.4f}  Δ={delta_v36:+.3f}%')
print(f'cross-attn contribution (v36_zero - v25_zero): {cross:+.4f}')
print()
if abs(cross) < 0.05:
    print('  → cross-attn 在 z=0 时无差异 → cross-attn 是装饰 (z 没用)')
elif cross > 0.20:
    print('  → cross-attn 部分用 z → v37 prefix-tuning 值得')
else:
    print('  → cross-attn 影响中等 → 二次 brainstorm')
"
```

- [ ] **Step 6.3: Commit 输出**

```bash
git add crystalllm/versions/v37/v36_zero.json
git commit -m "v37: measure v36 + zero PPL (A4 - cross-attn validation)"
```

---

## Task 7: 编写 gen_samples.py (生成质量检查)

**Files:**
- Create: `crystalllm/versions/v37/evaluation/gen_samples.py`

- [ ] **Step 7.1: 写 gen_samples.py**

写入 `crystalllm/versions/v37/evaluation/gen_samples.py`:

```python
# Copyright (c) 2026 Yiming Wang <yomin_noahwang@foxmail.com>. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""gen_samples.py — v37 zero-z 生成质量检查

对每个 (checkpoint, z_mode) 组合, 生成 10 个样本 × 50 token, 测量:
  - 非空格率 (non_space_rate)
  - 代码结构样本数 (matched_count)

用途: 辅助 PPL 决策, 看 z=0 时 decoder 是否走默认分布 (坍缩到空格).
"""
import argparse
import json
import sys
import io
import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); np.random.seed(42); random.seed(42)

EVAL_DIR = Path(__file__).resolve().parent
V37_DIR = EVAL_DIR.parent
PROJECT_ROOT = V37_DIR.parent.parent

sys.path.insert(0, str(V37_DIR / "pipeline"))
from zero_z_eval import load_decoder, load_val_data

KEYWORDS = ["import ", "def ", "class ", "function ", "var ", "const ", "let ",
            "void ", "return ", "if ", "else", "{", "}", "->", "()", "(int", "(char",
            "#if", "#endif", "#else", "public ", "private "]


@torch.no_grad()
def gen_one_sample(decoder, z, BOS_ID, V, itos, n_tokens=50):
    """生成一个 n_tokens 长度的样本, 返回 (text, generated_ids)"""
    cur = torch.tensor([[BOS_ID]], dtype=torch.long, device="cuda")
    generated = [BOS_ID]
    for _ in range(n_tokens):
        logits = decoder(z, cur)
        logits_t = logits[:, -1, :]
        probs = F.softmax(logits_t, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
        generated.append(next_id)
        cur = torch.tensor([generated], dtype=torch.long, device="cuda")
    text = "".join(itos.get(t, "<unk>") for t in generated[1:])
    return text, generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", choices=["v25", "v36"], required=True)
    parser.add_argument("--z_mode", choices=["encoded", "zero"], required=True)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--n_tokens", type=int, default=50)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    print(f"=== v37 gen_samples ===")
    print(f"  checkpoint: {args.checkpoint}, z_mode: {args.z_mode}")
    print(f"  n_samples: {args.n_samples}, n_tokens: {args.n_tokens}")

    decoder, V, D_Z, T = load_decoder(args.checkpoint, device="cuda")
    decoder.eval()
    val_texts, val_z, stoi, itos, V_check = load_val_data()
    assert V_check == V
    SPACE_ID = stoi.get(" ", -1)
    BOS_ID = stoi.get("<bos>", 1)

    non_space_rates = []
    samples = []
    matched = 0
    for i in range(args.n_samples):
        if args.z_mode == "encoded":
            z = val_z[i:i+1]
        else:
            z = torch.zeros(1, D_Z, device="cuda")
        text, gen_ids = gen_one_sample(decoder, z, BOS_ID, V, itos, n_tokens=args.n_tokens)
        samples.append(text)
        non_space_count = sum(1 for t in gen_ids[1:] if t != SPACE_ID)
        rate = non_space_count / args.n_tokens
        non_space_rates.append(rate)
        has_kw = any(kw in text for kw in KEYWORDS)
        if has_kw: matched += 1
        print(f"  sample {i}: non_space={rate:.2%} has_kw={has_kw} text={repr(text[:60])}")

    avg_rate = sum(non_space_rates) / len(non_space_rates)
    print(f"\n  [{args.checkpoint} + {args.z_mode}]")
    print(f"  avg non_space_rate: {avg_rate:.2%}")
    print(f"  matched (代码结构): {matched}/{args.n_samples}")

    result = {"checkpoint": args.checkpoint, "z_mode": args.z_mode,
              "n_samples": args.n_samples, "n_tokens": args.n_tokens,
              "non_space_rates": non_space_rates, "avg_rate": avg_rate,
              "samples": samples, "matched_count": matched}
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  saved: {args.output_json}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: 验证脚本 AST**

```bash
cd crystalllm/versions/v37/evaluation && python -c "import ast; ast.parse(open('gen_samples.py').read()); print('AST OK')"
```
预期: `AST OK`

- [ ] **Step 7.3: 运行 v25 + encoded (验证生成脚本)**

```bash
cd crystalllm/versions/v37 && python evaluation/gen_samples.py --checkpoint v25 --z_mode encoded --output_json samples_v25_encoded.json 2>&1 | tail -20
```
预期: 10 个样本生成, 非空格率应在 50-80% 范围 (v25 估).

- [ ] **Step 7.4: Commit**

```bash
git add crystalllm/versions/v37/evaluation/gen_samples.py crystalllm/versions/v37/samples_v25_encoded.json
git commit -m "v37: gen_samples.py + sanity run (v25 encoded samples)"
```

---

## Task 8: 跑剩下 3 个生成样本 (v25 zero, v36 encoded, v36 zero)

**Files:**
- Run: `crystalllm/versions/v37/evaluation/gen_samples.py`

- [ ] **Step 8.1: v25 + zero**

```bash
cd crystalllm/versions/v37 && python evaluation/gen_samples.py --checkpoint v25 --z_mode zero --output_json samples_v25_zero.json 2>&1 | tail -15
```

- [ ] **Step 8.2: v36 + encoded**

```bash
cd crystalllm/versions/v37 && python evaluation/gen_samples.py --checkpoint v36 --z_mode encoded --output_json samples_v36_encoded.json 2>&1 | tail -15
```

- [ ] **Step 8.3: v36 + zero**

```bash
cd crystalllm/versions/v37 && python evaluation/gen_samples.py --checkpoint v36 --z_mode zero --output_json samples_v36_zero.json 2>&1 | tail -15
```

- [ ] **Step 8.4: 对比生成质量**

```bash
cd crystalllm/versions/v37 && python -c "
import json
for ckpt in ['v25', 'v36']:
    for mode in ['encoded', 'zero']:
        d = json.load(open(f'samples_{ckpt}_{mode}.json'))
        print(f'{ckpt} {mode}: avg_non_space={d[\"avg_rate\"]:.2%}, matched={d[\"matched_count\"]}/10')
"
```
预期:
- v25 encoded: ~70% non_space (估), matched ~5/10
- v25 zero: 退化或维持, 关键看是否塌缩到 <50%
- v36 encoded: 85% non_space (已知), matched 6/10 (已知)
- v36 zero: 关键看 cross-attn 是否能让 decoder 仍维持质量

- [ ] **Step 8.5: Commit**

```bash
git add crystalllm/versions/v37/samples_v25_zero.json crystalllm/versions/v37/samples_v36_encoded.json crystalllm/versions/v37/samples_v36_zero.json
git commit -m "v37: complete 4 sample sets + quality comparison"
```

---

## Task 9: 聚合 v37_e2e.json + 决策报告

**Files:**
- Create: `crystalllm/versions/v37/v37_e2e.json`
- Create: `crystalllm/versions/v37/v37_decision.md`

- [ ] **Step 9.1: 写聚合脚本 (内联)**

```bash
cd crystalllm/versions/v37 && python -c "
import json

# 加载 4 个 PPL 结果
v25_e = json.load(open('v25_encoded.json'))
v25_z = json.load(open('v25_zero.json'))
v36_e = json.load(open('v36_encoded.json'))
v36_z = json.load(open('v36_zero.json'))

# 加载 4 个样本
s_v25_e = json.load(open('samples_v25_encoded.json'))
s_v25_z = json.load(open('samples_v25_zero.json'))
s_v36_e = json.load(open('samples_v36_encoded.json'))
s_v36_z = json.load(open('samples_v36_zero.json'))

# 派生指标
delta_v25 = (v25_z['PPL'] - v25_e['PPL']) / v25_e['PPL'] * 100
delta_v36 = (v36_z['PPL'] - v36_e['PPL']) / v36_e['PPL'] * 100
cross_attn_contribution = v36_z['PPL'] - v25_z['PPL']

# 决策
if abs(delta_v25) < 1 and abs(cross_attn_contribution) < 0.05:
    decision = 'C_strategic_relocation'
    decision_desc = 'z is dead weight → 战略重定位'
elif abs(delta_v25) < 1:
    decision = 'C_with_cross_attn_review'
    decision_desc = 'z 死路 + cross-attn 装饰 → 战略重定位 (但 cross-attn 也需重新审视)'
elif delta_v25 < 5:
    decision = 'secondary_brainstorm'
    decision_desc = 'z 微弱信号 → 二次 brainstorm'
else:
    decision = 'B_prefix_tuning'
    decision_desc = 'z 真有用 → v37 prefix-tuning'

e2e = {
    'A1_v25_encoded': {'PPL': v25_e['PPL'], 'avg_loss': v25_e['avg_loss']},
    'A2_v25_zero':    {'PPL': v25_z['PPL'], 'avg_loss': v25_z['avg_loss']},
    'A3_v36_encoded': {'PPL': v36_e['PPL'], 'avg_loss': v36_e['avg_loss']},
    'A4_v36_zero':    {'PPL': v36_z['PPL'], 'avg_loss': v36_z['avg_loss']},
    'delta_ppl_v25_pct': delta_v25,
    'delta_ppl_v36_pct': delta_v36,
    'cross_attn_contribution_ppl': cross_attn_contribution,
    'samples': {
        'v25_encoded': {'avg_non_space': s_v25_e['avg_rate'], 'matched': s_v25_e['matched_count']},
        'v25_zero':    {'avg_non_space': s_v25_z['avg_rate'], 'matched': s_v25_z['matched_count']},
        'v36_encoded': {'avg_non_space': s_v36_e['avg_rate'], 'matched': s_v36_e['matched_count']},
        'v36_zero':    {'avg_non_space': s_v36_z['avg_rate'], 'matched': s_v36_z['matched_count']},
    },
    'decision': decision,
    'decision_desc': decision_desc,
}

with open('v37_e2e.json', 'w', encoding='utf-8') as f:
    json.dump(e2e, f, indent=2, ensure_ascii=False)
print(json.dumps(e2e, indent=2, ensure_ascii=False))
print(f'\ndecision: {decision}')
"
```

- [ ] **Step 9.2: 验证 v37_e2e.json 已生成**

```bash
cat crystalllm/versions/v37/v37_e2e.json
```
预期字段: 4 PPL + 派生指标 + decision.

- [ ] **Step 9.3: 写 v37_decision.md**

写入 `crystalllm/versions/v37/v37_decision.md` (基于实测数据填充):

```markdown
# v37 Zero-z Ablation — 决策报告

> **承接 v36**: cross-attn 实验未达 PPL 目标. v37 通过 zero-z ablation 量化 decoder 对 z 的真实依赖度.
> **执行**: `python pipeline/zero_z_eval.py` × 4 (A1-A4) + `evaluation/gen_samples.py` × 4.

## 1. 实验结果

### 1.1 PPL 矩阵

| 编号 | 模型 | z_mode | PPL | Δ vs encoded |
|---|---|---:|---:|---:|
| A1 | v25 | encoded | **<FILL>** | (baseline) |
| A2 | v25 | **zero** | **<FILL>** | **<FILL>%** |
| A3 | v36 | encoded | **<FILL>** | (baseline) |
| A4 | v36 | **zero** | **<FILL>** | **<FILL>%** |

### 1.2 交叉对比

- **v25 ΔPPL (zero vs encoded)**: <FILL>%
- **v36 ΔPPL (zero vs encoded)**: <FILL>%
- **cross-attn 贡献 (v36_zero - v25_zero)**: <FILL> PPL

### 1.3 生成质量

| 模型 + z_mode | 非空格率 | 代码结构样本数 |
|---|---:|---:|
| v25 + encoded | <FILL>% | <FILL>/10 |
| v25 + zero | <FILL>% | <FILL>/10 |
| v36 + encoded | <FILL>% | <FILL>/10 |
| v36 + zero | <FILL>% | <FILL>/10 |

## 2. 决策

**场景**: <FILL: A / B / C, 见 spec §5.1>

**核心结论**: <FILL: 一句话总结, e.g. "z 是 dead weight" / "cross-attn 部分用 z" / "z 真有用">

## 3. 推荐下一步

### 若场景 A (z 死路, ΔPPL_v25 < 1%, cross-attn 贡献 < 0.05)

**走 C 战略重定位**:
- 接受 decoder 不消费 z 的事实
- 重新定义"信息结晶"含义: z 不是生成路线的输入, 而是 SpS 路由 / 数据压缩探针 / 可控性接口
- 或: 放弃混合, v25 + SpS 走速度优化路径 (复用 v31 思路)
- 不再做"让 decoder 用 z"的尝试

### 若场景 B (z 微弱信号, 1-5%)

**二次 brainstorm**:
- v22a 已验证 z 编码完美 (主题 acc 75-94%), 但 decoder 不消费
- 可能中间状态: z 有信息但 decoder 容量饱和 (v22a PPL 范围 0.4% = decoder 忽略 z)
- 需补做更细粒度 ablation: 部分维度 z=0, 维度子集测试

### 若场景 C (z 真有用, >5%)

**走 B v37 prefix-tuning**:
- 设计: z 拆成 M=8 memory tokens, 每层 prefix-tuning
- 比 cross-attn 更轻量, z 信息可选择性使用
- 但 KL=303 仍待修 (v38 路径)

## 4. 与 OKR 的关系

若决策为 C 战略重定位, 需:
- 更新 goal.md: KR1.2 "z 为全局条件" 措辞改为 "z 为可选上下文/可控性接口"
- KR3.1 (主题控制) 重新定义成功标准 (不再要求生成端体现主题, 改为 z 空间可分性)
- M3 1.5B 联合训练目标需重新审视

若决策为 B (prefix-tuning), 写 v37b prefix-tuning spec.
若决策为二次 brainstorm, 列出待补做的 ablation.

## 5. 文件清单

- `crystalllm/versions/v37/pipeline/zero_z_eval.py` — 统一 eval 脚本
- `crystalllm/versions/v37/evaluation/gen_samples.py` — 生成质量脚本
- `crystalllm/versions/v37/v37_e2e.json` — 聚合结果
- `crystalllm/versions/v37/v37_decision.md` — 本报告
- `samples_{v25,v36}_{encoded,zero}.json` — 4 套样本

## 6. 下一步 spec

根据本报告决策, 写下一个 spec (v37b / v38 / 战略重定位 / 二次 brainstorm).
```

**说明**: 用 `<FILL>` 占位符标记需根据实测结果填充的字段. 填好后删除所有 `<FILL>` 标记.

- [ ] **Step 9.4: 用实测数据填充 v37_decision.md**

```bash
cd crystalllm/versions/v37 && python -c "
import json
e2e = json.load(open('v37_e2e.json'))
content = open('v37_decision.md').read()

# 填充 PPL 矩阵
content = content.replace('<FILL>', f\"{e2e['A1_v25_encoded']['PPL']:.4f}\", 1)  # A1
content = content.replace('<FILL>', f\"{e2e['A2_v25_zero']['PPL']:.4f}\", 1)     # A2 PPL
content = content.replace('<FILL>', f\"{e2e['delta_ppl_v25_pct']:+.3f}\", 1)    # A2 delta
content = content.replace('<FILL>', f\"{e2e['A3_v36_encoded']['PPL']:.4f}\", 1)  # A3
content = content.replace('<FILL>', f\"{e2e['A4_v36_zero']['PPL']:.4f}\", 1)     # A4 PPL
content = content.replace('<FILL>', f\"{e2e['delta_ppl_v36_pct']:+.3f}\", 1)    # A4 delta

# 交叉对比
content = content.replace('<FILL>', f\"{e2e['delta_ppl_v25_pct']:+.3f}\", 1)
content = content.replace('<FILL>', f\"{e2e['delta_ppl_v36_pct']:+.3f}\", 1)
content = content.replace('<FILL>', f\"{e2e['cross_attn_contribution_ppl']:+.4f}\", 1)

# 生成质量
for ckpt in ['v25', 'v36']:
    for mode in ['encoded', 'zero']:
        s = e2e['samples'][f'{ckpt}_{mode}']
        content = content.replace('<FILL>', f\"{s['avg_non_space']*100:.1f}\", 1)
        content = content.replace('<FILL>', f\"{s['matched']}\", 1)

# 决策场景
delta = abs(e2e['delta_ppl_v25_pct'])
cross = abs(e2e['cross_attn_contribution_ppl'])
if delta < 1 and cross < 0.05:
    scene = 'A (z 死路, ΔPPL_v25 < 1%, cross-attn 贡献 < 0.05)'
    conclusion = 'z 是 dead weight + cross-attn 是装饰'
elif delta < 1:
    scene = 'A (z 死路, 但 cross-attn 有独立贡献)'
    conclusion = 'z 死路 + cross-attn 自身引入噪声'
elif delta < 5:
    scene = 'B (z 微弱信号, 1-5%)'
    conclusion = 'z 有信号但弱, 需二次 brainstorm'
else:
    scene = 'C (z 真有用, >5%)'
    conclusion = 'z 真有用, 走 v37 prefix-tuning'

content = content.replace('<FILL>', scene, 1)
content = content.replace('<FILL>', conclusion, 1)

# 检查是否还有 <FILL> 残留
import re
remaining = re.findall(r'<FILL>', content)
if remaining:
    print(f'WARNING: {len(remaining)} <FILL> placeholders remain, manual fill needed')
else:
    print('All <FILL> placeholders replaced.')

with open('v37_decision.md', 'w', encoding='utf-8') as f:
    f.write(content)
print('v37_decision.md updated.')
"
```

- [ ] **Step 9.5: 验证 v37_decision.md 无 <FILL> 残留**

```bash
grep -c '<FILL>' crystalllm/versions/v37/v37_decision.md || echo "0 placeholders remaining"
```
预期: `0 placeholders remaining` (grep 返回 0 或显示 "0 placeholders remaining").

- [ ] **Step 9.6: Commit**

```bash
git add crystalllm/versions/v37/v37_e2e.json crystalllm/versions/v37/v37_decision.md
git commit -m "v37: decision report - zero-z ablation results + next step"
```

---

## Task 10: 更新 README + TIMELINE 指针

**Files:**
- Modify: `README.md` (root) - 添加 v37 链接

- [ ] **Step 10.1: 在 README.md 的状态行添加 v37 链接**

打开 `README.md`, 找到 "Status (v36, 2026-06-19)" 这一行, 替换为:

```markdown
- **Status (v37, 2026-06-19).** Zero-z ablation complete — see [`crystalllm/versions/v37/v37_decision.md`](./crystalllm/versions/v37/v37_decision.md) for the decision and recommended next step. v25 remains the current PPL SOTA (2.47).
```

- [ ] **Step 10.2: Commit**

```bash
git add README.md
git commit -m "v37: README status pointer to decision report"
```

---

## Self-Review

### 1. Spec Coverage
- ✅ §2 目标: Task 3-6 测 4 PPL
- ✅ §2 量化指标: Task 9 派生 ΔPPL_v25, ΔPPL_v36_vs_v25, 生成质量
- ✅ §3 架构 (zero-z 实现): Task 2-6 在 `zero_z_eval.py` 实现 `torch.zeros(B, D_Z)` 替换
- ✅ §4 实验矩阵 (A1-A4): Task 3-6 各对应一个
- ✅ §5 决策矩阵 (双指标): Task 9.1 决策逻辑实现
- ✅ §6 风险 (zero-z 实现 bug): Task 3.2 + Task 5.2 sanity check baseline 复现兜底
- ✅ §7 文件交付: Task 1-9 创建全部 6 个文件

### 2. Placeholder Scan
- ✅ 无 TBD/TODO (Step 9.3 的 `<FILL>` 在 Step 9.4-9.5 自动替换, Step 9.5 验证)
- ✅ 无 "implement later" / "similar to Task N"
- ✅ 所有代码 step 都给出完整代码块

### 3. Type Consistency
- ✅ `DecoderV25(z, x)` / `DecoderV36(z, x)` 签名一致 (Task 2.1 + Task 7.1)
- ✅ `eval_ppl(decoder, val_batches, val_z, D_Z, V, z_mode)` 在 Task 2.1 定义, Task 3-6 调用一致
- ✅ `load_decoder(checkpoint_name, device)` 返回 (decoder, V, D_Z, T), Task 3-7 调用一致
- ✅ `gen_one_sample(decoder, z, BOS_ID, V, itos, n_tokens)` 在 Task 7.1 定义
- ✅ 文件命名: `v25_encoded.json` / `v25_zero.json` / `v36_encoded.json` / `v36_zero.json` (Task 3-6 + Task 9.1 一致)

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-19-v37-zero-z-ablation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - 派 subagent 逐 task 执行, 我在每 task 间 review

**2. Inline Execution** - 在本会话按顺序执行, 带 checkpoint

**Which approach?**