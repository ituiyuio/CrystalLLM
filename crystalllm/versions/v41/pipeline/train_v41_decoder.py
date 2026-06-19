"""train_v41_decoder.py — v41 Block-Diffusion Loss PoC (warm-start from v25)

核心改动 vs v25:
  - 训练目标: L_total = α*L_AR + (1-α)*L_block_diffusion
  - 架构: 完全复用 v25 DecoderV25 (z at pos 0, causal attention)
  - vocab: 扩展 V=2261 → V=2262 (新增 <mask> token)
  - 训练步数: 1500 (PoC 短周期)
  - LR: 3e-5 (v25 的 30%, 避免破坏 warm-start)
  - KL 项: 关闭 (PoC 测纯 loss 结构)

预期:
  - PPL: < v25 (2.47) 才算成功
  - 时间: ~30 min 训练 + 5 min 评估 (RTX 5090)
"""
import json, time, random, sys, io, os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
torch.manual_seed(42); random.seed(42); np.random.seed(42)


def P(*a, **kw):
    print(*a, **kw); sys.stdout.flush()


# ============================================================
# 路径
# ============================================================
V41_DIR = Path(__file__).resolve().parents[1]   # crystalllm/versions/v41
CRYSTALLLM_DIR = V41_DIR.parents[1]                  # crystalllm/
DATA = CRYSTALLLM_DIR / "data" / "processed"          # crystalllm/data/processed
V25_CKPT = CRYSTALLLM_DIR / "versions" / "v25" / "v25_decoder.pt"

P("=== v41 Block-Diffusion Loss PoC (warm-start from v25) ===")

# ============================================================
# Vocab (扩展 V+1 加 <mask>)
# ============================================================
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V_BASE = vocab["vocab_size"]  # 2261
BOS_ID = stoi.get("<bos>", 1)
PAD_ID = stoi.get("<pad>", 0)
EOS_ID = stoi.get("<eos>", 2)
MASK_ID = V_BASE  # 新增 <mask> 在末尾, id = 2261
V = V_BASE + 1
P(f"Vocab: V_BASE={V_BASE} → V={V} (added <mask>={MASK_ID})")

# ============================================================
# 数据
# ============================================================
df_train = pd.read_parquet(DATA / "v24_train.parquet")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
P(f"v24 train: {len(train_texts)} | val: {len(val_texts)}")

# ============================================================
# 配置
# ============================================================
B, T = 4, 512
D_Z = 256
DEC_LAYER, DEC_HEAD, DEC_EMBD = 24, 20, 1280  # 与 v25 完全一致
LR = 3e-5
WARMUP_STEPS = 100
STEPS = 1500
EVAL_EVERY = 250
ALPHA = 0.5  # AR vs block-diffusion 权重
BLOCK_SIZE = 16
MASK_RATE_MIN, MASK_RATE_MAX = 0.1, 0.5
DEVICE = "cuda"
N_VAL_BATCHES = 254  # 与 v40 对齐


# ============================================================
# 模型 (扩展 v25 DecoderV25: 加 mask_input 支持)
# ============================================================
class BlockCausal(nn.Module):
    """与 v25 一致: 24 层 causal Transformer block"""
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


class DecoderV25Extended(nn.Module):
    """v25 BAD-DP + 可选 mask_input 支持.

    mask_input: (B, T) bool, True 表示该位置在输入中替换为 <mask>.
                若 None, 走 v25 标准 forward.
    """
    def __init__(s, V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID, MASK_ID):
        super().__init__()
        s.T, s.BOS_ID, s.MASK_ID = T, BOS_ID, MASK_ID
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        s.pos = nn.Embedding(T + 2, DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight  # tied
    def forward(s, z, x, mask_input=None):
        B_, T_ = x.shape
        # 可选: 用 <mask> 替换被 mask 的 token
        if mask_input is not None:
            x = x.clone()
            x[mask_input] = s.MASK_ID
        z_emb = s.z_to_emb(z).unsqueeze(1)
        bos_emb = s.tok(torch.tensor([s.BOS_ID], device=x.device)).expand(B_, 1, -1)
        x_emb = s.tok(x)
        inp = torch.cat([z_emb, bos_emb, x_emb], dim=1)
        inp = inp + s.pos(torch.arange(T_ + 2, device=x.device))
        for b in s.blocks: inp = b(inp)
        logits = s.head(s.ln_f(inp))
        return logits[:, 1:T_ + 1]  # (B, T, V)


# ============================================================
# Loss components
# ============================================================
def make_block_mask(batch_size, seq_len, block_size=BLOCK_SIZE,
                    mask_rate_range=(MASK_RATE_MIN, MASK_RATE_MAX), device=DEVICE):
    """生成 per-block mask: 每个块独立采样 mask rate, 块内 token 独立 Bernoulli.

    返回: (B, T) bool, True = masked
    """
    n_blocks = seq_len // block_size
    assert seq_len % block_size == 0, f"T={seq_len} must be divisible by block_size={block_size}"

    # 每个块一个 mask rate (B, n_blocks)
    rates = torch.empty(batch_size, n_blocks, device=device).uniform_(*mask_rate_range)
    # 每个 token 独立 Bernoulli
    rand = torch.rand(batch_size, n_blocks, block_size, device=device)
    block_mask = rand < rates.unsqueeze(-1)  # (B, n_blocks, block_size)
    return block_mask.view(batch_size, seq_len)


def ar_loss(decoder, z, x, V):
    """L_AR: 标准 next-token CE, 与 v25 完全一致 (no mask)"""
    logits = decoder(z, x, mask_input=None)
    return F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))


def block_diffusion_loss(decoder, z, x, mask_rate_range, V):
    """L_block_diffusion: MDLM 风格, 在 masked positions 上计算 CE.

    实现:
      1. 生成 mask (B, T) bool
      2. decoder forward with mask_input=mask
      3. CE loss 仅在 masked positions 计算
      4. mean over masked positions (与 MDLM 一致)
    """
    mask = make_block_mask(x.size(0), x.size(1), mask_rate_range=mask_rate_range, device=x.device)
    logits = decoder(z, x, mask_input=mask)  # (B, T, V)
    # CE 仅在 masked positions
    loss_per_tok = F.cross_entropy(
        logits.reshape(-1, V), x.reshape(-1), reduction='none'
    ).reshape(x.shape)  # (B, T)
    n_masked = mask.float().sum().clamp(min=1)
    return (loss_per_tok * mask.float()).sum() / n_masked


# ============================================================
# Warm-start from v25
# ============================================================
P(f"\n=== Warm-start: 加载 v25 decoder ({V_BASE} → {V}) ===")
ckpt_v25 = torch.load(V25_CKPT, map_location=DEVICE, weights_only=False)
v25_state = ckpt_v25["decoder"]
P(f"v25 keys: {len(v25_state)}, sample: {list(v25_state.keys())[:3]}")

# 构建 v41 decoder (V+1)
decoder = DecoderV25Extended(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD,
                              BOS_ID=BOS_ID, MASK_ID=MASK_ID).to(DEVICE)
P(f"v41 decoder initialized (V={V}, params={sum(p.numel() for p in decoder.parameters())/1e6:.2f}M)")

# 加载 v25 权重, 处理 V 扩展
v41_state = decoder.state_dict()
loaded = 0; skipped = 0; extended = 0
for k, v in v25_state.items():
    if k not in v41_state:
        skipped += 1; continue
    if v.shape == v41_state[k].shape:
        v41_state[k] = v.clone()
        loaded += 1
    elif "tok.weight" in k or "head.weight" in k:
        # V → V+1 扩展: 复制 V 行, 新行用 mean 初始化
        assert v.shape[0] == V_BASE, f"unexpected shape {v.shape}"
        new_w = torch.cat([v, v.mean(dim=0, keepdim=True)], dim=0)
        v41_state[k] = new_w
        extended += 1
    else:
        skipped += 1
        P(f"  skipped {k}: {v.shape} vs {v41_state[k].shape}")

decoder.load_state_dict(v41_state)
P(f"Loaded: {loaded}, Extended: {extended}, Skipped: {skipped}")

# 加载 cached z
cache = np.load(DATA / "cached_v24_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)
P(f"Loaded v24 cached z: train {train_z_cache.shape} | val {val_z_cache.shape}")


# ============================================================
# Optimizer + Scheduler
# ============================================================
opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: (
    step / WARMUP_STEPS if step < WARMUP_STEPS
    else 0.5 * (1 + np.cos(np.pi * min(step - WARMUP_STEPS, STEPS - WARMUP_STEPS) / max(STEPS - WARMUP_STEPS, 1)))
))


# ============================================================
# 训练循环
# ============================================================
def get_batch(texts, B, T):
    """随机采样 B 个样本, 每个取 T 长度的 chunk"""
    ix = np.random.randint(0, len(texts), B)
    x_chunks = []
    for i in ix:
        text = texts[i]
        if len(text) < T: text = text + "\n" * (T - len(text))
        start = random.randint(0, max(0, len(text) - T))
        chunk = text[start:start + T]
        x_chunks.append([stoi.get(c, 0) for c in chunk])
    return torch.tensor(x_chunks, dtype=torch.long, device=DEVICE)


@torch.no_grad()
def eval_ppl(decoder, val_texts, val_z, B, T, n_batches=N_VAL_BATCHES):
    """评估 PPL (与 v37/v40 对齐)"""
    decoder.eval()
    total_loss = 0.0; n_tok = 0
    for bi in range(n_batches):
        i_start = bi * B
        if i_start + B > len(val_texts):
            break
        batch_texts = val_texts[i_start:i_start + B]
        chunks = []
        for text in batch_texts:
            if len(text) < T: text = text + "\n" * (T - len(text))
            start = random.randint(0, max(0, len(text) - T))
            chunk = text[start:start + T]
            chunks.append([stoi.get(c, 0) for c in chunk])
        x = torch.tensor(chunks, dtype=torch.long, device=DEVICE)
        z = val_z[i_start:i_start + B]
        logits = decoder(z, x, mask_input=None)  # (B, T, V=2262)
        # 用 V_BASE=2261 避免预测 <mask> token (loss 只对真实 token 计)
        loss = F.cross_entropy(
            logits[..., :V_BASE].reshape(-1, V_BASE),
            x.reshape(-1), reduction='sum'
        )
        total_loss += loss.item(); n_tok += x.numel()
    return float(np.exp(total_loss / n_tok))


P(f"\n=== 准备训练 (run train_main() to start) ===")
P(f"(L_total = {ALPHA}*L_AR + {1-ALPHA}*L_block_diffusion, block_size={BLOCK_SIZE}, "
  f"mask_rate~U({MASK_RATE_MIN},{MASK_RATE_MAX}))")


def train_main():
    """实际训练入口 (单独调用避免 import 时触发)"""
    global decoder, opt, sched
    t0 = time.time()
    log = []
    best_ppl = float('inf')

    for step in range(STEPS):
        decoder.train()
        x = get_batch(train_texts, B, T)
        ix = np.random.randint(0, len(train_texts), B)
        z = train_z_cache[torch.tensor(ix, device=DEVICE)]

        # 双 loss
        l_ar = ar_loss(decoder, z, x, V=V)
        l_diff = block_diffusion_loss(decoder, z, x,
                                        mask_rate_range=(MASK_RATE_MIN, MASK_RATE_MAX), V=V)
        loss = ALPHA * l_ar + (1 - ALPHA) * l_diff

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step(); sched.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            ppl = eval_ppl(decoder, val_texts, val_z_cache, B, T, n_batches=N_VAL_BATCHES)
            elapsed = time.time() - t0
            eta = elapsed / max(step, 1) * (STEPS - step)
            lr_now = opt.param_groups[0]['lr']
            P(f"  step {step:4d}/{STEPS} | L_AR {l_ar.item():.3f} | L_diff {l_diff.item():.3f} "
              f"| L_total {loss.item():.3f} | val_ppl {ppl:.4f} | lr {lr_now:.2e} "
              f"| {elapsed:.0f}s ETA {eta:.0f}s")
            log.append({
                "step": step, "l_ar": l_ar.item(), "l_diff": l_diff.item(),
                "l_total": loss.item(), "val_ppl": ppl, "lr": lr_now
            })
            if ppl < best_ppl:
                best_ppl = ppl


    # ============================================================
    # 保存
    # ============================================================
    SAVE = V41_DIR / "v41_decoder.pt"
    torch.save({"decoder": decoder.state_dict(),
                "config": {"V": V, "V_BASE": V_BASE, "MASK_ID": MASK_ID,
                           "T": T, "D_Z": D_Z,
                           "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD,
                           "BLOCK_SIZE": BLOCK_SIZE, "ALPHA": ALPHA,
                           "MASK_RATE_MIN": MASK_RATE_MIN, "MASK_RATE_MAX": MASK_RATE_MAX,
                           "warm_start_from": "v25_decoder",
                           "best_val_ppl": best_ppl,
                           "arch": "v41-block-diffusion-loss-poc-warm-start"}}, SAVE)
    P(f"\nModel saved: {SAVE}")

    LOG = V41_DIR / "v41_train_log.json"
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump({"log": log,
                   "config": {"STEPS": STEPS, "T": T, "D_Z": D_Z, "B": B,
                              "BLOCK_SIZE": BLOCK_SIZE, "ALPHA": ALPHA,
                              "LR": LR, "WARMUP_STEPS": WARMUP_STEPS,
                              "MASK_RATE_MIN": MASK_RATE_MIN, "MASK_RATE_MAX": MASK_RATE_MAX,
                              "warm_start_from": "v25_500M_decoder_T512",
                              "n_train": len(train_texts), "n_val": len(val_texts),
                              "decoder_params_M": sum(p.numel() for p in decoder.parameters())/1e6,
                              "best_val_ppl": best_ppl,
                              "n_val_batches_eval": N_VAL_BATCHES,
                              "arch": "v41-block-diffusion-loss-poc"}}, f, indent=2)
    P(f"Log saved: {LOG}")

    P(f"\n=== 训练完成 ({time.time()-t0:.0f}s, best_val_ppl={best_ppl:.4f}) ===")
    P(f"\n下一步: python eval_v41.py → PPL 对比 v25 (2.47)")
    return best_ppl


if __name__ == "__main__":
    train_main()