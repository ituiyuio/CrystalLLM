"""train_v42_decoder.py — v42 Per-Block z Injection PoC (warm-start from v25)

承接 v41: block-diffusion loss 失败 (PPL +3.58%). v42 改为只测**块结构本身**
- 在每块首部注入 z_emb (与 v25 pos 0 的 z_emb 相同)
- Loss: 纯 L_AR (不引入 mask-diffusion loss)
- 评估: PPL < v25 (2.47) 才算成功

预期:
- 成功条件: PPL < 2.47 (-0.8% vs v25)
- 时间: ~3 min 训练 + 1 min 评估 (RTX 5090, 100 steps)
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
# 常量与路径
# ============================================================
BLOCK_SIZE = 16
T_TOKEN = 512  # token-level sequence length
N_BLOCKS = T_TOKEN // BLOCK_SIZE  # 32
# Total positions: block 0 = BLOCK_SIZE + 2 (z + BOS + x), others = BLOCK_SIZE + 1 (z + x)
V25_POS_LEN = 514  # T + 2 (v25)
V42_POS_LEN = (BLOCK_SIZE + 2) + (N_BLOCKS - 1) * (BLOCK_SIZE + 1)  # 18 + 31*17 = 545

V42_DIR = Path(__file__).resolve().parents[1]
CRYSTALLLM_DIR = V42_DIR.parents[1]  # crystalllm/
DATA = CRYSTALLLM_DIR / "data" / "processed"
V25_CKPT = CRYSTALLLM_DIR / "versions" / "v25" / "v25_decoder.pt"

P("=== v42 Per-Block z Injection PoC (warm-start from v25) ===")
P(f"BLOCK_SIZE={BLOCK_SIZE}, N_BLOCKS={N_BLOCKS}, V25_POS_LEN={V25_POS_LEN}, V42_POS_LEN={V42_POS_LEN}")


# ============================================================
# Vocab
# ============================================================
vocab = json.load(open(DATA / "char_vocab.json", encoding="utf-8"))
stoi = vocab["stoi"]
itos = {int(k): v for k, v in vocab["itos"].items()}
V = vocab["vocab_size"]  # 2261 (no new tokens for v42)
BOS_ID = stoi["<bos>"]
P(f"Vocab V={V} (no new tokens)")


# ============================================================
# 数据
# ============================================================
df_train = pd.read_parquet(DATA / "v24_train.parquet")
df_val = pd.read_parquet(DATA / "v24_val.parquet")
train_texts = df_train["text"].tolist()
val_texts = df_val["text"].tolist()
P(f"train: {len(train_texts)} | val: {len(val_texts)}")


# ============================================================
# 配置
# ============================================================
B, T = 4, 512
D_Z = 256
DEC_LAYER, DEC_HEAD, DEC_EMBD = 24, 20, 1280
LR = 1e-6
WARMUP_STEPS = 30
STEPS = 100
EVAL_EVERY = 25
N_VAL_BATCHES_TRAIN = 16
N_VAL_BATCHES_FINAL = 254
DEVICE = "cuda"


# ============================================================
# Block-aware position computation
# ============================================================
def compute_block_positions(batch_size, t_token, block_size=BLOCK_SIZE, device=DEVICE):
    """生成 per-block 位置 ID.

    Block 0: 0 (z), 1 (BOS), 2..17 (x_0..x_15)
    Block k>0: pos_start (z), pos_start+1..pos_start+16 (x_{Bk}..x_{Bk+15})

    Returns: (B, V42_POS_LEN) long
    """
    n_blocks = t_token // block_size
    positions = []
    for b in range(n_blocks):
        if b == 0:
            # z, BOS, x_0..x_{B-1}
            block_pos = [0] + [1] + list(range(2, 2 + block_size))
        else:
            # z at start, then x_{Bb}..x_{Bb+B-1}
            # first block (block 1) z is at position 18
            base = 2 + b * (block_size + 1)  # 2 + 1*17 = 19 for block 1, but z should be at 18
            # Correct formula: block 0 has 18 positions (z+BOS+x), block k has 17 positions (z+x)
            # z position for block b: 18 + (b-1)*17 for b>=1
            # x positions for block b: z_pos+1 .. z_pos+16
            if b == 0:
                z_pos = 0
            else:
                z_pos = 18 + (b - 1) * (block_size + 1)
            block_pos = [z_pos] + list(range(z_pos + 1, z_pos + 1 + block_size))
        positions.extend(block_pos)
    pos_tensor = torch.tensor(positions, dtype=torch.long, device=device)
    return pos_tensor.unsqueeze(0).expand(batch_size, -1).contiguous()


# ============================================================
# Per-block input builder
# ============================================================
def build_per_block_input(z, x, z_to_emb, tok, bos_emb, block_size=BLOCK_SIZE, device=DEVICE):
    """构建 per-block 输入 (B, V42_POS_LEN, DEC_EMBD).

    Layout:
      Block 0:  [z_emb, BOS, x_emb_0..x_emb_{B-1}]      → B+2 positions
      Block k>0: [z_emb, x_emb_{Bk}..x_emb_{Bk+B-1}]    → B+1 positions
    """
    B_, T_ = x.shape
    n_blocks = T_ // block_size

    # z_emb: (B, 1, D) → expand to (B, n_blocks, D)
    z_emb = z_to_emb(z).unsqueeze(1)  # (B, 1, D)

    # x_emb: (B, T, D)
    x_emb = tok(x)

    blocks = []
    for b in range(n_blocks):
        z_block = z_emb  # same z for every block
        if b == 0:
            bos = bos_emb.expand(B_, 1, -1)
            x_block = x_emb[:, b*block_size:(b+1)*block_size, :]
            block = torch.cat([z_block, bos, x_block], dim=1)  # (B, B+2, D)
        else:
            x_block = x_emb[:, b*block_size:(b+1)*block_size, :]
            block = torch.cat([z_block, x_block], dim=1)  # (B, B+1, D)
        blocks.append(block)

    inp = torch.cat(blocks, dim=1)  # (B, V42_POS_LEN, D)
    return inp


# ============================================================
# 模型
# ============================================================
class BlockCausal(nn.Module):
    """与 v25/v41 一致"""
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


class DecoderV42(nn.Module):
    """Decoder with per-block z injection.

    输入: z (B, D_Z), x (B, T)
    输出: logits (B, T, V) — predicting x[0..T-1]

    内部构造 (B, V42_POS_LEN, D) 序列:
      Block 0:  [z_emb, BOS, x_emb_0..x_emb_{B-1}]
      Block k>0: [z_emb, x_emb_{Bk}..x_emb_{Bk+B-1}]
    """
    def __init__(s, V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD, BOS_ID, MASK_ID):
        super().__init__()
        s.T, s.BOS_ID, s.MASK_ID = T, BOS_ID, MASK_ID
        s.block_size = BLOCK_SIZE
        s.z_to_emb = nn.Linear(D_Z, DEC_EMBD)
        s.tok = nn.Embedding(V, DEC_EMBD)
        # pos embedding: 扩展到 V42_POS_LEN
        s.pos = nn.Embedding(V42_POS_LEN, DEC_EMBD)
        s.blocks = nn.ModuleList([BlockCausal(DEC_EMBD, DEC_HEAD) for _ in range(DEC_LAYER)])
        s.ln_f = nn.LayerNorm(DEC_EMBD)
        s.head = nn.Linear(DEC_EMBD, V, bias=False)
        s.tok.weight = s.head.weight

    def forward(s, z, x):
        B_, T_ = x.shape
        # 1. z_to_emb
        z_emb = s.z_to_emb(z).unsqueeze(1)  # (B, 1, D)

        # 2. BOS emb
        bos_emb = s.tok(torch.tensor([s.BOS_ID], device=x.device)).expand(B_, 1, -1)

        # 3. x_emb
        x_emb = s.tok(x)  # (B, T, D)

        # 4. Build per-block input
        n_blocks = T_ // s.block_size
        blocks = []
        for b in range(n_blocks):
            if b == 0:
                x_block = x_emb[:, 0:s.block_size, :]
                block = torch.cat([z_emb, bos_emb, x_block], dim=1)  # (B, B+2, D)
            else:
                x_block = x_emb[:, b*s.block_size:(b+1)*s.block_size, :]
                block = torch.cat([z_emb, x_block], dim=1)  # (B, B+1, D)
            blocks.append(block)
        inp = torch.cat(blocks, dim=1)  # (B, V42_POS_LEN, D)

        # 5. Add position embedding
        pos_ids = compute_block_positions(B_, T_, s.block_size, device=x.device)
        inp = inp + s.pos(pos_ids)

        # 6. Causal Transformer blocks
        for blk in s.blocks: inp = blk(inp)

        # 7. Get logits for x positions only (skip z and BOS positions)
        # Block 0: logits at positions 2..17 → predict x_0..x_15
        # Block k>0: logits at positions (z_pos+1)..(z_pos+16) → predict x_{Bk}..x_{Bk+15}
        # Simplest: collect logits at the x positions in each block
        logits = s.head(s.ln_f(inp))  # (B, V42_POS_LEN, V)

        x_logits = []
        for b in range(n_blocks):
            if b == 0:
                # x positions are 2..17 (block_size positions)
                x_logits.append(logits[:, 2:2+s.block_size, :])
            else:
                # z position varies; x positions are z_pos+1 .. z_pos+16
                z_pos = 18 + (b - 1) * (s.block_size + 1)
                x_logits.append(logits[:, z_pos+1:z_pos+1+s.block_size, :])
        out = torch.cat(x_logits, dim=1)  # (B, T, V)
        return out


# ============================================================
# AR loss (pure, no diffusion)
# ============================================================
def ar_loss(decoder, z, x, V):
    """L_AR: 标准 next-token CE"""
    logits = decoder(z, x)
    return F.cross_entropy(logits.reshape(-1, V), x.reshape(-1))


# ============================================================
# Warm-start from v25
# ============================================================
P(f"\n=== Warm-start: 加载 v25 decoder ===")
ckpt_v25 = torch.load(V25_CKPT, map_location=DEVICE, weights_only=False)
v25_state = ckpt_v25["decoder"]
P(f"v25 keys: {len(v25_state)}")

decoder = DecoderV42(V, T, D_Z, DEC_LAYER, DEC_HEAD, DEC_EMBD,
                      BOS_ID=BOS_ID, MASK_ID=V-1).to(DEVICE)
P(f"v42 decoder initialized (V={V}, params={sum(p.numel() for p in decoder.parameters())/1e6:.2f}M)")

# 加载 v25 权重, 处理 pos 扩展
v42_state = decoder.state_dict()
loaded = 0; skipped = 0; extended = 0
for k, v in v25_state.items():
    if k not in v42_state:
        skipped += 1; continue
    if v.shape == v42_state[k].shape:
        v42_state[k] = v.clone()
        loaded += 1
    elif "pos.weight" in k:
        # 扩展: v25 pos[0:514] 保持, 新增 pos[514:545] = v25 pos[0:31]
        assert v.shape[0] == V25_POS_LEN, f"unexpected v25 pos shape {v.shape}"
        new_pos = torch.cat([v, v[:V42_POS_LEN - V25_POS_LEN]], dim=0)
        v42_state[k] = new_pos
        extended += 1
    else:
        skipped += 1
        P(f"  skipped {k}: {v.shape} vs {v42_state[k].shape}")

decoder.load_state_dict(v42_state)
P(f"Loaded: {loaded}, Extended: {extended}, Skipped: {skipped}")

# 加载 cached z
cache = np.load(DATA / "cached_v24_z.npz")
train_z_cache = torch.tensor(cache["train_z"], dtype=torch.float32, device=DEVICE)
val_z_cache = torch.tensor(cache["val_z"], dtype=torch.float32, device=DEVICE)
P(f"Loaded z: train {train_z_cache.shape} | val {val_z_cache.shape}")


# ============================================================
# Optimizer + Scheduler
# ============================================================
opt = torch.optim.AdamW(decoder.parameters(), lr=LR, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: (
    step / WARMUP_STEPS if step < WARMUP_STEPS
    else 0.5 * (1 + np.cos(np.pi * min(step - WARMUP_STEPS, STEPS - WARMUP_STEPS) / max(STEPS - WARMUP_STEPS, 1)))
))


# ============================================================
# Training helpers
# ============================================================
def get_batch(texts, B, T):
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
def eval_ppl(decoder, val_texts, val_z, B, T, n_batches):
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
        logits = decoder(z, x)
        loss = F.cross_entropy(logits.reshape(-1, V), x.reshape(-1), reduction='sum')
        total_loss += loss.item(); n_tok += x.numel()
    return float(np.exp(total_loss / n_tok))


# ============================================================
# Training main
# ============================================================
def train_main():
    global decoder, opt, sched
    P(f"\n=== 训练 {STEPS} steps, B={B}, T={T}, LR={LR}, 纯 L_AR ===")
    P(f"(per-block z injection, {N_BLOCKS} blocks, total {V42_POS_LEN} positions)")
    t0 = time.time()
    log = []
    best_ppl = float('inf')

    for step in range(STEPS):
        decoder.train()
        x = get_batch(train_texts, B, T)
        ix = np.random.randint(0, len(train_texts), B)
        z = train_z_cache[torch.tensor(ix, device=DEVICE)]

        # 纯 AR loss (关键: 无 diffusion loss)
        loss = ar_loss(decoder, z, x, V=V)

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        opt.step(); sched.step()

        if step % EVAL_EVERY == 0 or step == STEPS - 1:
            ppl = eval_ppl(decoder, val_texts, val_z_cache, B, T, n_batches=N_VAL_BATCHES_TRAIN)
            elapsed = time.time() - t0
            eta = elapsed / max(step, 1) * (STEPS - step)
            lr_now = opt.param_groups[0]['lr']
            P(f"  step {step:4d}/{STEPS} | L_AR {loss.item():.3f} | val_ppl {ppl:.4f} "
              f"| lr {lr_now:.2e} | {elapsed:.0f}s ETA {eta:.0f}s")
            log.append({"step": step, "loss": loss.item(), "val_ppl": ppl, "lr": lr_now})
            if ppl < best_ppl:
                best_ppl = ppl

    # 保存
    SAVE = V42_DIR / "v42_decoder.pt"
    torch.save({"decoder": decoder.state_dict(),
                "config": {"V": V, "T": T, "D_Z": D_Z,
                           "DEC_LAYER": DEC_LAYER, "DEC_HEAD": DEC_HEAD, "DEC_EMBD": DEC_EMBD,
                           "BLOCK_SIZE": BLOCK_SIZE,
                           "V42_POS_LEN": V42_POS_LEN,
                           "warm_start_from": "v25_decoder",
                           "best_val_ppl": best_ppl,
                           "arch": "v42-per-block-z-injection-poc"}}, SAVE)
    P(f"\nModel saved: {SAVE}")

    LOG = V42_DIR / "v42_train_log.json"
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump({"log": log,
                   "config": {"STEPS": STEPS, "T": T, "D_Z": D_Z, "B": B,
                              "BLOCK_SIZE": BLOCK_SIZE, "LR": LR,
                              "WARMUP_STEPS": WARMUP_STEPS,
                              "n_train": len(train_texts), "n_val": len(val_texts),
                              "decoder_params_M": sum(p.numel() for p in decoder.parameters())/1e6,
                              "best_val_ppl": best_ppl,
                              "n_val_batches_train": N_VAL_BATCHES_TRAIN,
                              "arch": "v42-per-block-z-injection-poc"}}, f, indent=2)
    P(f"Log saved: {LOG}")

    P(f"\n=== 训练完成 ({time.time()-t0:.0f}s, best_val_ppl={best_ppl:.4f}) ===")
    P(f"下一步: python eval_v42.py → PPL 对比 v25 (2.47)")
    return best_ppl


if __name__ == "__main__":
    train_main()